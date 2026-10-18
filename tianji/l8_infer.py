"""L8 · 推理 —— KV/递推状态缓存、量化、投机解码。

站在谁之上（零重写）：
    - KV cache 增量解码            —— 所有自回归推理引擎的标配
    - 后训练量化（per-channel INT8）—— llama.cpp / GGML 的 Q8_0 方案：
                                     逐通道缩放，比逐张量量化误差小一个量级
    - 分组 INT4（group-wise）       —— AWQ (Lin et al. 2024) 与 GPTQ 的共识：
                                     「权重的重要性不一致」，按列分组给 scale
                                     能在 4bit 下保住质量
    - 投机解码                      —— Leviathan et al. 2023 / Chen et al. 2023；
                                     DeepSeek-V3 直接把 MTP 头当 draft 模型使用
    - Top-p / Top-k 采样            —— Holtzman et al. 2020 / Fan et al. 2018

本层的取舍原则（明确写在文档里）：
    MLA 推理时，理论上只缓存低秩潜向量 c_kv（DeepSeek 的做法）。
    本项目在 CPU + NumPy 后端下**选择缓存展开后的 K/V**：
    潜向量方案每步都要做 (T,H·d) 规模的展开矩阵乘，在 CPU 上反而更慢，
    而内存不是本项目的瓶颈。这是「按后端选择实现」而非「照抄论文」。

不变量（门禁复检）：
    I1 KV cache 等价：增量解码的 logits == 整序列重算的 logits（1e-10）
    I2 量化误差有界：INT8 的相对 Frobenius 误差 < 2^-7 量级；INT4 < 2^-3 量级
    I3 量化单调性：INT8 的困惑度退化 <= INT4 的困惑度退化
    I4 投机解码无损：**期望输出分布与逐一解码一致**（G14 机检：解析恒等 + 统计端到端；
                    G15 机检：γ>1 并行验证的位置指纹 + 统计 + 代数条件）
    I5 投机解码有效：接受率 ∈ (0,1)（draft≠target 时确实在 draft，G14/G15 机检）
    I6 投机解码加速：独立轻量 draft（TinyDraftGDN）+ γ>1 一次 batch 并行验证时，
                    每轮主干 forward 恰 1 次、产出 γ+接受数 ≥ 2 token ——
                    tokens/forward > 1 的代数必要条件由 G15 机检；
                    draft 递推 O(d²) 复用已门禁的 gdn_step（G11/G12/G13）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .l0_numerics import softmax, spawn_seed, rng
from .l4_deltanet import gdn_init_state, gdn_step, l2norm

__all__ = ["quantize_per_channel", "quantize_groupwise", "dequantize",
           "sample_top_p", "generate", "speculative_decode", "SpecStats",
           "TinyDraftGDN"]


# ---------------------------------------------------------------- 量化

def quantize_per_channel(W: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """逐通道（输出通道）对称量化。

    W (out, in) → 每个 out 通道一个 scale。
    为什么逐通道：同一输出通道内的量级相近，而跨通道可能差 10 倍以上；
    逐张量量化会让小量级通道被压成 0。
    """
    W = np.asarray(W, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(W), axis=-1, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    Wq = np.round(W / scale)
    return np.clip(Wq, -qmax - 1, qmax).astype(np.int32), scale.squeeze(-1)


def quantize_groupwise(W: np.ndarray, bits: int = 4, group: int = 32) -> tuple[np.ndarray, np.ndarray, int]:
    """分组量化：沿输入维每 group 个元素共享一个 scale（AWQ/GPTQ 的做法）。

    为什么 INT4 必须分组：4bit 只有 16 个电平，逐通道的 scale 不足以
    刻画通道内的量级变化；分组把动态范围再切细，是 4bit 可用的关键。
    """
    W = np.asarray(W, dtype=np.float64)
    out, in_ = W.shape
    # 分组大小不得超过输入维（否则 pad 会撑大矩阵）
    group = int(min(group, max(in_, 1)))
    # 必须整除，且**不能靠 pad 解决**：padding 会让 dequantize 的输出形状
    # 与原始权重不一致（例如 in_=48 / group=32 会 pad 到 64，
    # 回灌进模型后与 RMSNorm 增益形状对不上）。
    # 因此改为把 group 下调到「不超过设定值且能整除 in_ 的最大值」。
    if in_ % group != 0:
        cands = [g for g in range(1, min(group, in_) + 1) if in_ % g == 0]
        group = max(cands) if cands else in_
    qmax = 2 ** (bits - 1) - 1
    Wg = W.reshape(out, in_ // group, group)
    scale = np.max(np.abs(Wg), axis=-1) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    Wq = np.round(Wg / scale[..., None])
    Wq = np.clip(Wq, -qmax - 1, qmax).astype(np.int32).reshape(out, in_)
    return Wq, scale, group


def dequantize(Wq: np.ndarray, scale: np.ndarray, group: int | None = None) -> np.ndarray:
    Wq = np.asarray(Wq, dtype=np.float64)
    if group is None:
        return Wq * np.asarray(scale)[:, None]
    out, in_ = Wq.shape
    Wg = Wq.reshape(out, in_ // group, group)
    return (Wg * np.asarray(scale)[..., None]).reshape(out, in_)


# ---------------------------------------------------------------- 采样

def sample_top_p(logits: np.ndarray, top_p: float = 0.9, temperature: float = 1.0,
                 seed: int | None = None) -> int:
    """Top-p（nucleus）采样。

    为什么用 top-p 而不是 top-k：top-k 的固定 k 在不同分布形状下
    要么截断过多（分布尖锐时把长尾的合理候选切掉），要么保留过多噪声
    （分布平坦时）。top-p 按累积概率自适应。
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature and temperature != 1.0:
        logits = logits / temperature
    p = softmax(logits)
    order = np.argsort(-p)
    csum = np.cumsum(p[order])
    keep = order[csum <= top_p]
    if keep.size == 0:
        keep = order[:1]
    keep_p = p[keep] / p[keep].sum()
    g = rng(spawn_seed(seed if seed is not None else 0, "sample"))
    return int(g.choice(keep, p=keep_p))


# ---------------------------------------------------------------- 生成

def generate(model, prompt: np.ndarray, max_new: int = 32, *,
             temperature: float = 1.0, top_p: float = 0.9, seed: int = 0,
             use_cache: bool = True) -> list[int]:
    """自回归生成。

    use_cache=True 时走**真正的 O(1) 递推增量解码**：先把整段 prompt 逐
    token 送入以建立 GDN 递推状态 S 与 MLA 的 KV cache（prefill），随后每个
    新 token 只做一步秩-1 状态更新 + 一次 KV 追加，复杂度与上下文长度无关。
    这与门禁 G11 验证的「递推增量 ≡ 整序列重算」是同一套路径。
    """
    ids = list(np.asarray(prompt).reshape(-1).tolist())
    cache = {} if use_cache else None
    if use_cache:
        # ---- prefill：逐 token 建立递推/KV 状态（位置 0..L-1） ----
        out = None
        for t in range(len(ids)):
            out = model.forward(np.asarray([[ids[t]]], dtype=np.int64),
                                cache=cache, pos=t)
        # `out` 已是位置 L 的 logits（第一个待生成 token 的分布）
        nxt = sample_top_p(out["logits"][0, -1], top_p=top_p,
                           temperature=temperature, seed=seed + len(ids))
        ids.append(nxt)
        for _ in range(max_new - 1):
            pos = len(ids) - 1                      # 新 token 的绝对位置
            out = model.forward(np.asarray([[ids[pos]]], dtype=np.int64),
                                cache=cache, pos=pos)
            nxt = sample_top_p(out["logits"][0, -1], top_p=top_p,
                               temperature=temperature, seed=seed + len(ids))
            ids.append(nxt)
        return ids
    # ---- 不使用缓存：整序列重算（结果严格一致） ----
    for _ in range(max_new):
        x = np.asarray([ids], dtype=np.int64)
        out = model.forward(x)
        nxt = sample_top_p(out["logits"][0, -1], top_p=top_p,
                           temperature=temperature, seed=seed + len(ids))
        ids.append(nxt)
    return ids


# ---------------------------------------------------------------- 投机解码

@dataclass
class SpecStats:
    drafted: int = 0
    accepted: int = 0
    steps: int = 0

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(self.drafted, 1)

    @property
    def speedup(self) -> float:
        """平均每验证步（= 每次主干 forward）产出的 token 数。

        MTP 搭车路径（draft_model=None）：结构性 ≈1（每验证 1 token 恰需
        1 次 forward）。独立 draft 路径（TinyDraftGDN + γ>1）：每轮 1 次
        主干 forward 产出 γ_draft 个 token（全接受时 γ+1），结构收益 ≥1、
        期望 1+α —— G15 机检代数条件 tokens/forward ≥ 1。
        """
        return self.produced / max(self.steps, 1)

    @property
    def produced(self) -> int:
        """产出的 token 总数（接受 + 拒绝重采样 + 全接受 bonus）。"""
        # drafted 计 draft 次数；steps 计验证轮数。产出 = drafted + bonus 轮数
        # 但直接跟踪最稳：accepted + (steps - accepted_accept_rounds) ...
        # 简化：每轮产出 ≥1 且 = drafted_in_round + (1 若 bonus/resample)。
        # 由于 drafted 恰为每 draft 的 token 数、每轮恰 1 个收尾 token
        # （重采样或 bonus），产出 = drafted + steps_with_finish。
        return self._produced

    def __post_init__(self):
        self._produced = 0


class TinyDraftGDN:
    """独立轻量 draft 模型：嵌入 → 单层 GDN 递推 → vocab 头。

    零重写（集大成）：
        递推算子直接复用已门禁的 `gdn_step`（G11/G12/G13 机检到机器精度）
        与 `l2norm`；不含 attention/MoE，参数独立、O(d²)/步。
    正确性与质量解耦（拒绝采样的美）：
        draft 分布可以是任意分布 —— 质量只影响接受率 α，绝不影响输出
        分布（G14/G15 机检无损性）。因此参数可随机初始化，无需训练。
    状态管理：
        `step(token)` = 「token 确认进入上下文，返回下一个 token 的 logits」。
        draft 期间临时推进的状态用快照回滚（见 speculative_decode）。
    """

    def __init__(self, vocab_size: int, d: int = 32, seed: int = 0):
        g = rng(spawn_seed(seed, "draft"))
        scale = 1.0 / np.sqrt(d)
        self.V = int(vocab_size)
        self.d = d
        self.W_emb = g.standard_normal((self.V, d)) * scale
        self.W_q = g.standard_normal((d, d)) * scale
        self.W_k = g.standard_normal((d, d)) * scale
        self.W_v = g.standard_normal((d, d)) * scale
        self.W_out = g.standard_normal((self.V, d)) * scale
        self.S = gdn_init_state(1, 1, d)
        self.last_logits = np.zeros(self.V)

    def warmup(self, tokens) -> None:
        """把确认的上下文逐 token 喂入递推状态（prefill）。"""
        for t in np.asarray(tokens).reshape(-1).tolist():
            self.last_logits = self.step(t)

    def step(self, token: int) -> np.ndarray:
        """token 确认进入上下文 → 返回下一个 token 的 logits（不 softmax）。"""
        h = self.W_emb[int(token)]
        q = l2norm(h @ self.W_q)[None, None, :]
        k = l2norm(h @ self.W_k)[None, None, :]      # gdn_step 内部亦会归一化
        v = (h @ self.W_v)[None, None, :]
        o, self.S = gdn_step(self.S, q, k, v, 0.9, 0.9)
        return o[0, 0] @ self.W_out.T


def speculative_decode(model, prompt: np.ndarray, max_new: int = 32, *,
                       gamma: int = 1, temperature: float = 1.0, top_p: float = 0.9,
                       seed: int = 0,
                       draft_model: "TinyDraftGDN | None" = None
                       ) -> tuple[list[int], SpecStats]:
    """draft-then-verify 投机解码（Leviathan et al. 2023 / Chen et al. 2023）。

    无损性（G14/G15 机检，这是投机解码的核心定理）：
        从 draft 分布 q 采样，以 min(1, p(x)/q(x)) 接受；拒绝时从残差分布
        ∝ max(0, p−q) 重采样。由恒等式
            Σ_x min(p(x), q(x)) = 1 − Σ_x max(p(x)−q(x), 0)
        可得单步输出分布**逐点等于** p —— 无论 draft 多差都严格无损。

    两条路径：
        draft_model=None（MTP 搭车）：draft 分布来自主模型 forward 顺带返回的
            MTP 头 logits，每验证 1 token 恰需 1 次 forward，γ 夹到 1
            （cfg.mtp_depth=1 为单步预测）。G14 机检。
        draft_model=TinyDraftGDN（独立轻量 draft）：γ>1 生效 —— draft 递推
            O(d²)/步（复用已门禁 gdn_step），然后**一次**主干 forward 并行
            验证全部 γ 个 draft（logits 的 γ+1 个位置给出 p_1..p_{γ+1}）：
            逐位接受；第 k 位拒绝则从 residual(p_k, q_k) 重采样并丢弃后续
            draft；全接受则 bonus 直接从 p_{γ+1} 采样（精确服从 target，
            无需验证）。每轮恰 1 次主干 forward、产出 ≥1（全接受 γ+1）。
            G15 机检（位置指纹 + 统计 + 代数条件）。

    状态同步（实现最易错处，被 G15 指纹测试钉死）：
        draft 临时推进的状态逐步快照 S_0..S_γ；接受位 i → 回滚到 S_i 并
        把 x_i 的 step logits 留作下轮 draft 分布；拒绝位 k → 回滚到 S_k
        （只含被接受的 x_1..x_{k-1}）并把重采样 token y 确认喂入；全接受
        → 状态停在 S_γ，bonus b 确认喂入。
    """
    ids = list(np.asarray(prompt).reshape(-1).tolist())
    st = SpecStats()
    g = np.random.default_rng(spawn_seed(seed, "spec"))
    T = temperature if temperature else 1.0
    prompt_len = len(ids)

    if draft_model is None:
        # ---------------- MTP 搭车路径（γ=1，G14 机检） ----------------
        if model.cfg.mtp_depth <= 0:
            ids = generate(model, prompt, max_new, temperature=temperature,
                           top_p=top_p, seed=seed)
            return ids, st
        produced = 0
        while produced < max_new and st.steps < max_new * 4:
            out = model.forward(np.asarray([ids], dtype=np.int64))
            p = softmax(out["logits"][0, -1] / T)           # 验证分布（target）
            q = softmax(out["mtp_logits"][0, -1] / T)       # draft 分布（MTP 头）
            st.steps += 1
            xd = int(g.choice(len(q), p=q))
            st.drafted += 1
            st._produced += 1
            if g.random() < min(1.0, p[xd] / max(q[xd], 1e-300)):
                ids.append(xd)
                produced += 1
                st.accepted += 1
            else:
                resid = np.maximum(p - q, 0.0)
                s = resid.sum()
                resid = p if s <= 1e-12 else resid / s      # p==q 恒接受，数值保护
                ids.append(int(g.choice(len(resid), p=resid)))
                produced += 1
        return ids[:prompt_len + max_new], st

    # ---------------- 独立 draft 路径（γ>1，G15 机检） ----------------
    dm = draft_model
    gamma_eff = max(1, int(gamma))
    dm.warmup(ids)
    produced = 0
    while produced < max_new and st.steps < max_new * 4:
        # ---- 1) draft：γ 个 token 逐步递推（状态快照 S_0..S_γ）
        drafts: list[int] = []
        dprobs: list[np.ndarray] = []
        Ssnaps: list[np.ndarray] = [dm.S.copy()]
        step_logits: list[np.ndarray] = []
        q_cur = softmax(dm.last_logits / T)
        for _ in range(gamma_eff):
            x_i = int(g.choice(len(q_cur), p=q_cur))
            drafts.append(x_i)
            dprobs.append(q_cur)
            sl = dm.step(x_i)                               # x_i 暂进 draft 状态
            step_logits.append(sl)
            Ssnaps.append(dm.S.copy())
            q_cur = softmax(sl / T)
        # ---- 2) verify：一次主干 forward 并行验证全部 γ 个 draft
        out = model.forward(np.asarray([ids + drafts], dtype=np.int64))
        L0 = len(ids)
        # logits[t] 是对位置 t+1 的预测 → p_i = logits[L0-1+i]（i=1..γ+1）
        p_list = [softmax(out["logits"][0, L0 - 1 + i] / T)
                  for i in range(gamma_eff + 1)]
        st.steps += 1
        # ---- 3) 逐位接受 / 拒绝重采样 / 全接受 bonus
        all_accepted = True
        for i in range(gamma_eff):
            if produced >= max_new:                         # 已产满：剩余 draft 丢弃
                break
            st.drafted += 1
            st._produced += 1
            x_i, q_i, p_i = drafts[i], dprobs[i], p_list[i]
            if g.random() < min(1.0, p_i[x_i] / max(q_i[x_i], 1e-300)):
                ids.append(x_i)
                produced += 1
                st.accepted += 1
                dm.S = Ssnaps[i + 1]                        # 含 x_1..x_i
                dm.last_logits = step_logits[i]
            else:
                all_accepted = False
                resid = np.maximum(p_i - q_i, 0.0)
                s = resid.sum()
                resid = p_i if s <= 1e-12 else resid / s
                y = int(g.choice(len(resid), p=resid))
                ids.append(y)
                produced += 1
                dm.S = Ssnaps[i]                            # 只含被接受的 x_1..x_{i-1}
                dm.last_logits = dm.step(y)                 # y 确认进上下文
                break
        if all_accepted and produced < max_new:
            # bonus：直接从 p_{γ+1} 采样 —— 精确服从 target，无需验证
            b = int(g.choice(len(p_list[-1]), p=p_list[-1]))
            st._produced += 1
            ids.append(b)
            produced += 1
            dm.last_logits = dm.step(b)                     # b 确认（状态已含 x_1..x_γ）
    return ids[:prompt_len + max_new], st
