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
    I4 投机解码无损：**期望输出分布与逐一解码一致**（G14 机检：解析恒等 + 统计端到端）
    I5 投机解码有效：接受率 ∈ (0,1)（draft≠target 时确实在 draft，G14 机检）；
                    加速比在本后端结构性 ≈1 —— draft 搭车主干 forward（MTP 头依赖
                    主干 hidden），每验证 1 token 恰需 1 次 forward；真实加速（论文的
                    1+α）需要独立轻量 draft + 一次 batch 并行验证（GPU 场景）。
                    **不虚报加速，如实机检无损性**（宣称必有门禁守护）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .l0_numerics import softmax, spawn_seed, rng

__all__ = ["quantize_per_channel", "quantize_groupwise", "dequantize",
           "sample_top_p", "generate", "speculative_decode", "SpecStats"]


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
        """平均每验证步产出的 token 数。

        本后端（draft 搭车主干 forward）结构性 ≈1：每验证 1 个 draft 恰需
        1 次主干 forward。真实加速（论文的 1+接受率）需要独立轻量 draft
        模型 + 一次 batch 并行验证 γ 个 token（GPU 场景）—— 详见 G14 门禁
        的诚实边界说明。保留此字段以兼容 bench，值如实反映 ≈1。
        """
        return (self.accepted + self.steps - self.accepted) / max(self.steps, 1)


def speculative_decode(model, prompt: np.ndarray, max_new: int = 32, *,
                       gamma: int = 1, temperature: float = 1.0, top_p: float = 0.9,
                       seed: int = 0) -> tuple[list[int], SpecStats]:
    """用 MTP 头做 draft、主模型逐位拒绝采样验证的投机解码。

    站在谁之上（零重写）：
        Leviathan et al. 2023 (ICML) / Chen et al. 2023 的 draft-then-verify
        拒绝采样；DeepSeek-V3 把 MTP 头当 draft 使用。

    无损性（G14 机检，这是投机解码的核心定理）：
        每步从 draft 分布 q 采样 xd，以 min(1, p(xd)/q(xd)) 的概率接受；
        拒绝时从残差分布 ∝ max(0, p−q) 重采样。由恒等式
            Σ_x min(p(x), q(x)) = 1 − Σ_x max(p(x)−q(x), 0)
        可得单步输出分布**逐点等于** p —— 无论 draft 多差都严格无损：
            P(out=x) = q(x)·min(1, p/q) + P(拒绝)·residual(x) = p(x)。

    诚实的加速边界（宣称必有门禁守护）：
        本实现的 draft 分布搭车主模型 forward（MTP 头依赖主干 hidden），
        每验证 1 个 token 恰需 1 次主干 forward，加速比结构性 ≈1。
        论文的 1+α 加速需要「独立轻量 draft 模型 + 一次 batch 并行验证
        γ 个 draft」（GPU 场景 batch 收益）。本后端（CPU+NumPy，forward
        成本 ∝ 序列长）下 batch 无收益，因此只兑现无损性并机检之，不虚报。

    gamma 说明：
        cfg.mtp_depth=1（MTP 单步预测）→ 有效 γ=1；更大的 gamma 被夹到 1。
    """
    if model.cfg.mtp_depth <= 0:
        ids = generate(model, prompt, max_new, temperature=temperature, top_p=top_p, seed=seed)
        return ids, SpecStats()

    ids = list(np.asarray(prompt).reshape(-1).tolist())
    st = SpecStats()
    g = np.random.default_rng(spawn_seed(seed, "spec"))
    T = temperature if temperature else 1.0
    prompt_len = len(ids)
    produced = 0
    while produced < max_new and st.steps < max_new * 4:    # 安全阀防死循环
        out = model.forward(np.asarray([ids], dtype=np.int64))
        p = softmax(out["logits"][0, -1] / T)               # 验证分布（target）
        q = softmax(out["mtp_logits"][0, -1] / T)           # draft 分布（MTP 头）
        st.steps += 1
        xd = int(g.choice(len(q), p=q))
        st.drafted += 1
        if g.random() < min(1.0, p[xd] / max(q[xd], 1e-300)):
            ids.append(xd)
            produced += 1
            st.accepted += 1
        else:
            # 拒绝：从残差分布 ∝ max(0, p−q) 重采样（无损性的另一半）
            resid = np.maximum(p - q, 0.0)
            s = resid.sum()
            resid = p if s <= 1e-12 else resid / s          # p==q 时恒接受，此为数值保护
            ids.append(int(g.choice(len(resid), p=resid)))
            produced += 1
    return ids[:prompt_len + max_new], st
