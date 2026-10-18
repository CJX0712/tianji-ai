"""模型组装 —— 天玑 TIANJI 混合架构语言模型。

架构总览（3:1 混合，2026 年业界已收敛的布局）：

    ┌───────────────────────────────────────────────────────────┐
    │ 每 4 层：3 × GatedDeltaNet（线性） + 1 × MLA（全注意力）    │
    │ 每层：Pre-RMSNorm → Mixer → 残差 → Pre-RMSNorm → FFN/MoE   │
    └───────────────────────────────────────────────────────────┘

布局策略（两种，可在基准里直接对比）：
    "interleave" —— Qwen3-Next / Kimi Linear 的做法：每 4 层固定 3:1 交错。
    "edge"       —— Mamba-3 (ICLR 2026) 的发现：把全注意力层集中在网络
                    的**首尾**更有效。理由是「早层捕获全局结构、晚层做输出
                    所需的精确检索」，中间层交给线性递推。
                    本项目把它做成可选项，用基准数据说话。

为什么不用纯注意力或纯线性：
    纯注意力：KV cache 随上下文线性增长，长上下文成本不可接受。
    纯线性：在「精确检索某个先前 token」这类任务上结构性失败
            （S-NIAH-2 上 Gated DeltaNet 92% vs Mamba2 56%）；
            MiniMax M2 曾全砍线性层、后又在 M2.5 回退，是公开的教训。
    混合：让线性层处理「流」，让少数全注意力层处理「检索」。

MLA 为何作为全注意力层的默认：
    DeepSeek-V2/V3 的多头潜在注意力把 KV 压到低秩潜向量（d_c ≪ H·d_h），
    推理时缓存量降一个数量级。混合架构里全注意力层本就少，
    MLA 进一步把这少数几层的缓存压到最小。
    GQA 路径仍然保留（cfg.mla=False），用于消融对比。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .l0_numerics import spawn_seed, rng
from .l1_contract import Ctx
from .l3_primitives import (Linear, RMSNorm, RoPE, SwiGLU, CausalAttention,
                            CrossEntropy, rope_tables, silu)
from .l4_deltanet import GatedDeltaNet, CausalConv1D, gdn_init_state, gdn_step
from .l5_moe import MoELayer, sigmoid

__all__ = ["TianjiConfig", "TianjiModel", "init_params", "build_layer_layout"]

LINEAR = Linear()
RMS = RMSNorm()
ROPE = RoPE()
ATTN = CausalAttention()
GDN = GatedDeltaNet()
CE = CrossEntropy()
SWIGLU = SwiGLU()


@dataclass
class TianjiConfig:
    vocab_size: int = 512
    d_model: int = 96
    n_layers: int = 8
    n_heads: int = 4
    n_kv_heads: int = 2              # GQA（仅 mla=False 时使用）
    hybrid_ratio: int = 3            # 每 (ratio+1) 层中 ratio 层是 GDN
    layout: str = "interleave"       # "interleave" | "edge"
    mla: bool = True                 # 全注意力层用 MLA（否则 GQA + RoPE）
    d_latent_q: int = 48             # MLA query 压缩维
    d_latent_kv: int = 32            # MLA KV 低秩压缩维（远小于 H·d_head）
    d_rope: int = 8                  # MLA 解耦 RoPE 维
    n_experts: int = 4
    moe_top_k: int = 2
    shared_expert: bool = True
    moe_every: int = 2               # 每 N 层的 FFN 用 MoE（其余 dense SwiGLU）
    d_ff: int = 192                  # dense FFN 隐藏维
    d_ff_expert: int = 96            # 每个专家的隐藏维
    conv_kernel: int = 4
    mtp_depth: int = 1               # 多令牌预测（0 = 关闭）
    mtp_lambda: float = 0.1          # MTP 损失权重
    tie_embeddings: bool = True
    max_len: int = 512
    rope_base: float = 10000.0

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"
        return self.d_model // self.n_heads

    @property
    def dim_inner(self) -> int:
        return self.n_heads * self.d_head


def build_layer_layout(n_layers: int, ratio: int = 3, layout: str = "interleave") -> list[str]:
    """返回每层 mixer 类型（"gdn" / "attn"）。"""
    n_attn = max(1, int(round(n_layers / (ratio + 1))))
    if layout == "interleave":
        attn_idx = {(i + 1) * (ratio + 1) - 1 for i in range(n_attn)}
        attn_idx = {i for i in attn_idx if i < n_layers} or {n_layers - 1}
    elif layout == "edge":
        head = (n_attn + 1) // 2
        attn_idx = set(range(head)) | set(range(n_layers - (n_attn - head), n_layers))
        attn_idx = {i for i in attn_idx if i < n_layers}
    else:
        raise ValueError(f"未知 layout: {layout}")
    return ["attn" if i in attn_idx else "gdn" for i in range(n_layers)]


# ---------------------------------------------------------------- 初始化

def init_params(cfg: TianjiConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """参数初始化。

    为什么残差分支的 out_proj 要额外按 1/√(2·n_layers) 缩放：
    GPT-2 起的 residual scaling —— 深层残差堆叠时，若每条分支都按标准方差
    初始化，激活方差会随层数线性累积。缩放后累积方差保持 O(1)，
    这是让深层模型在不依赖长 warmup 也不炸掉的低成本手段。
    """
    g = rng(spawn_seed(seed, "init"))
    d = cfg.d_model
    P: dict[str, np.ndarray] = {}

    def w(shape, std):
        return (g.standard_normal(shape) * std).astype(np.float64)

    resid = 1.0 / np.sqrt(2 * cfg.n_layers)
    P["tok_emb"] = w((cfg.vocab_size, d), 0.02)
    layers = build_layer_layout(cfg.n_layers, cfg.hybrid_ratio, cfg.layout)

    for i, kind in enumerate(layers):
        p = f"L{i}."
        P[p + "norm1_w"] = np.ones(d)
        P[p + "norm2_w"] = np.ones(d)
        if kind == "gdn":
            di = cfg.dim_inner
            P[p + "W_q"] = w((di, d), 0.02)
            P[p + "W_k"] = w((di, d), 0.02)
            P[p + "W_v"] = w((di, d), 0.02)
            P[p + "conv_w"] = w((3 * di, cfg.conv_kernel), 0.02)
            P[p + "W_beta"] = w((cfg.n_heads, d), 0.02)
            P[p + "W_alpha"] = w((cfg.n_heads, d), 0.02)
            P[p + "W_gate"] = w((di, d), 0.02)
            P[p + "out_norm_w"] = np.ones(di)
            P[p + "W_o"] = w((d, di), 0.02 * resid)
        elif cfg.mla:
            H, dh, dr = cfg.n_heads, cfg.d_head, cfg.d_rope
            P[p + "W_dq"] = w((cfg.d_latent_q, d), 0.02)
            P[p + "W_uq"] = w((H * dh, cfg.d_latent_q), 0.02)
            P[p + "W_qr"] = w((H * dr, d), 0.02)
            P[p + "W_dkv"] = w((cfg.d_latent_kv, d), 0.02)
            P[p + "W_uk"] = w((H * dh, cfg.d_latent_kv), 0.02)
            P[p + "W_uv"] = w((H * dh, cfg.d_latent_kv), 0.02)
            P[p + "W_kr"] = w((dr, d), 0.02)
            P[p + "W_o"] = w((d, H * dh), 0.02 * resid)
        else:
            H, dh, Hkv = cfg.n_heads, cfg.d_head, cfg.n_kv_heads
            P[p + "W_q"] = w((H * dh, d), 0.02)
            P[p + "W_k"] = w((Hkv * dh, d), 0.02)
            P[p + "W_v"] = w((Hkv * dh, d), 0.02)
            P[p + "W_o"] = w((d, H * dh), 0.02 * resid)

        if _is_moe_layer(cfg, i):
            E, df = cfg.n_experts, cfg.d_ff_expert
            P[p + "router"] = w((E, d), 0.02)
            P[p + "Wg"] = w((E, df, d), 0.02)
            P[p + "Wu"] = w((E, df, d), 0.02)
            P[p + "Wd"] = w((E, d, df), 0.02 * resid)
            if cfg.shared_expert:
                P[p + "sWg"] = w((df, d), 0.02)
                P[p + "sWu"] = w((df, d), 0.02)
                P[p + "sWd"] = w((d, df), 0.02 * resid)
        else:
            df = cfg.d_ff
            P[p + "Wg"] = w((df, d), 0.02)
            P[p + "Wu"] = w((df, d), 0.02)
            P[p + "Wd"] = w((d, df), 0.02 * resid)

    P["final_norm_w"] = np.ones(d)
    if cfg.mtp_depth > 0:
        P["mtp_norm_w"] = np.ones(d)
        P["mtp_emb_norm_w"] = np.ones(d)
        P["W_mtp_proj"] = w((d, 2 * d), 0.02 * resid)
    if not cfg.tie_embeddings:
        P["lm_head"] = w((cfg.vocab_size, d), 0.02)
    return P


def _is_moe_layer(cfg: TianjiConfig, i: int) -> bool:
    return cfg.n_experts > 1 and (i % cfg.moe_every == (cfg.moe_every - 1))


# ---------------------------------------------------------------- 模型

class TianjiModel:
    def __init__(self, cfg: TianjiConfig, params: dict[str, np.ndarray] | None = None,
                 seed: int = 0):
        self.cfg = cfg
        self.P = params if params is not None else init_params(cfg, seed)
        self.layers = build_layer_layout(cfg.n_layers, cfg.hybrid_ratio, cfg.layout)
        rope_dim = cfg.d_rope if cfg.mla else cfg.d_head
        self.cos, self.sin = rope_tables(rope_dim, cfg.max_len, cfg.rope_base)
        self.moe = MoELayer(cfg.n_experts, cfg.moe_top_k, cfg.shared_expert)
        self.balancers: dict[int, object] = {}
        self._conv = {i: CausalConv1D(cfg.conv_kernel) for i in range(cfg.n_layers)}
        self._rec: list[dict] = []

    def is_moe_layer(self, i: int) -> bool:
        return _is_moe_layer(self.cfg, i)

    def head_w(self) -> np.ndarray:
        return self.P["tok_emb"] if self.cfg.tie_embeddings else self.P["lm_head"]

    def param_count(self) -> int:
        return int(sum(np.asarray(v).size for v in self.P.values()))

    # ============================================================ 前向

    def forward(self, tokens: np.ndarray, targets: np.ndarray | None = None,
                cache: dict | None = None, pos: int = 0):
        cfg, P = self.cfg, self.P
        B, T = tokens.shape
        rec: list[dict] = []
        self._rec = rec
        x = P["tok_emb"][tokens].copy()

        for i, kind in enumerate(self.layers):
            p = f"L{i}."
            h, c1 = self._rms(x, P[p + "norm1_w"])
            if kind == "gdn":
                m, sub = self._gdn_fwd(h, i, p, cache, pos)
            else:
                m, sub = self._attn_fwd(h, i, p, cache, pos)
            xm = x + m
            h2, c2 = self._rms(xm, P[p + "norm2_w"])
            if self.is_moe_layer(i):
                f, sub2 = self._moe_fwd(h2, i, p)
            else:
                f, sub2 = self._ffn_fwd(h2, p)
            rec.append(dict(kind=kind, i=i, p=p, c1=c1, sub=sub, c2=c2, sub2=sub2,
                            moe=self.is_moe_layer(i)))
            x = xm + f

        xf, cf = self._rms(x, P["final_norm_w"])
        W = self.head_w()
        logits = xf @ W.T
        out = {"logits": logits, "xf": xf, "loss": None, "aux": {}}
        self._tail = dict(cf=cf, xf=xf, W=W)

        if cfg.mtp_depth > 0:
            # MTP logits 对**推理期也导出**：投机解码（G14）的 draft 分布依赖它。
            # logits2 本身不依赖 targets（targets 只用于 CE loss）——
            # 旧版把 MTP 计算整个锁在 targets 块内，导致推理期拿不到 draft
            # 分布、投机解码退化为恒接受空壳（已修复，2026-10-19）。
            lm2, cmtp = self._mtp_fwd(xf, tokens, targets)
            out["mtp_logits"] = cmtp["logits2"]
        else:
            cmtp = None

        if targets is not None:
            cce = Ctx()
            lm = float(CE.forward(cce, logits, targets))
            out["aux"]["loss_main"] = lm
            loss = lm
            if cfg.mtp_depth > 0:
                out["aux"]["loss_mtp"] = lm2
                loss = loss + cfg.mtp_lambda * lm2
            self._tail.update(cce=cce, cmtp=cmtp, targets=targets, tokens=tokens)
            out["loss"] = loss
        return out

    def _rms(self, x, w):
        c = Ctx()
        return RMS.forward(c, x, w), c

    def _gdn_fwd(self, h, i, p, cache=None, pos=0):
        cfg, P = self.cfg, self.P
        B, T, d = h.shape
        H, dh, di = cfg.n_heads, cfg.d_head, cfg.dim_inner
        s = {}

        if cache is not None:
            # ========================================================
            # O(1) 递推增量解码（关闭 README「已知边界」第 4 条）
            # 每个新 token 只做一次的秩-1 状态更新，而非整序列重算。
            # 操作序列与 `GatedDeltaNet.forward` 的循环体逐位一致，
            # 故门禁可证「递推增量 ≡ 整序列重算」到机器精度（≈1e-16）。
            # ========================================================
            assert T == 1, "GDN 缓存解码仅支持单 token 步进（自回归生成）"
            K = self._conv[i].K
            # 1) 用最近 K-1 个隐藏态拼出卷积窗口（左边界零填充与全序列重算一致）
            buf = cache.get(f"gdn_h{i}")
            if buf is None:
                buf = np.zeros((B, K - 1, d), dtype=np.float64)
            window = np.concatenate([buf, h], axis=1)          # (B, K, d)
            cache[f"gdn_h{i}"] = window[:, -K + 1:, :]          # 滚动保留最后 K-1 个
            # 2) 投影窗口 → 因果卷积 → 取当前 token 那一行
            qw = LINEAR.forward(s.setdefault("cq", Ctx()), window, P[p + "W_q"])
            kw = LINEAR.forward(s.setdefault("ck", Ctx()), window, P[p + "W_k"])
            vw = LINEAR.forward(s.setdefault("cv", Ctx()), window, P[p + "W_v"])
            qkvw = np.concatenate([qw, kw, vw], axis=-1)
            qkv = self._conv[i].forward(s.setdefault("cc", Ctx()), qkvw,
                                        P[p + "conv_w"])[:, -1:, :]          # (B,1,3di)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, 1, H, dh).transpose(0, 2, 1, 3)
            k = k.reshape(B, 1, H, dh).transpose(0, 2, 1, 3)
            v = v.reshape(B, 1, H, dh).transpose(0, 2, 1, 3)
            beta = sigmoid(LINEAR.forward(s.setdefault("cb", Ctx()), h, P[p + "W_beta"]))
            alpha = sigmoid(LINEAR.forward(s.setdefault("ca", Ctx()), h, P[p + "W_alpha"]))
            s["beta_t"] = np.ascontiguousarray(beta.transpose(0, 2, 1))      # (B,H,1)
            s["alpha_t"] = np.ascontiguousarray(alpha.transpose(0, 2, 1))
            # 3) 递推步进（O(1) 秩-1 更新）
            S = cache.get(f"gdn_S{i}", gdn_init_state(B, H, dh))
            o, S2 = gdn_step(S, q[:, :, 0, :], k[:, :, 0, :], v[:, :, 0, :],
                             s["beta_t"][:, :, 0], s["alpha_t"][:, :, 0])
            cache[f"gdn_S{i}"] = S2
            # 保持与全序列路径一致的 head-major 排列：(B,H,D) -> (B,1,H*D)=di
            o = o.reshape(B, 1, di)                            # (B,H,D) -> (B,1,di)
            on, c_on = self._rms(o, P[p + "out_norm_w"])
            gate_raw = LINEAR.forward(s.setdefault("cga", Ctx()), h, P[p + "W_gate"])
            gate = silu(gate_raw)
            s["on"], s["gate"], s["c_on"], s["gate_raw"] = on, gate, c_on, gate_raw
            y = LINEAR.forward(s.setdefault("co", Ctx()), on * gate, P[p + "W_o"])
            return y, s

        q = LINEAR.forward(s.setdefault("cq", Ctx()), h, P[p + "W_q"])
        k = LINEAR.forward(s.setdefault("ck", Ctx()), h, P[p + "W_k"])
        v = LINEAR.forward(s.setdefault("cv", Ctx()), h, P[p + "W_v"])
        qkv = np.concatenate([q, k, v], axis=-1)
        qkv = self._conv[i].forward(s.setdefault("cc", Ctx()), qkv, P[p + "conv_w"])
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        beta = sigmoid(LINEAR.forward(s.setdefault("cb", Ctx()), h, P[p + "W_beta"]))
        alpha = sigmoid(LINEAR.forward(s.setdefault("ca", Ctx()), h, P[p + "W_alpha"]))
        s["beta_t"] = np.ascontiguousarray(beta.transpose(0, 2, 1))
        s["alpha_t"] = np.ascontiguousarray(alpha.transpose(0, 2, 1))
        o = GDN.forward(s.setdefault("cg", Ctx()), q, k, v, s["beta_t"], s["alpha_t"])
        o = o.transpose(0, 2, 1, 3).reshape(B, T, di)
        on, c_on = self._rms(o, P[p + "out_norm_w"])
        gate_raw = LINEAR.forward(s.setdefault("cga", Ctx()), h, P[p + "W_gate"])
        gate = silu(gate_raw)
        s["on"], s["gate"], s["c_on"], s["gate_raw"] = on, gate, c_on, gate_raw
        y = LINEAR.forward(s.setdefault("co", Ctx()), on * gate, P[p + "W_o"])
        return y, s

    def _attn_fwd(self, h, i, p, cache, pos: int = 0):
        cfg, P = self.cfg, self.P
        B, T, d = h.shape
        H, dh = cfg.n_heads, cfg.d_head
        s = {}
        if cfg.mla:
            dr = cfg.d_rope
            cq = LINEAR.forward(s.setdefault("cdq", Ctx()), h, P[p + "W_dq"])
            qn = LINEAR.forward(s.setdefault("cuq", Ctx()), cq, P[p + "W_uq"])
            qr = LINEAR.forward(s.setdefault("cqr", Ctx()), h, P[p + "W_qr"]).reshape(B, T, H, dr)
            # RoPE 表形状：q 是 (B,T,H,dr) → 需要 (T,1,dr)；k 是 (B,T,dr) → 需要 (T,dr)
            # RoPE 必须用**绝对位置**；增量解码时 T=1 但位置是 pos。
            # 写成 self.cos[:T] 会让每个新 token 都拿到位置 0 的编码 ——
            # 不报错，但长文本质量彻底崩掉，是最隐蔽的一类 cache bug。
            cq4, sq4 = self.cos[pos:pos + T][:, None, :], self.sin[pos:pos + T][:, None, :]
            qr = ROPE.forward(s.setdefault("crq", Ctx()), qr, cq4, sq4)
            q = np.concatenate([qn.reshape(B, T, H, dh), qr], axis=-1)
            ckv = LINEAR.forward(s.setdefault("cdkv", Ctx()), h, P[p + "W_dkv"])
            kn = LINEAR.forward(s.setdefault("cuk", Ctx()), ckv, P[p + "W_uk"])
            vv = LINEAR.forward(s.setdefault("cuv", Ctx()), ckv, P[p + "W_uv"])
            kr = LINEAR.forward(s.setdefault("ckr", Ctx()), h, P[p + "W_kr"])
            kr = ROPE.forward(s.setdefault("crk", Ctx()), kr,
                               self.cos[pos:pos + T, :dr], self.sin[pos:pos + T, :dr])
            kr = np.broadcast_to(kr[:, :, None, :], (B, T, H, dr))
            k = np.concatenate([kn.reshape(B, T, H, dh), kr], axis=-1)
            v = vv.reshape(B, T, H, dh)
        else:
            Hkv = cfg.n_kv_heads
            q = LINEAR.forward(s.setdefault("cq", Ctx()), h, P[p + "W_q"]).reshape(B, T, H, dh)
            k = LINEAR.forward(s.setdefault("ck", Ctx()), h, P[p + "W_k"]).reshape(B, T, Hkv, dh)
            v = LINEAR.forward(s.setdefault("cv", Ctx()), h, P[p + "W_v"]).reshape(B, T, Hkv, dh)
            q = ROPE.forward(s.setdefault("crq", Ctx()), q,
                             cq4 := self.cos[pos:pos + T][:, None, :],
                             self.sin[pos:pos + T][:, None, :])
            k = ROPE.forward(s.setdefault("crk", Ctx()), k, cq4, self.sin[:T][:, None, :])
        qh = np.ascontiguousarray(q.transpose(0, 2, 1, 3))
        kh = np.ascontiguousarray(k.transpose(0, 2, 1, 3))
        vh = np.ascontiguousarray(v.transpose(0, 2, 1, 3))
        if cache is not None:
            if f"k{i}" in cache:
                kh = np.concatenate([cache[f"k{i}"], kh], axis=2)
                vh = np.concatenate([cache[f"v{i}"], vh], axis=2)
            cache[f"k{i}"], cache[f"v{i}"] = kh, vh
        o = ATTN.forward(s.setdefault("ca", Ctx()), qh, kh, vh)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = LINEAR.forward(s.setdefault("co", Ctx()), o, P[p + "W_o"])
        return y, s

    def _moe_fwd(self, h, i, p):
        P = self.P
        B, T, d = h.shape
        x2 = h.reshape(B * T, d)
        bias = self.balancers[i].bias if i in self.balancers else None
        c = Ctx()
        y = self.moe.forward(c, x2, P[p + "router"], P[p + "Wg"], P[p + "Wu"], P[p + "Wd"],
                             bias, P.get(p + "sWg"), P.get(p + "sWu"), P.get(p + "sWd"))
        if i in self.balancers:
            self.balancers[i].update(np.asarray(c["mask"]).sum(axis=0))
        return y.reshape(B, T, d), {"cm": c, "shape": (B, T, d)}

    def _ffn_fwd(self, h, p):
        P = self.P
        c = Ctx()
        return SWIGLU.forward(c, h, P[p + "Wg"], P[p + "Wu"], P[p + "Wd"]), {"cf": c}

    def _mtp_fwd(self, xf, tokens, targets=None):
        """多令牌预测（DeepSeek-V3 MTP 的轻量版）。

        V3 原版：额外堆一个完整 Transformer 层，输入 = RMSNorm(h_t) ⊕ Emb(t_{t+1})。
        本项目版：用单个线性投影替代该层。
        为什么这样取舍：CPU 上多堆一层会让每步训练时间翻倍，而 MTP 的收益
        （更密集的监督信号 + 推理期可当投机解码的 draft）主要来自
        「额外的预测头」本身。这一点在文档中明确标注，不做模糊处理。

        targets=None（推理期）时只返回 logits2，不算 CE loss。
        """
        cfg, P = self.cfg, self.P
        B, T, d = xf.shape
        hn, c_hn = self._rms(xf, P["mtp_norm_w"])
        nxt = np.concatenate([tokens[:, 1:], tokens[:, :1]], axis=1)
        en, c_en = self._rms(P["tok_emb"][nxt], P["mtp_emb_norm_w"])
        hp = LINEAR.forward(c1 := Ctx(), np.concatenate([hn, en], axis=-1), P["W_mtp_proj"])
        logits2 = hp @ self.head_w().T
        c2 = None
        lm = None
        if targets is not None:
            tgt2 = np.concatenate([targets[:, 1:], np.full((B, 1), -1)], axis=1)
            c2 = Ctx()
            lm = float(CE.forward(c2, logits2, tgt2))
        return lm, {"c1": c1, "c2": c2, "c_hn": c_hn, "c_en": c_en,
                    "nxt": nxt, "hp": hp, "logits2": logits2}

    # ============================================================ 反向

    def backward(self, g_logits: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """反传，返回参数梯度字典。

        g_logits 给定时，从「logits 的梯度」起步而不是从交叉熵起步 ——
        这是对齐层（DPO / SimPO / IPO）需要的入口：那些损失不是逐 token
        交叉熵，但仍是 logits 的可微函数，复用同一套骨架即可。
        """
        cfg, P = self.cfg, self.P
        G: dict[str, np.ndarray] = {}
        t = self._tail

        def add(name, g):
            if g is None:
                return
            G[name] = g.copy() if name not in G else G[name] + g

        # ---- MTP 头（g_logits 给定时走对齐路径，跳过 CE / MTP）
        g_xf = np.zeros_like(t["xf"])
        if g_logits is not None:
            W = t["W"]
            add("tok_emb" if cfg.tie_embeddings else "lm_head",
                np.asarray(g_logits).reshape(-1, cfg.vocab_size).T
                @ t["xf"].reshape(-1, cfg.d_model))
            g_xf = g_xf + np.asarray(g_logits) @ W
        elif t.get("cmtp") is not None:
            m = t["cmtp"]
            g_l2, _ = CE.backward(m["c2"], np.float64(cfg.mtp_lambda))
            W = t["W"]
            add("tok_emb" if cfg.tie_embeddings else "lm_head",
                g_l2.reshape(-1, cfg.vocab_size).T @ m["hp"].reshape(-1, cfg.d_model))
            g_hp = g_l2 @ W
            g_cat, g_Wp, _ = LINEAR.backward(m["c1"], g_hp)
            add("W_mtp_proj", g_Wp)
            d = cfg.d_model
            g_hn, g_en = g_cat[..., :d], g_cat[..., d:]
            gx_hn, g_wn = RMS.backward(m["c_hn"], g_hn)
            add("mtp_norm_w", g_wn)
            gx_en, g_we = RMS.backward(m["c_en"], g_en)
            add("mtp_emb_norm_w", g_we)
            g_xf = g_xf + gx_hn
            g_emb = np.zeros_like(P["tok_emb"])
            np.add.at(g_emb, m["nxt"].reshape(-1), gx_en.reshape(-1, d))
            add("tok_emb", g_emb)

        # ---- 主头
        if g_logits is not None:
            g_x, g_fn = RMS.backward(t["cf"], g_xf)
            add("final_norm_w", g_fn)
        else:
            g_l1, _ = CE.backward(t["cce"], np.float64(1.0))
            W = t["W"]
            add("tok_emb" if cfg.tie_embeddings else "lm_head",
                g_l1.reshape(-1, cfg.vocab_size).T @ t["xf"].reshape(-1, cfg.d_model))
            g_xf = g_xf + g_l1 @ W
            g_x, g_fn = RMS.backward(t["cf"], g_xf)
            add("final_norm_w", g_fn)
        self._dbg = {"g_xf": g_xf, "g_x": g_x, "cf": t["cf"]}

        # ---- 逐层逆序（Pre-Norm 的反传顺序极易写反，这里显式展开）
        #
        # 前向：  h = Norm(x) → m = Mixer(h) → xm = x + m
        #         h2 = Norm(xm) → f = FFN(h2) → x_out = xm + f
        #
        # 反传必须**先过子层、再过归一化**：
        #         g_f = g_out ； g_h2 = FFNᵇ(g_f) ； (g_xm, g_w2) = Norm2ᵇ(g_h2)
        #         g_xm += g_out（残差直通）
        #         g_m = g_xm  ； g_h = Mixerᵇ(g_m) ； (g_x, g_w1) = Norm1ᵇ(g_h)
        #         g_x += g_xm（残差直通）
        # 早期版本把 Norm 的 backward 放在子层之前，等价于把残差流的梯度
        # 当成归一化层输出的梯度，梯度全错但形状全对 —— 是最阴险的一类 bug：
        # 不会崩、会「训练」，只是永远训不好。
        for r in reversed(self._rec):
            i, p = r["i"], r["p"]
            # ---- FFN 分支
            if r["moe"]:
                g_h2, gd = self._moe_bwd(i, p, r["sub2"], g_x)
            else:
                g_h2, gd = self._ffn_bwd(p, r["sub2"], g_x)
            for k, v in gd.items():
                add(p + k, v)
            g_xm_norm, g_w2 = RMS.backward(r["c2"], g_h2)
            add(p + "norm2_w", g_w2)
            g_xm = g_x + g_xm_norm
            # ---- Mixer 分支
            if r["kind"] == "gdn":
                g_h, gd = self._gdn_bwd(i, p, r["sub"], g_xm)
            else:
                g_h, gd = self._attn_bwd(i, p, r["sub"], g_xm)
            for k, v in gd.items():
                add(p + k, v)
            g_x_norm, g_w1 = RMS.backward(r["c1"], g_h)
            add(p + "norm1_w", g_w1)
            g_x = g_xm + g_x_norm
        self._dbg = {"g_x": g_x}

        # ---- embedding gather 的反向（scatter-add）
        g_emb = np.zeros_like(P["tok_emb"])
        np.add.at(g_emb, self._tail["tokens"].reshape(-1), g_x.reshape(-1, cfg.d_model))
        add("tok_emb", g_emb)
        return G

    def _gdn_bwd(self, i, p, s, gy):
        cfg = self.cfg
        H, dh, di = cfg.n_heads, cfg.d_head, cfg.dim_inner
        B, T = gy.shape[0], gy.shape[1]
        g_on_gate, g_Wo, _ = LINEAR.backward(s["co"], gy)
        g_on = g_on_gate * s["gate"]
        g_gate_raw = g_on_gate * s["on"]
        gr = s["gate_raw"]
        sg = 1.0 / (1.0 + np.exp(-np.clip(gr, -60.0, 60.0)))
        dsilu = sg * (1.0 + gr * (1.0 - sg))            # SiLU'(z) = σ(z)(1 + z(1−σ(z)))
        g_gate_lin, g_Wgate, _ = LINEAR.backward(s["cga"], g_gate_raw * dsilu)
        g_o, g_onw = RMS.backward(s["c_on"], g_on)
        go = np.ascontiguousarray(g_o.reshape(B, T, H, dh).transpose(0, 2, 1, 3))
        dq, dk, dv, db, da = GDN.backward(s["cg"], go)
        # beta/alpha 在前向是 (B,T,H) 经 sigmoid 后转置成 (B,H,T)；此处转回来
        bt = s["beta_t"].transpose(0, 2, 1)
        at = s["alpha_t"].transpose(0, 2, 1)
        db = np.ascontiguousarray(db.transpose(0, 2, 1)) * bt * (1.0 - bt)
        da = np.ascontiguousarray(da.transpose(0, 2, 1)) * at * (1.0 - at)
        g_beta_lin, g_Wbeta, _ = LINEAR.backward(s["cb"], db)
        g_alpha_lin, g_Walpha, _ = LINEAR.backward(s["ca"], da)
        # 反 reshape / 反短卷积
        dq = np.ascontiguousarray(dq.transpose(0, 2, 1, 3)).reshape(B, T, di)
        dk = np.ascontiguousarray(dk.transpose(0, 2, 1, 3)).reshape(B, T, di)
        dv = np.ascontiguousarray(dv.transpose(0, 2, 1, 3)).reshape(B, T, di)
        g_qkv = np.concatenate([dq, dk, dv], axis=-1)
        g_qkv_pre, g_conv = self._conv[i].backward(s["cc"], g_qkv)
        gq, gk, gv = np.split(g_qkv_pre, 3, axis=-1)
        gx_q, g_Wq, _ = LINEAR.backward(s["cq"], gq)
        gx_k, g_Wk, _ = LINEAR.backward(s["ck"], gk)
        gx_v, g_Wv, _ = LINEAR.backward(s["cv"], gv)
        gh = gx_q + gx_k + gx_v + g_gate_lin + g_beta_lin + g_alpha_lin
        return gh, {"W_q": g_Wq, "W_k": g_Wk, "W_v": g_Wv, "conv_w": g_conv,
                    "W_beta": g_Wbeta, "W_alpha": g_Walpha, "W_gate": g_Wgate,
                    "out_norm_w": g_onw, "W_o": g_Wo}

    def _attn_bwd(self, i, p, s, gy):
        cfg = self.cfg
        B, T = gy.shape[0], gy.shape[1]
        H, dh = cfg.n_heads, cfg.d_head
        g_o, g_Wo, _ = LINEAR.backward(s["co"], gy)
        dO = np.ascontiguousarray(g_o.reshape(B, T, H, dh).transpose(0, 2, 1, 3))
        dq, dk, dv = ATTN.backward(s["ca"], dO)
        gd = {"W_o": g_Wo}
        if cfg.mla:
            H, dh, dr = cfg.n_heads, cfg.d_head, cfg.d_rope
            dq = np.ascontiguousarray(dq.transpose(0, 2, 1, 3))       # (B,T,H,dh+dr)
            dk = np.ascontiguousarray(dk.transpose(0, 2, 1, 3))
            dv = np.ascontiguousarray(dv.transpose(0, 2, 1, 3))
            d_qn, d_qr = dq[..., :dh], dq[..., dh:]
            d_kn, d_kr = dk[..., :dh], dk[..., dh:]
            # k_rope 是所有头共享的（前向做了 broadcast），反向需按头求和
            d_kr = d_kr.sum(axis=2)
            d_kr = ROPE.backward(s["crk"], d_kr)[0]
            g_kr, g_Wkr, _ = LINEAR.backward(s["ckr"], d_kr)
            d_qr = ROPE.backward(s["crq"], d_qr)[0]
            g_qr, g_Wqr, _ = LINEAR.backward(s["cqr"], d_qr.reshape(B, -1, H * dr))
            g_kn, g_Wuk, _ = LINEAR.backward(s["cuk"], d_kn.reshape(B, -1, H * dh))
            g_vv, g_Wuv, _ = LINEAR.backward(s["cuv"], dv.reshape(B, -1, H * dh))
            g_ckv = g_kn + g_vv
            g_kv_in, g_Wdkv, _ = LINEAR.backward(s["cdkv"], g_ckv)
            g_qn_in, g_Wuq, _ = LINEAR.backward(s["cuq"], d_qn.reshape(B, -1, H * dh))
            g_cq, g_Wdq, _ = LINEAR.backward(s["cdq"], g_qn_in)
            gd.update({"W_uq": g_Wuq, "W_dq": g_Wdq, "W_qr": g_Wqr, "W_kr": g_Wkr,
                       "W_uk": g_Wuk, "W_uv": g_Wuv, "W_dkv": g_Wdkv})
            gh = g_cq + g_qr + g_kv_in + g_kr
        else:
            H, dh = cfg.n_heads, cfg.d_head
            # RoPE 作用在 4 维 (B,T,H,dh) 上，必须先还原再压平 ——
            # 直接对 (B,T,H·dh) 施加旋转会把「头维」误当成特征维。
            dq4 = np.ascontiguousarray(dq.transpose(0, 2, 1, 3)).reshape(B, T, H, dh)
            dk4 = np.ascontiguousarray(dk.transpose(0, 2, 1, 3)).reshape(B, T, cfg.n_kv_heads, dh)
            dv4 = np.ascontiguousarray(dv.transpose(0, 2, 1, 3)).reshape(B, T, cfg.n_kv_heads, dh)
            dq = ROPE.backward(s["crq"], dq4)[0].reshape(B, -1, H * dh)
            dk = ROPE.backward(s["crk"], dk4)[0].reshape(B, -1, cfg.n_kv_heads * dh)
            dv = dv4.reshape(B, -1, cfg.n_kv_heads * dh)
            gq, g_Wq, _ = LINEAR.backward(s["cq"], dq)
            gk, g_Wk, _ = LINEAR.backward(s["ck"], dk)
            gv, g_Wv, _ = LINEAR.backward(s["cv"], dv)
            gd.update({"W_q": g_Wq, "W_k": g_Wk, "W_v": g_Wv})
            gh = gq + gk + gv
        return gh, gd

    def _moe_bwd(self, i, p, s, gy):
        P = self.P
        B, T, d = s["shape"]
        c = s["cm"]
        dx2, drouter, dWg, dWu, dWd, _, dsWg, dsWu, dsWd = self.moe.backward(c, gy.reshape(B * T, d))
        gd = {"router": drouter, "Wg": dWg, "Wu": dWu, "Wd": dWd}
        if dsWg is not None:
            gd.update({"sWg": dsWg, "sWu": dsWu, "sWd": dsWd})
        return dx2.reshape(B, T, d), gd

    def _ffn_bwd(self, p, s, gy):
        gx, gWg, gWu, gWd = SWIGLU.backward(s["cf"], gy)
        return gx, {"Wg": gWg, "Wu": gWu, "Wd": gWd}
