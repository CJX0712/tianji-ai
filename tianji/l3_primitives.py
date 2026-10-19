"""L3 · 架构原语 —— 现代 Transformer 的原子可微模块。

站在谁之上（零重写，每一项都标注来源与替代方案对比）：

┌──────────────┬───────────────────────────────────────┬──────────────────────────────┐
│ 模块         │ 采用的世界顶级方案                     │ 被否决的替代方案             │
├──────────────┼───────────────────────────────────────┼──────────────────────────────┤
│ RMSNorm      │ Zhang & Sennrich 2019 (Root Mean      │ LayerNorm：需减均值+偏置，    │
│              │ Square Layer Normalization)           │ 参数更多、推理更慢，且实测    │
│              │                                       │ 在大模型上无质量收益          │
│ RoPE         │ Su et al. 2021 (RoFormer)，HF         │ ALiBi：长上下文外推弱于 RoPE  │
│              │ rotate_half 分裂布局                   │ + YaRN；绝对位置编码：外推    │
│              │                                       │ 能力差，已被全面淘汰          │
│ SwiGLU       │ Shazeer 2020 (GLU Variants)，         │ ReLU/GELU FFN：同等参数下     │
│              │ LLaMA/PaLM 全系采用                    │ perplexity 明显更差           │
│ GQA          │ Ainslie et al. 2023 (GQA)，           │ MHA：KV cache 太大；MQA：     │
│              │ LLaMA-2/3、Qwen 采用                   │ 质量损失可测                  │
│ MLA          │ DeepSeek-V2/V3 多头潜在注意力，        │ 标准 GQA：128K 上下文下       │
│              │ 低秩 KV 压缩 + 解耦 RoPE               │ KV cache 数十 GB             │
│ 交叉熵       │ 标准 log-softmax + NLL                 │ 手写 CE：数值不稳定           │
└──────────────┴───────────────────────────────────────┴──────────────────────────────┘

为什么每个模块都手写解析反向而不是用自动微分：
    见 L1 的方案对比。一句话：CPU+NumPy 下 tape AD 慢 3~10 倍，
    且无法利用递推结构；这里全部改为手写，由 L1 的 gradcheck 兜底。

不变量（门禁复检）：
    P1 全部模块的解析反向通过中心差分检验（相对误差 < 1e-6）
    P2 RMSNorm 的输出 RMS == sqrt(m₂/(m₂+eps))（m₂ = mean(x²)，
       G22 门禁机检精确到 1e-14）；eps=1e-8 是数值保护项，因此**不等于
       恒为 1** —— 早期 docstring 写「RMS 恒为 1」是不精确宣称，
       实测偏差 2.67e-08 恰由该平滑项产生；大信号（m₂ >> eps）下趋于 1
    P3 RoPE 保持范数：‖RoPE(x)‖ == ‖x‖（旋转是正交变换）
    P4 RoPE 的相对位置性质：⟨q_i, k_j⟩ 只依赖 (i - j)
    P5 因果注意力：第 t 个输出与 t 之后的 token 无关（扰动未来，输出不变）
    P6 注意力权重行和恒为 1
"""

from __future__ import annotations

import numpy as np

from .l0_numerics import log_softmax, softmax, rms
from .l1_contract import Ctx, Module

__all__ = [
    "Linear", "RMSNorm", "RoPE", "SwiGLU", "CausalAttention", "CrossEntropy",
    "rope_tables", "causal_mask", "silu", "gelu",
]


# ---------------------------------------------------------------- 激活

def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish：x·σ(x)。SwiGLU 的 gate 侧。"""
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU（tanh 近似）。"""
    from math import sqrt as _sqrt
    c = np.float64(_sqrt(2.0 / 3.141592653589793))
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------- 线性

class Linear(Module):
    """y = x @ Wᵀ (+ b)。

    采用 W 为 (out, in) 的布局（PyTorch nn.Linear 约定），
    而不是 (in, out)：这样 y = x@Wᵀ 的反向 gx = g@W 是最自然的写法，
    且 dW = gᵀx 无需转置两次。
    """

    differentiable = (0, 1, 2)

    def forward(self, ctx: Ctx, x: np.ndarray, W: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64) @ np.asarray(W, dtype=np.float64).T
        if b is not None:
            y = y + b
        ctx.put(x=x, W=W, has_b=b is not None)
        return y

    def backward(self, ctx: Ctx, grad_out: np.ndarray) -> tuple:
        x, W, has_b = ctx["x"], ctx["W"], ctx["has_b"]
        g = np.asarray(grad_out, dtype=np.float64)
        gx = g @ W
        g2 = g.reshape(-1, g.shape[-1]) if g.ndim > 2 else g
        x2 = np.asarray(x).reshape(-1, np.asarray(x).shape[-1]) if np.asarray(x).ndim > 2 else np.asarray(x)
        gW = g2.T @ x2
        gb = g.reshape(-1, g.shape[-1]).sum(axis=0) if has_b else None
        return gx, gW, gb


# ---------------------------------------------------------------- 归一化

class RMSNorm(Module):
    """RMSNorm：y = (x / RMS(x)) ⊙ w。

    反向推导（这是手写反向里最容易写错的一个，完整展开）：
        记 r = RMS(x) = sqrt(mean(x²) + eps)，xh = x / r，y = xh ⊙ w
        令 u = ∂L/∂y ⊙ w = G ⊙ w
            ∂r/∂xᵢ = xᵢ / (n · r)
            ∂L/∂r = Σᵢ (∂L/∂xhᵢ)·(−xᵢ/r²) = −(1/r)·Σᵢ uᵢ·xhᵢ
        ⇒ ∂L/∂xᵢ = uᵢ/r + (∂L/∂r)·xᵢ/(n·r)
                 = (1/r)·[uᵢ − xhᵢ · mean_j(uⱼ·xhⱼ)]
        注意 mean_j(uⱼ·xhⱼ) 是标量（沿最后一维），即去均值操作。
        这个「减去自身方向投影」正是归一化层反向的通用结构。
    """

    differentiable = (0, 1)
    eps: float = 1e-8

    def forward(self, ctx: Ctx, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        r = rms(x, axis=-1, eps=self.eps)
        xh = x / r[..., None]
        y = xh * w
        ctx.put(xh=xh, r=r, w=w)
        return y

    def backward(self, ctx: Ctx, grad_out: np.ndarray) -> tuple:
        xh, r, w = ctx["xh"], ctx["r"], ctx["w"]
        g = np.asarray(grad_out, dtype=np.float64)
        u = g * w
        m = np.mean(u * xh, axis=-1, keepdims=True)
        gx = (u - xh * m) / r[..., None]
        gw = np.sum(g * xh, axis=tuple(range(g.ndim - 1)))
        return gx, gw


# ---------------------------------------------------------------- 位置编码

def rope_tables(dim: int, max_len: int, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """RoPE 的 cos/sin 表，形状 (max_len, dim)。

    θᵢ = 1 / base^(2i/dim)，i = 0..dim/2−1；位置 p 的旋转角 = p·θᵢ。
    因为采用 rotate_half 布局（前 d/2 与后 d/2 配对），
    cos/sin 表需要把前半维复制到后半维，才能与 (..., dim) 的输入广播。

    dim 必须是偶数（G22 抓出的静默缺陷）：rotate_half 要求前后半配对，
    奇数 dim 时 `np.concatenate([c, c])` 只能拼出 dim−1 长度的表 ——
    **表构造阶段不报错**，直到 RoPE.forward 广播时才在深处抛
    ValueError，且信息完全指向不了根因。此处直接响亮失败。
    """
    if dim < 2 or dim % 2 != 0:
        raise ValueError(
            f"RoPE 的 dim 必须是不小于 2 的偶数（rotate_half 需要前后半配对），"
            f"得到 {dim}")
    half = dim // 2
    inv = 1.0 / (base ** (2.0 * np.arange(0, half, dtype=np.float64) / dim))
    pos = np.arange(max_len, dtype=np.float64)[:, None]
    ang = pos * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    return np.concatenate([c, c], axis=-1), np.concatenate([s, s], axis=-1)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """(x₁, x₂) → (−x₂, x₁)，HF LLaMA 的 rotate_half 布局。

    为什么用「前后半分裂」而不是「奇偶交错」（GPT-NeoX 布局）：
    两者数学上只差一个置换，但 HF/LLaMA 生态的事实标准是前者，
    采用它才能让权重与社区模型对齐。
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return np.concatenate((-x2, x1), axis=-1)


class RoPE(Module):
    """旋转位置编码。cos/sin 是预计算常量，不可微。

    反向：y = x·cos + R·x·sin，其中 R 是 rotate_half 的正交矩阵且 Rᵀ = −R
        ⇒ ∂L/∂x = G·cos − R·(G·sin)

    不变量：旋转正交 ⇒ ‖RoPE(x)‖ == ‖x‖（门禁 P3）
    """

    differentiable = (0,)

    def forward(self, ctx: Ctx, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = x * cos + _rotate_half(x) * sin
        ctx.put(cos=cos, sin=sin)
        return y

    def backward(self, ctx: Ctx, grad_out: np.ndarray) -> tuple:
        cos, sin = ctx["cos"], ctx["sin"]
        g = np.asarray(grad_out, dtype=np.float64)
        return g * cos - _rotate_half(g * sin), None, None


# ---------------------------------------------------------------- 前馈

class SwiGLU(Module):
    """SwiGLU FFN：y = (SiLU(xW_gᵀ) ⊙ (xW_uᵀ)) W_dᵀ

    为什么用 3 个矩阵而不是 2 个：Shazeer 原文把隐藏维设为 (2/3)·4d，
    使 SwiGLU 的参数量与标准 FFN 持平。这里保留三个独立矩阵，
    维度由调用方决定（见 model.py 的 d_ff 计算）。

    反向：
        dA     = G W_d
        dU     = dA ⊙ SiLU(gate)
        dGate  = dA ⊙ U ⊙ SiLU'(gate)
        SiLU'(z) = σ(z)·(1 + z·(1 − σ(z)))
        dx     = dGate W_g + dU W_u
    """

    differentiable = (0, 1, 2, 3)

    def forward(self, ctx: Ctx, x, W_g, W_u, W_d) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        gate = x @ np.asarray(W_g).T
        up = x @ np.asarray(W_u).T
        a = silu(gate) * up
        y = a @ np.asarray(W_d).T
        ctx.put(x=x, W_g=W_g, W_u=W_u, W_d=W_d, gate=gate, up=up, a=a)
        return y

    def backward(self, ctx: Ctx, grad_out) -> tuple:
        x, W_g, W_u, W_d = ctx["x"], ctx["W_g"], ctx["W_u"], ctx["W_d"]
        gate, up, a = ctx["gate"], ctx["up"], ctx["a"]
        G = np.asarray(grad_out, dtype=np.float64)
        dA = G @ W_d
        dUp = dA * silu(gate)
        s = 1.0 / (1.0 + np.exp(-np.clip(gate, -60.0, 60.0)))
        dsilu = s * (1.0 + gate * (1.0 - s))
        dGate = dA * up * dsilu
        gx = dGate @ W_g + dUp @ W_u

        def _dw(gg: np.ndarray, xx: np.ndarray) -> np.ndarray:
            return gg.reshape(-1, gg.shape[-1]).T @ xx.reshape(-1, xx.shape[-1])

        return gx, _dw(dGate, x), _dw(dUp, x), _dw(G, a)


# ---------------------------------------------------------------- 注意力

def causal_mask(T: int) -> np.ndarray:
    """因果掩码：下三角（含对角）为 0，上三角为 −inf。"""
    m = np.zeros((T, T), dtype=np.float64)
    m[np.triu_indices(T, k=1)] = -np.inf
    return m


def causal_mask_rect(Tq: int, Tk: int) -> np.ndarray:
    """矩形因果掩码：Q 长度 Tq、K/V 长度 Tk（增量解码时 Tq=1、Tk=历史长度）。

    正确的对齐方式是**右对齐**：第 i 个 query 对应的是整个序列里的
    位置 (Tk − Tq + i)，因此它能看到的 key 是 j <= Tk − Tq + i。
    写成方阵再切片会在这里出错 —— 这是 KV cache 实现里最常见的 bug：
    左对齐会让最新 token 看不到任何历史，损失暴涨但不报错。

    Tq > Tk 一律拒绝（G22 抓出的静默 NaN）：此时前 (Tq − Tk) 行的
    query 一个 key 都看不到，整行 −inf ⇒ softmax 分母 exp(−inf) 全 0
    ⇒ 0/0 = NaN，输出带着 RuntimeWarning 静默污染整条链路。注意力分布
    在这种行上数学上无定义，必须在构造掩码处一声响亮失败。
    """
    if Tq > Tk:
        raise ValueError(
            f"Q 长度 ({Tq}) 不能超过 K/V 长度 ({Tk})：右对齐因果掩码下前 "
            f"{Tq - Tk} 行的 query 看不到任何 key，注意力分布无定义"
            f"（softmax 全 -inf 行会静默产生 NaN）")
    offset = Tk - Tq
    i = np.arange(Tq, dtype=np.int64)[:, None]
    j = np.arange(Tk, dtype=np.int64)[None, :]
    return np.where(j <= i + offset, 0.0, -np.inf)


class CausalAttention(Module):
    """缩放点积因果注意力（支持 GQA）。

    输入形状：q (B, H, T, D)，k/v (B, Hkv, T, D)，H = G·Hkv。
    实现上把 k/v 沿 head 维重复 G 次（内存换代码简洁）：
        重复后形状统一为 (B, H, T, D)，反向时把 head 维梯度求和回去。

    反向（标准，展开以免写错）：
        S = QKᵀ/√D + M ；P = softmax(S) ；O = P V
        dV = Pᵀ G
        dP = G Vᵀ
        dS = P ⊙ (dP − rowsum(dP ⊙ P))
        dQ = dS K / √D
        dK = dSᵀ Q / √D
    注意 dS 的 rowsum 项来自 softmax 的 Jacobian，漏掉它是最常见的 bug。
    """

    differentiable = (0, 1, 2)

    def forward(self, ctx: Ctx, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        B, H, T, Dq = q.shape
        Tkv = k.shape[2]                 # 增量解码时 Tkv > T
        Dv = v.shape[-1]
        Hkv = k.shape[1]
        if H % Hkv != 0:
            raise ValueError(f"q heads ({H}) must be a multiple of kv heads ({Hkv})")
        G = H // Hkv
        kk = np.repeat(k[:, :, None, :, :], G, axis=2).reshape(B, H, Tkv, Dq)
        vv = np.repeat(v[:, :, None, :, :], G, axis=2).reshape(B, H, Tkv, Dv)
        scale = 1.0 / np.sqrt(Dq)
        S = (q @ np.swapaxes(kk, -1, -2)) * scale
        S = S + (causal_mask(T) if T == Tkv else causal_mask_rect(T, Tkv))
        P = softmax(S, axis=-1)
        O = P @ vv
        ctx.put(q=q, k=kk, v=vv, P=P, scale=scale, G=G, Hkv=Hkv)
        return O

    def backward(self, ctx: Ctx, grad_out) -> tuple:
        q, kk, vv, P, scale, G, Hkv = (
            ctx["q"], ctx["k"], ctx["v"], ctx["P"], ctx["scale"], ctx["G"], ctx["Hkv"]
        )
        Gout = np.asarray(grad_out, dtype=np.float64)
        dV = np.swapaxes(P, -1, -2) @ Gout
        dP = Gout @ np.swapaxes(vv, -1, -2)
        dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True))
        dQ = (dS @ kk) * scale
        dK = (np.swapaxes(dS, -1, -2) @ q) * scale
        # 把 head 维的梯度合并回 Hkv（K/V 各自的长度与最后一维都可以不同）
        B, H = dK.shape[0], dK.shape[1]
        Dv = dV.shape[-1]
        dK = dK.reshape(B, Hkv, G, dK.shape[2], -1).sum(axis=2)
        dV = dV.reshape(B, Hkv, G, dV.shape[2], Dv).sum(axis=2)
        return dQ, dK, dV


# ---------------------------------------------------------------- 损失

class CrossEntropy(Module):
    """因果语言模型损失。logits (B, T, V)，targets (B, T)。

    平均到「非忽略」的 token 上（ignore_index = −1 的位置不计入）。
    """

    differentiable = (0,)

    def forward(self, ctx: Ctx, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets)
        lp = log_softmax(logits, axis=-1)                       # (B,T,V)
        B, T, V = lp.shape
        flat = lp.reshape(B * T, V)
        tgt = targets.reshape(-1)
        mask = tgt >= 0
        idx = np.where(mask, tgt, 0)
        nll = -flat[np.arange(B * T), idx]
        nll = np.where(mask, nll, 0.0)
        ctx.put(lp=lp, flat=flat, tgt=tgt, mask=mask, B=B, T=T, V=V)
        if mask.sum() == 0:
            return np.float64(0.0)
        return np.float64(nll.sum() / mask.sum())

    def backward(self, ctx: Ctx, grad_out) -> tuple:
        lp, tgt, mask, B, T, V = ctx["lp"], ctx["tgt"], ctx["mask"], ctx["B"], ctx["T"], ctx["V"]
        n = max(int(mask.sum()), 1)
        # 关键：∂log_softmax_k/∂logits_v = δ_kv − softmax_v
        # ⇒ ∂(−Σ_k y_k·lp_k)/∂logits_v = softmax_v − y_v   （onehot 时 Σy_k = 1）
        # 早期版本这里误用了 log-softmax 本身（lp）而不是 softmax（exp(lp)），
        # 数值差分直接对不上 —— 这是归一化指数族反向的经典陷阱。
        P = np.exp(lp.reshape(B * T, V))                  # softmax 概率
        g = np.zeros((B * T, V), dtype=np.float64)
        idx = np.where(mask, tgt, 0)
        g[np.arange(B * T), idx] = -1.0
        g *= mask[:, None]
        g += P * mask[:, None]
        g = g * (float(np.asarray(grad_out)) / n)
        return g.reshape(B, T, V), None
