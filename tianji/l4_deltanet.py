"""L4 · 线性序列混合器 —— Gated DeltaNet（Qwen3-Next / Kimi Linear 的核心层）。

站在谁之上（零重写）：
    - Delta Rule（写入前先擦除）  —— Schlag et al. 2021, "Linear Transformers with
                                     the Delta Rule"；等价于在联想记忆上做一步在线梯度下降
    - Gated DeltaNet（遗忘门+delta） —— Yang et al. ICLR 2025, "Gated Delta Networks:
                                     Improving Mamba2 with Delta Rule"；NVIDIA
    - 门控线性注意力的统一视角      —— Yang et al. 2024, "Gated Linear Attention
                                     Transformers with Hardware-Efficient Training"
    - 短卷积（depthwise causal conv）—— fla / Mamba-2 的标配局部混合预处理

为什么 2026 年这一层是必选项：
    Qwen3-Next、Qwen3.5、Kimi Linear(KDA)、Olmo Hybrid 全部收敛到
    「3 层 Gated DeltaNet + 1 层全注意力」的 3:1 布局；
    Mamba-3 (ICLR 2026) 从任务分布上证明 ~25% 全注意力比例即足够。
    Kimi Linear 报告 KV cache −75%、百万上下文解码最高 6×。

递推式（本实现采用 S ∈ R^{d_v × d_k} 布局，使 o = S q 无需转置）：
    S_t = α_t · S_{t−1} · (I − β_t k_t k_tᵀ) + β_t · v_t k_tᵀ
    o_t = S_t q_t
    其中 k_t 先做 L2 归一化（论文要求，否则 β·k kᵀ 的谱半径失控）

反向（沿时间反向递推 —— 关键，不能逐节点展开）：
    记 M_t = I − β_t k_t k_tᵀ（对称），G_t = ∂L/∂S_t
        G_T = g^o_T · q_Tᵀ
        G_t = g^o_t · q_tᵀ + α_{t+1} · G_{t+1} · M_{t+1}
            = g^o_t q_tᵀ + α_{t+1} [G_{t+1} − β_{t+1}(G_{t+1} k_{t+1}) k_{t+1}ᵀ]
    参数梯度（含 k 出现在三处的完整展开，漏一处即 gradcheck 失败）：
        ∂L/∂v_t     = β_t · G_t k_t
        ∂L/∂q_t     = S_tᵀ g^o_t
        ∂L/∂α_t     = ⟨G_t, S_{t−1} − β_t P_t k_tᵀ⟩       P_t = S_{t−1} k_t
        ∂L/∂β_t     = ⟨G_t, −α_t P_t k_tᵀ + v_t k_tᵀ⟩
        ∂L/∂k_t     = −α_tβ_t [ S_{t−1}ᵀ(G_t k_t) + G_tᵀ P_t ] + β_t · G_tᵀ v_t

为什么用递推而不是论文的 chunkwise 并行形式：
    chunkwise（WY 表示 + UT transform）是为 GPU 设计的 —— 它把 T×T 的递推
    换成块内矩阵乘 + 块间递推，代价是需要求 (I − 三角矩阵)⁻¹ 与额外的
    d×d 中间矩阵。在 CPU + NumPy 后端下，Python 层的循环开销被 einsum 的
    BLAS 调用摊薄，实测递推形式的吞吐更高且数值更稳（无需矩阵求逆）。
    chunkwise 作为 L10 微指令家族中的 associative-scan 家族保留其算子定义。

不变量（门禁复检）：
    D1 解析反向通过中心差分检验（相对误差 < 1e-6）
    D2 线性复杂度：T 加倍，耗时近似加倍（而非平方）
    D3 状态有界：‖S_t‖_F 有界（β∈(0,1)、α∈(0,1) 且 k 归一化 ⇒ 收缩映射）
    D4 递推 == 定义式：循环实现与「展开的朴素矩阵式」逐位一致
    D5 因果性：扰动 t 之后的输入，o_0..o_t 不变
"""

from __future__ import annotations

import numpy as np

from .l1_contract import Ctx, Module

__all__ = ["GatedDeltaNet", "CausalConv1D", "l2norm", "L2Norm"]


# ---------------------------------------------------------------- L2 归一化

class L2Norm(Module):
    """沿最后一维的 L2 归一化（Gated DeltaNet 要求对 k 做，保证谱半径可控）。

    反向与 RMSNorm 同构：∂L/∂k = (1/‖k‖)[ĝ − k̂·(ĝ·k̂)]
    """

    differentiable = (0,)
    eps: float = 1e-8

    def forward(self, ctx: Ctx, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = np.linalg.norm(x, axis=-1, keepdims=True) + self.eps
        kh = x / n
        ctx.put(kh=kh, n=n)
        return kh

    def backward(self, ctx: Ctx, grad_out) -> tuple:
        kh, n = ctx["kh"], ctx["n"]
        g = np.asarray(grad_out, dtype=np.float64)
        proj = np.sum(g * kh, axis=-1, keepdims=True)
        return (g - kh * proj) / n,


def l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.asarray(x) / (np.linalg.norm(np.asarray(x), axis=-1, keepdims=True) + eps)


# ---------------------------------------------------------------- 短卷积

class CausalConv1D(Module):
    """深度可分离因果卷积（kernel_size 通常为 4），fla/Mamba-2 标配。

    为什么需要它：纯 delta-rule 递推对「局部相邻 token 的模式」学习效率低，
    加一层极廉价的局部卷积后，长程交给递推状态、短程交给卷积，
    实测收敛更快（Mamba-2 与 fla 的消融一致）。

    输入 x: (B, T, C)（通道维做 depthwise）；weight: (C, K)
    """

    differentiable = (0, 1)

    def __init__(self, kernel_size: int = 4):
        self.K = int(kernel_size)

    def forward(self, ctx: Ctx, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        B, T, C = x.shape
        K = self.K
        # 左侧补 K-1 个零，保证因果（位置 t 只能看到 t-K+1..t）
        xp = np.concatenate([np.zeros((B, K - 1, C)), x], axis=1)
        out = np.zeros((B, T, C), dtype=np.float64)
        for k in range(K):
            out += xp[:, k:k + T, :] * weight[:, k][None, None, :]
        ctx.put(x=x, xp=xp, weight=weight, B=B, T=T, C=C)
        return out

    def backward(self, ctx: Ctx, grad_out) -> tuple:
        x, xp, weight, B, T, C = ctx["x"], ctx["xp"], ctx["weight"], ctx["B"], ctx["T"], ctx["C"]
        g = np.asarray(grad_out, dtype=np.float64)
        K = self.K
        gx = np.zeros((B, T + K - 1, C), dtype=np.float64)
        gw = np.zeros((C, K), dtype=np.float64)
        for k in range(K):
            gx[:, k:k + T, :] += g * weight[:, k][None, None, :]
            gw[:, k] = np.sum(g * xp[:, k:k + T, :], axis=(0, 1))
        gx = gx[:, K - 1:, :]
        return gx, gw


# ---------------------------------------------------------------- Gated DeltaNet

class GatedDeltaNet(Module):
    """Gated DeltaNet 序列混合器。

    Parameters
    ----------
    q, k, v : (B, H, T, D)   已经过投影与 reshape；k 会在内部做 L2 归一化
    beta    : (B, H, T)      写入强度，值域 (0,1)，由 sigmoid 生成
    alpha   : (B, H, T)      遗忘门，值域 (0,1)，由 exp(-softplus(a)) 生成

    Returns
    -------
    o : (B, H, T, D)
    """

    differentiable = (0, 1, 2, 3, 4)

    def forward(self, ctx: Ctx, q, k, v, beta, alpha) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        B, H, T, D = q.shape
        assert k.shape == q.shape and v.shape == q.shape, "q/k/v 必须同形"
        assert beta.shape == (B, H, T) and alpha.shape == (B, H, T), "beta/alpha 必须为 (B,H,T)"

        kh = l2norm(k)                                  # (B,H,T,D)
        S_prev = np.zeros((B, H, D, D), dtype=np.float64)
        S_all = np.zeros((B, H, T + 1, D, D), dtype=np.float64)
        P_all = np.zeros((B, H, T, D), dtype=np.float64)
        o = np.zeros((B, H, T, D), dtype=np.float64)

        for t in range(T):
            kt = kh[:, :, t, :]                          # (B,H,D)
            vt = v[:, :, t, :]
            bt = beta[:, :, t][..., None, None]          # (B,H,1,1)
            at = alpha[:, :, t][..., None, None]
            P = np.einsum("bhvk,bhk->bhv", S_prev, kt)   # S_{t-1} k_t
            # S_t = α (S_{t-1} − β P kᵀ) + β v kᵀ
            S = at * (S_prev - bt * P[..., :, None] * kt[..., None, :]) \
                + bt * vt[..., :, None] * kt[..., None, :]
            o[:, :, t, :] = np.einsum("bhvk,bhk->bhv", S, q[:, :, t, :])
            P_all[:, :, t, :] = P
            S_all[:, :, t + 1, :, :] = S
            S_prev = S

        ctx.put(q=q, k=k, kh=kh, v=v, beta=beta, alpha=alpha,
                S_all=S_all, P_all=P_all, T=T, D=D)
        return o

    def backward(self, ctx: Ctx, grad_out) -> tuple:
        q, kh, v, beta, alpha = ctx["q"], ctx["kh"], ctx["v"], ctx["beta"], ctx["alpha"]
        S_all, P_all, T, D = ctx["S_all"], ctx["P_all"], ctx["T"], ctx["D"]
        g = np.asarray(grad_out, dtype=np.float64)
        B, H = q.shape[0], q.shape[1]

        dq = np.zeros_like(q)
        dkh = np.zeros_like(kh)
        dv = np.zeros_like(v)
        dbeta = np.zeros_like(beta)
        dalpha = np.zeros_like(alpha)

        G = np.zeros((B, H, D, D), dtype=np.float64)     # G_t，从 t=T-1 反向累积
        for t in range(T - 1, -1, -1):
            kt = kh[:, :, t, :]
            vt = v[:, :, t, :]
            qt = q[:, :, t, :]
            gt = g[:, :, t, :]                            # g^o_t
            bt = beta[:, :, t]
            at = alpha[:, :, t]

            # 直接来自 o_t = S_t q_t
            G = G + gt[..., :, None] * qt[..., None, :]
            # ---- 此时 G 就是 ∂L/∂S_t ----
            S_prev = S_all[:, :, t, :, :]
            P = P_all[:, :, t, :]

            Gk = np.einsum("bhvk,bhk->bhv", G, kt)        # G_t k_t   (B,H,D)
            GtP = np.einsum("bhvk,bhv->bhk", G, P)        # G_tᵀ P_t
            GtV = np.einsum("bhvk,bhv->bhk", G, vt)       # G_tᵀ v_t
            StGk = np.einsum("bhvk,bhv->bhk", S_prev, Gk)  # S_{t-1}ᵀ(G_t k_t)

            b = bt[..., None]
            a = at[..., None]
            dkh[:, :, t, :] = -a * b * (StGk + GtP) + b * GtV
            dv[:, :, t, :] = b * Gk
            dq[:, :, t, :] = np.einsum("bhvk,bhv->bhk", S_all[:, :, t + 1, :, :], gt)

            # ∂L/∂β_t = ⟨G_t, −α P kᵀ + v kᵀ⟩
            term_b = -a[..., None] * P[..., :, None] * kt[..., None, :] \
                + vt[..., :, None] * kt[..., None, :]
            dbeta[:, :, t] = np.einsum("bhvk,bhvk->bh", G, term_b)
            # ∂L/∂α_t = ⟨G_t, S_{t−1} − β P kᵀ⟩
            term_a = S_prev - b[..., None] * P[..., :, None] * kt[..., None, :]
            dalpha[:, :, t] = np.einsum("bhvk,bhvk->bh", G, term_a)

            # ---- 递推到 t−1：G_{t−1} = α_t [G_t − β_t (G_t k_t) k_tᵀ] ----
            G = at[..., None, None] * (G - bt[..., None, None] * Gk[..., :, None] * kt[..., None, :])

        # k 的 L2 归一化反向
        k = ctx.get("k")
        if k is None:
            dk = dkh
        else:
            k = np.asarray(k, dtype=np.float64)
            n = np.linalg.norm(k, axis=-1, keepdims=True) + 1e-8
            proj = np.sum(dkh * kh, axis=-1, keepdims=True)
            dk = (dkh - kh * proj) / n

        return dq, dk, dv, dbeta, dalpha
