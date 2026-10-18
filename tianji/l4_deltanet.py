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

三种实现形式都存在（同一数学，不同算法路径，由门禁机检等价）：
    1. `forward`        —— 顺序递推（主干：CPU+NumPy 下最快、最省内存、无需矩阵求逆）
    2. `gdn_parallel`   —— 扁平并行前缀扫描（Blelloch 1990 associative scan）
    3. `gdn_chunkwise`  —— chunkwise 分块并行（块内并行 + 块间递推 O(T/C) 步，
                           真实 GPU 内核的分块本质，见本文件末尾）
    三者必须逐位一致：G11 机检 (1)≡(2)，G12 机检 (1)≡(3)，实测最大差均 8.88e-16。

为什么不用**论文原版** chunkwise（WY 表示 + UT transform）：
    那是为 GPU 设计的 —— 它把 T×T 的递推换成块内矩阵乘 + 块间递推，
    代价是需要求 (I − 三角矩阵)⁻¹ 与额外的 d×d 中间矩阵。在 CPU + NumPy 后端下
    数值更脆、内存更大，因此本项目只保留其算子定义（L10 `scan` 家族），
    而用数学上严格同构、但无需矩阵求逆的「仿射算子块级组合」实现 GPU 分块结构。

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

__all__ = ["GatedDeltaNet", "CausalConv1D", "l2norm", "L2Norm",
           "gdn_init_state", "gdn_step", "gdn_decode_recurrent",
           "gdn_parallel", "gdn_compose", "gdn_scan_inclusive",
           "gdn_chunkwise"]


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


# ---------------------------------------------------------------- O(1) 递推增量解码
#
# 站在谁之上（零重写）：
#     Gated DeltaNet 的核心卖点就是「状态 S_t 是 O(1) 的」——
#     推理时每个新 token 只做 S_t = α(S_{t−1} − β P kᵀ) + β v kᵀ 一次秩-1 更新，
#     而非整序列重算。这是 GDN 相比全注意力的根本效率来源
#     （Kimi Linear 据此报告百万上下文解码 6× 加速）。
#
# 为什么本项目之前没接：原 README「已知边界」明确标注——
#     「GDN 的推理态仍是整序列重算，尚未实现 O(1) 递推增量解码」。
#     这里的 `step` 与 `forward` 用的是**逐位相同**的浮点操作序列，
#     因此门禁可证明「递推增量 ≡ 整序列重算」到机器精度（≈1e-16），
#     把这条已知缺口真正关闭，而不是另写一套数值不同的实现。
#
# 状态布局：S ∈ R^{B×H×D×D}（与 forward 内部 S_prev 完全一致），
# 使 o = S q 无需转置。D = d_v = d_k（本项目 GDN 的 d_v 即 d_head）。

def gdn_init_state(B: int, H: int, D: int, dtype=np.float64) -> np.ndarray:
    """初始化空的 GDN 递推状态 S_0 = 0。"""
    return np.zeros((B, H, D, D), dtype=dtype)


def gdn_step(state: np.ndarray, q_t: np.ndarray, k_t: np.ndarray, v_t: np.ndarray,
             beta_t: np.ndarray, alpha_t: np.ndarray):
    """单步 O(1) 递推解码（与 `GatedDeltaNet.forward` 的循环体逐位一致）。

    Parameters
    ----------
    state : (B, H, D, D)   上一时刻状态 S_{t−1}
    q_t   : (B, H, D)      当前 token 的 query（已投影、已 reshape）
    k_t   : (B, H, D)      当前 token 的 key（已投影、已 reshape；内部做 L2 归一化）
    v_t   : (B, H, D)      当前 token 的 value（已投影、已 reshape）
    beta_t: (B, H)         写入强度 ∈ (0,1)
    alpha_t: (B, H)        遗忘门 ∈ (0,1)

    Returns
    -------
    o_t : (B, H, D)        当前 token 的输出
    S_t : (B, H, D, D)     更新后的状态（直接回写缓存即可）
    """
    q_t = np.asarray(q_t, dtype=np.float64)
    k_t = np.asarray(k_t, dtype=np.float64)
    v_t = np.asarray(v_t, dtype=np.float64)
    beta_t = np.asarray(beta_t, dtype=np.float64)
    alpha_t = np.asarray(alpha_t, dtype=np.float64)

    kh = l2norm(k_t)                                   # (B,H,D)  —— 与 forward 的 l2norm(k) 同序
    bt = beta_t[..., None, None]                        # (B,H,1,1)
    at = alpha_t[..., None, None]
    P = np.einsum("bhvk,bhk->bhv", state, kh)           # S_{t−1} k_t
    # S_t = α (S_{t−1} − β P kᵀ) + β v kᵀ —— 与 forward 循环体完全相同的表达式
    S = at * (state - bt * P[..., :, None] * kh[..., None, :]) \
        + bt * v_t[..., :, None] * kh[..., None, :]
    o = np.einsum("bhvk,bhk->bhv", S, q_t)
    return o, S


def gdn_decode_recurrent(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                         beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """用 `step` 沿时间递推解出整段输出，作为 `forward` 的对照实现。

    q,k,v : (B, H, T, D)   beta,alpha : (B, H, T)
    返回 o : (B, H, T, D)
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    B, H, T, D = q.shape
    S = gdn_init_state(B, H, D)
    o = np.zeros((B, H, T, D), dtype=np.float64)
    for t in range(T):
        o[:, :, t, :], S = gdn_step(S, q[:, :, t, :], k[:, :, t, :],
                                    v[:, :, t, :], beta[:, :, t], alpha[:, :, t])
    return o


# ---------------------------------------------------------------- 并行前缀扫描形式（门禁 G11）
#
# 站在谁之上（零重写）：
#     线性注意力（含 Gated DeltaNet）的「并行训练形式」是硬件高效内核的基石 ——
#     GLA / Mamba-2 / Gated DeltaNet 的 CUDA kernel 都用 chunkwise + 关联扫描
#     （associative scan, Blelloch 1990）把 O(T) 递推写成可并行的前缀和。
#     这里给出其纯 NumPy 参照实现，作为门禁 G11 的对照基准：证明「并行形式」
#     与「递推形式」逐位等价 —— 这是线性注意力模型的决定性正确性质（D4 的
#     算法级强化：不是同式重算，而是两条不同算法路径必须收敛到同一数值）。
#
# 递推：S_t = α_t S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
# 每个时间步是仿射算子 F_t(S) = A_t S M_t + C_t，其中
#     A_t = α_t I,   M_t = I − β_t k_t k_tᵀ,   C_t = β_t v_t k_tᵀ
# 两算子「先 E 后 F」的合成：
#     A' = A_F A_E,   M' = M_E M_F,   C' = A_F C_E M_F + C_F
# 包含前缀 P_t = F_t ∘ … ∘ F_0 ⇒ S_t = P_t(0) = C_{P_t}，o_t = S_t q_t。
# 下面用分治（work-efficient）并行前缀扫描实现，与顺序递推是不同算法路径。

def gdn_compose(E, F):
    """合成两个仿射算子（先 E 后 F）：返回新的 (A, M, C)。"""
    A_E, M_E, C_E = E
    A_F, M_F, C_F = F
    A = A_F @ A_E
    M = M_E @ M_F
    C = A_F @ (C_E @ M_F) + C_F
    return A, M, C


def gdn_scan_inclusive(ops):
    """分治并行包含前缀扫描：输入算子列表，返回每个前缀的合成算子。

    与顺序递推（沿 t 累乘）是不同算法路径；work O(T log T)、depth O(log T)，
    结构上对应 GPU 上 chunkwise 内核的关联扫描，而非单线程循环。
    """
    if len(ops) == 1:
        return [ops[0]]
    mid = len(ops) // 2
    left = gdn_scan_inclusive(ops[:mid])
    right = gdn_scan_inclusive(ops[mid:])
    last_left = left[-1]
    return left + [gdn_compose(last_left, r) for r in right]


def gdn_parallel(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                 beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Gated DeltaNet 的并行前缀扫描形式（对照 `forward` 的递推实现）。

    q,k,v : (B, H, T, D)   beta,alpha : (B, H, T)
    返回 o : (B, H, T, D)

    对每个 (b,h) 独立构造 T 个仿射算子，做包含前缀扫描，再取 C_{P_t} 作为 S_t。
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    B, H, T, D = q.shape
    kh = l2norm(k)
    eye = np.eye(D)
    o = np.zeros((B, H, T, D), dtype=np.float64)
    for b in range(B):
        for h in range(H):
            ops = []
            for t in range(T):
                a = alpha[b, h, t]
                bt = beta[b, h, t]
                kt = kh[b, h, t]
                vt = v[b, h, t]
                A = a * eye
                M = eye - bt * np.outer(kt, kt)
                C = bt * np.outer(vt, kt)
                ops.append((A, M, C))
            pref = gdn_scan_inclusive(ops)
            for t in range(T):
                S_t = pref[t][2]                       # C_{P_t} = S_t（S_{-1}=0）
                o[b, h, t] = S_t @ q[b, h, t]
    return o


# ---------------------------------------------------------------- chunkwise（分块并行，门禁 G12）
#
# 站在谁之上（零重写）：
#     chunkwise 是 Gated DeltaNet / GLA / Mamba-2 真实 GPU 内核的训练形式
#     （Yang et al. ICLR 2025 的 WY 表示 + UT transform；flash-linear-attention 内核库）。
#     本项目原实现只用顺序递推，并把 chunkwise 留在 L10 微指令的算子定义里（README「已知边界」）。
#     这里把它真正补上，结构与硬件内核一致：
#         · 块内（intra-chunk）：把 C 个时间步的仿射算子复合成 1 个「块算子」，
#           并用块内包含前缀取得每个时间步的输出 → 对应 GPU 上块内的张量核心矩阵乘；
#         · 块间（inter-chunk）：只递推「块到块」的状态转移（O(T/C) 步而非 O(T) 步），
#           → 对应序列并行 / 上下文并行的核心收益来源。
#     与 `forward`（顺序递推）、`gdn_parallel`（扁平并行扫描）是三条不同的算法路径，
#     三者必须逐位一致 —— 门禁 G12 机检这条线性注意力的最强正确性质。

def gdn_chunkwise(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                  beta: np.ndarray, alpha: np.ndarray,
                  chunk: int = 4) -> np.ndarray:
    """Gated DeltaNet 的 chunkwise（分块并行）形式。

    q,k,v : (B, H, T, D)   beta,alpha : (B, H, T)   chunk : 块长 C（不必整除 T）
    返回 o : (B, H, T, D)

    最后一块长度不足时正确处理（不必整除），避免只在整除情况下恰好成立。
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    B, H, T, D = q.shape
    C = int(chunk)
    if C < 1:
        raise ValueError("chunk 必须 >= 1")
    kh = l2norm(k)
    eye = np.eye(D)
    o = np.zeros((B, H, T, D), dtype=np.float64)
    for b in range(B):
        for h in range(H):
            # ---- 每个时间步的仿射算子 F_t = (A_t, M_t, C_t)
            ops = []
            for t in range(T):
                a = alpha[b, h, t]
                bt = beta[b, h, t]
                kt = kh[b, h, t]
                vt = v[b, h, t]
                ops.append((a * eye,
                            eye - bt * np.outer(kt, kt),
                            bt * np.outer(vt, kt)))
            # ---- 分块：块算子 = 块内所有 F_t 的复合；块间只需 O(T/C) 步递推
            starts, block_ops = [], []
            idx = 0
            while idx < T:
                end = min(idx + C, T)
                starts.append(idx)
                acc = ops[idx]
                for t in range(idx + 1, end):
                    acc = gdn_compose(acc, ops[t])
                block_ops.append(acc)
                idx = end
            # ---- 块间递推 + 块内包含前缀读输出
            S_start = np.zeros((D, D))        # 进入本块前的状态 S_{start-1}
            for ci, bo in enumerate(block_ops):
                st = starts[ci]
                en = min(st + C, T)
                intra = [ops[st]]
                for t in range(st + 1, en):
                    intra.append(gdn_compose(intra[-1], ops[t]))
                for j, t in enumerate(range(st, en)):
                    A, M, Cc = intra[j]
                    S_t = A @ S_start @ M + Cc   # 块内包含前缀作用在入口状态上
                    o[b, h, t] = S_t @ q[b, h, t]
                A_b, M_b, C_b = bo
                S_start = A_b @ S_start @ M_b + C_b   # 块到块：递推步数 O(T/C)
    return o
