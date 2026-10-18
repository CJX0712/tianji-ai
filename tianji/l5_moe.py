"""L5 · 稀疏化 —— DeepSeek-V3 风格 MoE（无辅助损失负载均衡）。

站在谁之上（零重写，含被否决方案的明确理由）：

┌────────────────────────────┬────────────────────────────────┬─────────────────────────────┐
│ 设计点                     │ 采用方案（世界顶级）             │ 否决的替代方案               │
├────────────────────────────┼────────────────────────────────┼─────────────────────────────┤
│ 路由打分                   │ Sigmoid 亲和度（DeepSeek-V3）   │ Softmax：专家间被强制归一化， │
│                            │                                │ 打分相互挤压，Top-K 区分度差 │
│ 负载均衡                   │ 动态偏置 b_i，γ=0.001（V3）     │ Switch 辅助损失 α·E·Σf·P：   │
│                            │ 偏置只影响「谁被选中」，         │ 梯度与主目标对抗，α 越大     │
│                            │ 不影响「选中后算多少」           │ 语言建模质量越差（有公开消融）│
│ 专家结构                   │ 路由专家 + 共享专家（V3）        │ 纯路由专家：通用知识在每个    │
│                            │                                │ 专家里重复存储，浪费容量      │
│ 容量与丢弃                 │ 无容量、不丢弃 token（V3）       │ Switch 容量因子 + drop：     │
│                            │                                │ 丢弃的 token 走残差直通，     │
│                            │                                │ 引入训练/推理不一致           │
│ 数值稳定                   │ 极小序列级均衡损失 α=0.0001     │ ST-MoE 的 router z-loss：    │
│                            │ （V3 保留，仅防单序列极端失衡）  │ V3 已用无辅助损失路径，       │
│                            │                                │ z-loss 与之正交且此处非必需   │
└────────────────────────────┴────────────────────────────────┴─────────────────────────────┘

为什么「无辅助损失」是真正的进步：
    辅助损失把「均衡」写进损失函数，于是均衡的梯度与语言建模的梯度在同一条
    反向路径上互相拉扯：α 小了压不住 collapse，α 大了模型质量肉眼可见地掉。
    V3 把均衡搬出损失函数，改成只对路由生效的控制器（偏置 b_i），
    梯度路径彻底解耦 —— 这是架构级解法，不是调参技巧。

不变量（门禁复检）：
    M1 解析反向通过中心差分检验（相对误差 < 1e-6）
    M2 门控归一化：每个 token 的选中专家权重之和 == 1
    M3 偏置控制器有效：持续过载的专家，其偏置单调下降（负载均衡收敛）
    M4 稀疏性：每个 token 只激活 K 个路由专家（+共享专家），
       与稠密基线相比 FLOPs 比为 K/E
    M5 容量守恒：所有 token 都被处理，无丢弃（V3 语义）
"""

from __future__ import annotations

import numpy as np

from .l0_numerics import topk
from .l1_contract import Ctx, Module

__all__ = ["MoELayer", "LoadBalancer", "sigmoid", "sequence_balance_loss"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值安全 sigmoid：分段避免大负数 exp 溢出。"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def sequence_balance_loss(affinity: np.ndarray, assign: np.ndarray, alpha: float = 1e-4):
    """DeepSeek-V3 的极小序列级均衡损失：L = α Σᵢ fᵢ·Pᵢ

    fᵢ = 分给专家 i 的 token 比例（离散量，按 V3 的实现 detach），
    Pᵢ = 专家 i 的平均亲和度（可微）。

    为什么 detach fᵢ：fᵢ 由 top-k 的离散选择产生，几乎处处梯度为零；
    若强行估计其梯度会引入巨大方差。V3 只用它来防止「单条序列内部」
    的极端失衡，权重 0.0001 量级，主目标几乎不受影响。
    """
    affinity = np.asarray(affinity, dtype=np.float64)
    assign = np.asarray(assign, dtype=np.float64)
    E = affinity.shape[-1]
    f = assign.mean(axis=0) if assign.ndim > 1 else assign      # (E,)
    P = affinity.mean(axis=0)                                    # (E,)
    loss = alpha * float(np.sum(f * P)) * E
    grad = alpha * f * E / max(affinity.shape[0], 1)             # ∂L/∂affinity
    grad = np.broadcast_to(grad, affinity.shape).copy()
    return loss, grad


class LoadBalancer:
    """无辅助损失的偏置控制器（DeepSeek-V3 原义）。

    b_i 的更新不经过梯度，只在每步训练后由实际负载驱动：
        过载 → b_i −= γ ；欠载 → b_i += γ
    V3 的 γ = 0.001，且在 14.3T token 之后归零（路由格局冻结）。

    为什么用 sign 而不是比例：V3 原文即按固定步长调整；用比例会让
    已经很接近均衡的专家产生振荡，固定步长在长期更稳。
    """

    def __init__(self, n_experts: int, gamma: float = 1e-3, bias_cap: float = 1.0):
        self.n = int(n_experts)
        self.gamma = float(gamma)
        self.cap = float(bias_cap)
        self.bias = np.zeros(self.n, dtype=np.float64)
        self.history: list[np.ndarray] = []
        self.step = 0

    def update(self, counts: np.ndarray) -> None:
        counts = np.asarray(counts, dtype=np.float64)
        if counts.size == 0:
            return
        avg = counts.mean()
        delta = np.where(counts > avg, -self.gamma, np.where(counts < avg, self.gamma, 0.0))
        self.bias = np.clip(self.bias + delta, -self.cap, self.cap)
        self.history.append(counts.copy())
        self.step += 1

    def maxcv(self, window: int = 32) -> float:
        """近期负载的变异系数（越小越均衡）。用于门禁 M3。

        历史不足两条时返回 inf —— 「无法判断」必须显式表示成 inf，
        不能返回 0 假装很均衡。
        """
        if len(self.history) < 2:
            return float("inf")
        h = np.array(self.history[-window:], dtype=np.float64)
        r = h / (h.mean(axis=1, keepdims=True) + 1e-12)
        return float(r.std())

    def reset(self) -> None:
        self.bias[:] = 0.0
        self.history.clear()
        self.step = 0


class MoELayer(Module):
    """DeepSeek-V3 风格 MoE 层。

    参数布局（全部 3D/2D 张量，便于一次 einsum 处理）：
        x        : (N, d)          N = B·T，已展平
        router   : (E, d)          路由专家打分
        Wg, Wu   : (E, dff, d)     每个专家的 SwiGLU gate/up
        Wd       : (E, d, dff)     每个专家的 down
        bias     : (E,)            负载均衡偏置（不可微，由 LoadBalancer 维护）
        sWg, sWu, sWd : 共享专家（可选，2D）

    门控值（V3 语义）：
        s_i   = σ(uᵀe_i)                     ← 参与「算多少」
        sel   = TopK(s_i + b_i)              ← 偏置只参与「谁被选中」
        g_i   = s_i / Σ_{j∈sel} s_j
    """

    differentiable = (0, 1, 2, 3, 4, 5, 6, 7)

    def __init__(self, n_experts: int, top_k: int = 2, shared: bool = True):
        self.E = int(n_experts)
        self.K = int(top_k)
        self.shared = bool(shared)

    def forward(self, ctx: Ctx, x, router, Wg, Wu, Wd, bias, sWg=None, sWu=None, sWd=None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        router = np.asarray(router, dtype=np.float64)
        Wg = np.asarray(Wg, dtype=np.float64)
        Wu = np.asarray(Wu, dtype=np.float64)
        Wd = np.asarray(Wd, dtype=np.float64)
        bias = np.zeros(self.E, dtype=np.float64) if bias is None else np.asarray(bias, dtype=np.float64)

        N, d = x.shape
        E, K = self.E, self.K

        logits = x @ router.T                    # (N, E)
        # 数值下界保护：clip 到 [-60, 60]（与 silu 同惯例）。不加此保护时，
        # 极端输入（大激活 × 大 router 权重）可使选中专家的 sigmoid 全部
        # 下溢为 0，归一化 Z→1e-12 而 gated 全零 → 该 token 门控行和=0
        # （M2 失效），路由专家输出整体丢失。clip 后 σ(±60)∈[8.8e-27, 1-8.8e-27]，
        # 任意 E 个专家之和恒远大于 Z 的 1e-12 保护阈值，归一化永不塌陷。
        logits = np.clip(logits, -60.0, 60.0)
        s = sigmoid(logits)                      # 亲和度（参与门控值）
        _, sel = topk(s + bias, K, axis=-1)      # 偏置只参与选择
        mask = np.zeros((N, E), dtype=np.float64)
        np.put_along_axis(mask, sel, 1.0, axis=-1)
        gated = s * mask
        # 归一化必须无加法偏置：Z = Σ + ε 会在选中亲和度极小（极端输入下
        # σ 低至 8.8e-27）时被 ε 主导，门控行和偏离 1（M2 失效）。
        # 正确写法：Σ>0 时严格归一（行和恒 1）；Σ==0 时兜底防除零（g=0）。
        Z = np.sum(gated, axis=-1, keepdims=True)
        Z = np.where(Z > 0.0, Z, 1.0)
        g = gated / Z                            # (N, E)，行和恒为 1（不变量 M2）

        # ---- 按专家分组计算（稀疏：每个 token 只走 K 个专家）
        Wg = np.asarray(Wg)
        dff = Wg.shape[1]
        out = np.zeros((N, d), dtype=np.float64)
        token_ids: list[np.ndarray] = []
        hids: list[np.ndarray] = []
        outs: list[np.ndarray] = []
        gates: list[np.ndarray] = []
        for e in range(E):
            ids = np.flatnonzero(mask[:, e])
            token_ids.append(ids)
            if ids.size == 0:
                hids.append(np.zeros((0, dff)))
                outs.append(np.zeros((0, d)))
                gates.append(np.zeros(0))
                continue
            xe = x[ids]                                  # (n_e, d)
            gate = silu(xe @ Wg[e].T)
            up = xe @ Wu[e].T
            h = gate * up
            ye = h @ Wd[e].T                                 # FFN_e 的输出
            out[ids] += ye * g[ids, e][:, None]
            hids.append(h)
            outs.append(ye)
            gates.append(g[ids, e])

        shared_h = None
        if self.shared and sWg is not None:
            sg = silu(x @ np.asarray(sWg).T)
            su = x @ np.asarray(sWu).T
            shared_h = sg * su
            out += shared_h @ np.asarray(sWd).T

        ctx.put(x=x, router=router, Wg=Wg, Wu=Wu, Wd=Wd, sWg=sWg, sWu=sWu, sWd=sWd,
                logits=logits, s=s, mask=mask, g=g, Z=Z, sel=sel,
                token_ids=token_ids, hids=hids, outs=outs, gates=gates, shared_h=shared_h,
                N=N, d=d, dff=dff, E=E)
        return out

    def backward(self, ctx: Ctx, grad_out) -> tuple:
        x, router, Wg, Wu, Wd = ctx["x"], ctx["router"], ctx["Wg"], ctx["Wu"], ctx["Wd"]
        sWg, sWu, sWd = ctx["sWg"], ctx["sWu"], ctx["sWd"]
        s, mask, g, Z = ctx["s"], ctx["mask"], ctx["g"], ctx["Z"]
        token_ids, hids, outs, gates = ctx["token_ids"], ctx["hids"], ctx["outs"], ctx["gates"]
        shared_h, N, d, dff, E = ctx["shared_h"], ctx["N"], ctx["d"], ctx["dff"], ctx["E"]
        G = np.asarray(grad_out, dtype=np.float64)

        dx = np.zeros((N, d), dtype=np.float64)
        dWg = np.zeros_like(Wg)
        dWu = np.zeros_like(Wu)
        dWd = np.zeros_like(Wd)
        dg = np.zeros((N, E), dtype=np.float64)

        # ---- 每个专家的 FFN 反向
        for e in range(E):
            ids = token_ids[e]
            if ids.size == 0:
                continue
            h = hids[e]
            ge = gates[e]
            xe = x[ids]
            Gy = G[ids] * ge[:, None]                      # ∂L/∂FFN_e(x)
            graw = xe @ Wg[e].T                            # gate 的线性输出
            upr = xe @ Wu[e].T                             # up 的线性输出
            dH = Gy @ Wd[e]                                # (n_e, dff) = ∂L/∂h
            dWd[e] += Gy.T @ h                             # (d, dff) —— 注意是 h，不是 dH
            dUp = dH * silu(graw)                          # h = silu(graw) ⊙ upr
            dGate = dH * upr
            dsilu = sigmoid(graw) * (1.0 + graw * (1.0 - sigmoid(graw)))
            dGraw = dGate * dsilu
            dWg[e] += dGraw.T @ xe
            dWu[e] += dUp.T @ xe
            dx[ids] += dGraw @ Wg[e] + dUp @ Wu[e]
            dg[ids, e] = np.sum(G[ids] * outs[e], axis=1)  # ∂L/∂g_e = ⟨∂L/∂out, FFN_e(x)⟩

        # ---- 共享专家
        dsWg = dsWu = dsWd = None
        if self.shared and sWg is not None:
            sWg = np.asarray(sWg); sWu = np.asarray(sWu); sWd = np.asarray(sWd)
            dH = G @ sWd                                   # (N, dff) = ∂L/∂h_shared
            dsWd = G.T @ shared_h                          # (d, dff)
            sgraw = x @ sWg.T
            sup = x @ sWu.T
            dUp = dH * silu(sgraw)                         # h = silu(sgraw) ⊙ sup
            dGate = dH * sup
            dsilu = sigmoid(sgraw) * (1.0 + sgraw * (1.0 - sigmoid(sgraw)))
            dGraw = dGate * dsilu
            dsWg = dGraw.T @ x
            dsWu = dUp.T @ x
            # ∂L/∂x 只经 SwiGLU 的两条支路；不能再加 dH @ sWd.T ——
            # 那是「out = x @ sWdᵀ」才会有的项，而这里中间隔着 SwiGLU。
            dx += dGraw @ sWg + dUp @ sWu

        # ---- 门控反向：g = s·mask / Z
        # ∂L/∂s_l = mask_l·[ dg_l/Z − (Σᵢ dg_i·g_i)/Z ]   （Z 为选中亲和度之和）
        sumdg_g = np.sum(dg * g, axis=-1, keepdims=True)   # Σᵢ dgᵢ gᵢ
        ds = mask * (dg / Z - sumdg_g / Z)
        # sigmoid 反向
        dlogits = ds * s * (1.0 - s)
        drouter = dlogits.T @ x
        dx += dlogits @ router

        return dx, drouter, dWg, dWu, dWd, None, dsWg, dsWu, dsWd


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
