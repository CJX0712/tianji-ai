"""L9 · 对齐 —— DPO / IPO / SimPO / KTO。

站在谁之上（零重写）：
    - DPO   Rafailov et al. 2023, "Direct Preference Optimization"：
            把 RLHF 的「奖励模型 + PPO」两步压缩成一步分类损失，
            关键恒等式是 reward = β·log(π/π_ref)，从而无需显式奖励模型。
    - IPO   Azar et al. 2024：把 DPO 的 logistic 换成平方损失，
            修掉 DPO 在「确定性偏好」下会无限拉大 logit 差的病态。
    - SimPO Meng et al. 2024：去掉参考模型，改用长度归一化的平均 logprob，
            消除「长回答天然 logprob 更负」带来的长度偏置。
    - KTO   Ethayarajh et al. 2024：只需要 (x, y, 好/坏) 的二元标注，
            不需要成对偏好，用前景理论式的价值函数。

本实现的共同骨架：
    三者都只是「logits 的标量可微函数」，因此统一走
    model.backward(g_logits=...) 这一个入口，复用同一套反传。
    这也是为什么 L1 要求 backward 必须从任意 logits 梯度起步。

不变量（门禁复检）：
    A1 DPO 损失的梯度方向与偏好一致：chosen 的 logprob 上升、rejected 下降
    A2 DPO 在 Δ=0（无偏好）时损失 == log 2，且梯度不为零
    A3 IPO 的最优解是 Δ = 1/(2β)，而不是无穷大（DPO 的病态由此修掉）
    A4 SimPO 的长度归一化：把 chosen 复制成两倍长，平均 logprob 不变
    A5 参考模型在对齐过程中参数严格不变（冻结检查）
"""

from __future__ import annotations

import numpy as np

from .l0_numerics import log_softmax

__all__ = ["seq_logprob", "mean_logprob", "grad_logprob", "dpo_terms", "ipo_terms",
           "simpo_terms", "kto_terms", "Aligner", "freeze_copy"]


def seq_logprob(logits: np.ndarray, targets: np.ndarray) -> float:
    """序列的 log π(y|x) = Σ_t log π(y_t | y_<t)。"""
    lp = log_softmax(np.asarray(logits, dtype=np.float64), axis=-1)
    tgt = np.asarray(targets)
    B, T = tgt.shape
    flat = lp.reshape(B * T, -1)
    idx = np.where(tgt.reshape(-1) >= 0, tgt.reshape(-1), 0)
    m = (tgt.reshape(-1) >= 0).astype(np.float64)
    return float(np.sum(flat[np.arange(B * T), idx] * m))


def mean_logprob(logits: np.ndarray, targets: np.ndarray) -> float:
    """长度归一化的平均 logprob（SimPO 用）。"""
    tgt = np.asarray(targets)
    n = max(int((tgt >= 0).sum()), 1)
    return seq_logprob(logits, targets) / n


def grad_logprob(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """∂(Σ log π)/∂logits = onehot(y) − softmax(logits)。

    与 L3 的 CrossEntropy 反向同源 —— 归一化指数族的 Jacobian 就是
    (I − p p 的行的形式)：d log p_k / d z_v = δ_kv − p_v。
    """
    logits = np.asarray(logits, dtype=np.float64)
    lp = log_softmax(logits, axis=-1)
    P = np.exp(lp)
    tgt = np.asarray(targets)
    B, T, V = logits.shape
    g = -P.reshape(B * T, V).copy()
    idx = np.where(tgt.reshape(-1) >= 0, tgt.reshape(-1), 0)
    g[np.arange(B * T), idx] += 1.0
    g *= (tgt.reshape(-1) >= 0).astype(np.float64)[:, None]
    return g.reshape(B, T, V)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


def dpo_terms(lp_p_c: float, lp_p_r: float, lp_ref_c: float, lp_ref_r: float,
              beta: float = 0.1) -> tuple[float, float]:
    """返回 (loss, dΔ系数)。Δ = (lp_p^c − lp_ref^c) − (lp_p^r − lp_ref^r)。

    loss = −log σ(βΔ)；∂loss/∂Δ = −β·σ(−βΔ)
    """
    delta = (lp_p_c - lp_ref_c) - (lp_p_r - lp_ref_r)
    loss = -np.log(_sig(beta * delta) + 1e-12)
    return float(loss), float(-beta * _sig(-beta * delta))


def ipo_terms(lp_p_c, lp_p_r, lp_ref_c, lp_ref_r, beta: float = 0.1,
              tau: float = 0.1) -> tuple[float, float]:
    """IPO：loss = (Δ − τ/(2β))²。

    为什么能修掉 DPO 的病态：DPO 的 logistic 损失在 Δ→∞ 时梯度→0 但
    损失也→0，因此模型会不断拉大 logit 差；平方损失把最优钉在
    Δ = τ/(2β)，是一个有限目标。
    """
    delta = (lp_p_c - lp_ref_c) - (lp_p_r - lp_ref_r)
    target = tau / (2.0 * beta)
    loss = (delta - target) ** 2
    return float(loss), float(2.0 * (delta - target))


def simpo_terms(mp_c: float, mp_r: float, beta: float = 2.0,
                gamma: float = 0.5) -> tuple[float, float]:
    """SimPO：无参考模型，用平均 logprob，并加一个固定的边界 γ。

    Δ_simpo = (β/|c|)·logπ(c) − (β/|r|)·logπ(r) − γ
    注意传入的是**平均** logprob（mean_logprob 的结果）。
    """
    delta = beta * mp_c - beta * mp_r - gamma
    loss = -np.log(_sig(delta) + 1e-12)
    return float(loss), float(-_sig(-delta))


def kto_terms(lp_p: float, lp_ref: float, desirable: bool, beta: float = 0.1) -> tuple[float, float]:
    """KTO：二元好/坏标注，价值函数 v = β·(logπ − logπ_ref) − KL 项。"""
    z = beta * (lp_p - lp_ref)
    if desirable:
        loss = -np.log(_sig(z) + 1e-12)
        return float(loss), float(-beta * _sig(-z))
    loss = -np.log(_sig(-z) + 1e-12)
    return float(loss), float(beta * _sig(z))


def freeze_copy(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """参考模型：深拷贝且冻结。

    为什么必须深拷贝：参考模型的 logprob 在对齐过程中必须恒定，
    若与 policy 共享数组，一步更新就会让 Δ 的基准漂移，
    这是 DPO 实现里最隐蔽的一类错误（不报错，只是对齐无效）。
    """
    return {k: v.copy() for k, v in params.items()}


class Aligner:
    """偏好优化器。

    method: "dpo" | "ipo" | "simpo" | "kto"
    """

    def __init__(self, policy, ref_params: dict[str, np.ndarray] | None = None,
                 method: str = "dpo", beta: float = 0.1):
        self.policy = policy
        self.ref_params = ref_params
        self.method = method
        self.beta = float(beta)
        self.ref_model = None
        if method in ("dpo", "ipo", "kto") and ref_params is not None:
            self.ref_model = type(policy)(policy.cfg, ref_params)

    def _ref_logprob(self, x, y) -> tuple[float, float]:
        """返回 (seq_logprob, mean_logprob)。"""
        if self.ref_model is None:
            return 0.0, 0.0
        out = self.ref_model.forward(x, y)
        return seq_logprob(out["logits"], y), mean_logprob(out["logits"], y)

    def loss_and_grads(self, x_c, y_c, x_r, y_r) -> tuple[float, dict[str, np.ndarray]]:
        m = self.policy
        # ---- chosen
        out_c = m.forward(x_c, y_c)
        lp_c = seq_logprob(out_c["logits"], y_c)
        mp_c = mean_logprob(out_c["logits"], y_c)
        ref_c, refm_c = self._ref_logprob(x_c, y_c)
        # ---- rejected（必须重新前向：backward 会消费前向记录）
        out_r = m.forward(x_r, y_r)
        lp_r = seq_logprob(out_r["logits"], y_r)
        mp_r = mean_logprob(out_r["logits"], y_r)
        ref_r, refm_r = self._ref_logprob(x_r, y_r)

        if self.method == "simpo":
            loss, coef = simpo_terms(mp_c, mp_r, beta=self.beta)
            # Δ 对 chosen 的系数是 +β/|c|，对 rejected 是 −β/|r|
            nc = max(int((np.asarray(y_c) >= 0).sum()), 1)
            nr = max(int((np.asarray(y_r) >= 0).sum()), 1)
            coef_c = coef * self.beta / nc
            coef_r = -coef * self.beta / nr
        elif self.method == "ipo":
            loss, coef = ipo_terms(lp_c, lp_r, ref_c, ref_r, beta=self.beta)
            coef_c, coef_r = coef, -coef
        else:
            loss, coef = dpo_terms(lp_c, lp_r, ref_c, ref_r, beta=self.beta)
            coef_c, coef_r = coef, -coef

        # ---- 反传：先 chosen（需要重新前向，因为 out_r 已覆盖记录）
        out_c = m.forward(x_c, y_c)
        g_c = m.backward(g_logits=coef_c * grad_logprob(out_c["logits"], y_c))
        out_r = m.forward(x_r, y_r)
        g_r = m.backward(g_logits=coef_r * grad_logprob(out_r["logits"], y_r))
        G = {k: v.copy() for k, v in g_c.items()}
        for k, v in g_r.items():
            G[k] = v.copy() if k not in G else G[k] + v
        return float(loss), G
