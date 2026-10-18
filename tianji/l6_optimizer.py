"""L6 · 优化器 —— AdamW / Muon / Muon2 与混合路由。

站在谁之上（零重写）：

┌──────────┬──────────────────────────────────────────┬──────────────────────────────────┐
│ 优化器   │ 来源                                      │ 定位                             │
├──────────┼──────────────────────────────────────────┼──────────────────────────────────┤
│ AdamW    │ Loshchilov & Hutter 2019                  │ 1D 参数（bias/gain）、embedding、 │
│          │                                           │ LM head 的默认选择               │
│ Muon     │ Keller Jordan 2024（NanoGPT speedrun）    │ 2D 隐藏权重的矩阵感知优化         │
│          │ Newton-Schulz 五次迭代，系数               │                                  │
│          │ (3.4445, −4.7750, 2.0315)                 │                                  │
│ Muon2    │ Zhang et al. 2026-04, arXiv:2604        │ **本项目默认**：NS 迭代 −40%，    │
│          │ "Muon2: Boosting Muon via Adaptive        │ 3 步达到 Muon 5 步的正交化质量    │
│          │  Second-Moment Preconditioning"           │                                  │
└──────────┴──────────────────────────────────────────┴──────────────────────────────────┘

为什么默认选 Muon2 而不是 Muon：
    Muon 的 Newton-Schulz 迭代要在「病态的动量矩阵」上做极近似 ——
    动量矩阵的条件数越差，NS 收敛越慢，5 步往往不够。
    Muon2 在 NS 之前先插一层 Adam 式二阶矩预条件，把谱结构修好，
    于是 3 步 NS 就达到甚至超过 Muon 5 步的「方向对齐度」。
    论文在 60M~15.5B 的 GPT/LLaMA 预训练上一致优于 Muon。
    这是「用更好的数学换更少计算」的典型案例，符合本项目的一贯取向。

为什么必须混合（Muon 管 2D、AdamW 管其余）：
    Muon 的正交化对 embedding / LM head 是有害的 ——
    这两处的梯度语义是「查表式的稀疏更新」，正交化会破坏其各向异性结构。
    所有公开的大规模实践（Moonshot/Kimi、Keller Jordan 本人）都明确要求
    embedding、LM head、bias、RMSNorm gain 走 AdamW。

不变量（门禁复检）：
    O1 Newton-Schulz 输出奇异值落在 (0.5, 1.5)（Muon 的公开验收标准，
       G19 门禁机检 —— 修复前 l0_numerics 宣称「N3 门禁守着」但门禁不存在，
       属于宣称-门禁不一致，已补齐）
    O2 Muon2 在 3 步 NS 下的正交化质量 >= Muon 在 5 步下的质量
       （用「与真极分解 UVᵀ 的余弦相似度」度量）
    O3 AdamW 的 decoupled weight decay：纯权重衰减项不受梯度缩放影响
       （G19 机检：梯度 ×1000 后更新逐位一致到 1e-5 内）
    O4 优化器更新后损失下降（合成凸问题上单调）
    O5 同 seed 两次运行逐位一致（确定性）
    O6 非有限梯度（NaN/Inf）不污染参数：HybridOptimizer 按 GradScaler
       语义跳过整个 step，参数逐位不变、skipped_steps 计数（G19 机检）
    O7 Muon2 无 Nesterov 变体：Newton-Schulz 首步谱归一化使预条件方向上的
       任何常数缩放被完全擦除（机检 NS(1.9·G) vs NS(G) = 1.7e-08；对照
       Muon 的真实 Nesterov 差异 3.9e-01）。早期实现有一个 `P = P + beta*P`
       的「Nesterov 前瞻」分支，数学上等于 P *= (1+beta)，被 NS 归一化
       完全擦除 —— 假功能，已删除。注意：这类常数缩放在行为上**不可检测**
       （这正是 no-op 的定义），G19 门禁 (d) 检查的是「更新匹配文档纯公式」，
       捕获的是真实算法漂移（矩更新/预条件/scale 被改错）
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .l0_numerics import newton_schulz, clip_grad_norm

__all__ = ["AdamW", "Muon", "Muon2", "HybridOptimizer", "orthogonalization_quality"]


def orthogonalization_quality(G: np.ndarray, O: np.ndarray) -> float:
    """正交化质量：O 与真极分解 UVᵀ 的方向对齐度（余弦相似度）。

    为什么用这个指标而不是「奇异值与 1 的距离」：
    Muon 的更新方向才是影响训练的东西，极分解的 UVᵀ 才是理论目标；
    奇异值是否恰好为 1 反而不重要（Muon 的公开说明：0.5~1.5 就够）。
    """
    U, _, Vt = np.linalg.svd(np.asarray(G, dtype=np.float64), full_matrices=False)
    target = U @ Vt
    num = float(np.sum(O * target))
    den = float(np.sqrt(np.sum(O ** 2) * np.sum(target ** 2))) + 1e-12
    return num / den


class AdamW:
    """解耦权重衰减的 Adam。

    为什么用 decoupled（AdamW）而不是 L2 正则加到梯度上（Adam）：
    Adam 的自适应学习率会把 L2 梯度一起缩放，导致权重衰减强度随梯度量级漂移。
    """

    def __init__(self, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.lr = float(lr)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, p in params.items():
            g = grads.get(k)
            if g is None:
                continue
            m = self.m.get(k)
            v = self.v.get(k)
            if m is None:
                m = np.zeros_like(p)
                v = np.zeros_like(p)
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * (g * g)
            self.m[k], self.v[k] = m, v
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            p -= self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.wd * p)


class Muon:
    """Muon：SGD-momentum + Newton-Schulz 正交化。

    更新：M ← βM + G（Nesterov 变体下 G ← βM + G）
          O ← NewtonSchulz(M, steps)
          W ← W − lr·O·scale
    scale = max(1, rows/cols)^0.5，用于补偿长宽比（Moonshot 的实现细节）。
    """

    name = "Muon"

    def __init__(self, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        self.lr = float(lr)
        self.beta = float(momentum)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self.wd = float(weight_decay)
        self.m: dict[str, np.ndarray] = {}

    def _pre(self, k: str, g: np.ndarray) -> np.ndarray:
        m = self.m.get(k)
        if m is None:
            m = np.zeros_like(g)
        m = self.beta * m + g
        self.m[k] = m
        return (g + self.beta * m) if self.nesterov else m

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for k, p in params.items():
            g = grads.get(k)
            if g is None or g.ndim < 2:
                continue
            u = self._pre(k, g)
            # ndim > 2（例如 MoE 专家权重 (E, dff, d)）先按 (首维, -1) 展平，
            # 正交化后再还原 —— 与 HunyuanVideo / Muon 官方实现一致。
            shape = p.shape
            u2 = u.reshape(shape[0], -1) if p.ndim > 2 else u
            O = newton_schulz(u2, steps=self.ns_steps).reshape(shape)
            scale = max(1.0, shape[0] / max(int(np.prod(shape[1:])), 1)) ** 0.5
            p -= self.lr * O * scale
            if self.wd:
                p -= self.lr * self.wd * p


class Muon2(Muon):
    """Muon2：在 Newton-Schulz 之前插入 Adam 式二阶矩预条件。

        M ← β₁M + (1−β₁)G
        V ← β₂V + (1−β₂)G²
        P ← M̂ / (√V̂ + ε)          ← 关键新增：修好动量矩阵的谱
        O ← NewtonSchulz(P, steps=3)
        W ← W − lr·O·scale

    为什么这样能少 40% 的 NS 迭代：
        NS 迭代本质是在求极分解的近似，其收敛速度取决于输入矩阵的条件数。
        动量矩阵 G 通常是重尾、病态的；二阶矩预条件把各方向的量级拉平，
        等价于是给 NS 一个更好的起点。论文用「方向对齐度」量化了这一点。

    为什么没有 Nesterov 变体（G19 门禁钉死的性质）：
        Muon 的 Nesterov（方向 = g + β·m）是真实的前瞻外推，机检 T/F
        更新差异 3.9e-01。但 Muon2 早期实现曾有一个对应分支
        `P = P + β·P`，数学上只是 P *= (1+β) 常数缩放 —— 而
        Newton-Schulz 首步即按谱范数归一（l0_numerics.newton_schulz），
        常数缩放被完全擦除（机检：NS(1.9·G) vs NS(G) = 1.7e-08）。
        即该分支是零效果假功能，已删除。此类常数缩放在行为上不可检测
        （no-op 的数学必然），G19 门禁 (d) 因此检查「更新匹配文档纯公式」，
        捕获真实算法漂移而非自欺欺人的 T/F 对照。
    """

    name = "Muon2"

    def __init__(self, lr=0.02, betas=(0.9, 0.95), eps=1e-8,
                 ns_steps=3, weight_decay=0.0):
        super().__init__(lr=lr, momentum=betas[0], nesterov=False,
                         ns_steps=ns_steps, weight_decay=weight_decay)
        self.b2 = float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, p in params.items():
            g = grads.get(k)
            if g is None or g.ndim < 2:
                continue
            m = self.m.get(k)
            v = self.v.get(k)
            if m is None:
                m = np.zeros_like(g)
                v = np.zeros_like(g)
            m = self.beta * m + (1 - self.beta) * g
            v = self.b2 * v + (1 - self.b2) * (g * g)
            self.m[k], self.v[k] = m, v
            mhat = m / (1 - self.beta ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            P = mhat / (np.sqrt(vhat) + self.eps)
            shape = p.shape
            P2 = P.reshape(shape[0], -1) if p.ndim > 2 else P
            O = newton_schulz(P2, steps=self.ns_steps).reshape(shape)
            scale = max(1.0, shape[0] / max(int(np.prod(shape[1:])), 1)) ** 0.5
            p -= self.lr * O * scale
            if self.wd:
                p -= self.lr * self.wd * p


class HybridOptimizer:
    """按参数角色路由到 Muon2（2D 隐藏权重）或 AdamW（其余）。

    路由规则（与 Moonshot / Keller Jordan 的公开配置一致）：
        - ndim >= 2 且不在 ADAM_ONLY 名单中  → Muon2
        - 其余（1D、embedding、LM head、gain）→ AdamW

    为什么用名单而不是纯维度规则：embedding 与 lm_head 虽然也是 2D，
    但它们的梯度是稀疏查表式的，正交化会破坏语义结构 ——
    这个例外必须显式声明，不能靠维度猜。

    训练安全性（GradScaler 语义）：全局梯度范数非有限（NaN/Inf）时
    跳过整个 step（参数逐位不变，skipped_steps 累计）—— 否则一个坏
    梯度会经 clip 缩放因子把全部参数静默污染成 NaN。
    """

    ADAM_ONLY_EXACT = ("tok_emb", "lm_head")
    ADAM_ONLY_SUFFIX = ("norm1_w", "norm2_w", "final_norm_w", "conv_w")

    def __init__(self, muon2: Muon2 | None = None, adamw: AdamW | None = None,
                 max_grad_norm: float = 1.0):
        self.muon = muon2 if muon2 is not None else Muon2()
        self.adam = adamw if adamw is not None else AdamW()
        self.max_grad_norm = float(max_grad_norm)
        self.last_grad_norm = 0.0
        self.skipped_steps = 0

    def is_muon(self, name: str, p: np.ndarray) -> bool:
        if p.ndim < 2:
            return False
        base = name.split(".")[-1]
        if base in self.ADAM_ONLY_EXACT or name in self.ADAM_ONLY_EXACT:
            return False
        if any(base.endswith(s) or name.endswith(s) for s in self.ADAM_ONLY_SUFFIX):
            return False
        return True

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        # errstate：非有限梯度下 clip 内部会出现 inf*0=nan 的浮点警告，
        # 语义上无害（紧接着就被守卫跳过），但不该污染训练日志
        with np.errstate(invalid="ignore", over="ignore"):
            grads, norm = clip_grad_norm(grads, self.max_grad_norm)
        self.last_grad_norm = norm
        # ---- 非有限梯度守卫（O6，G19 机检）
        # 一个 NaN/Inf 梯度会让 clip 的缩放因子变成 nan/inf，进而把**全部**
        # 参数静默污染成非有限值 —— 不报错、不崩溃，模型无声死亡。
        # 语义对齐 PyTorch AMP GradScaler：检测到非有限梯度时跳过整个 step，
        # 参数逐位不变，只累计 skipped_steps 供上层观测/报警。
        if not np.isfinite(norm):
            self.skipped_steps += 1
            return
        muon_p, muon_g = {}, {}
        adam_p, adam_g = {}, {}
        for k, p in params.items():
            g = grads.get(k)
            if g is None:
                continue
            if self.is_muon(k, p):
                muon_p[k], muon_g[k] = p, g
            else:
                adam_p[k], adam_g[k] = p, g
        if muon_p:
            self.muon.step(muon_p, muon_g)
        if adam_p:
            self.adam.step(adam_p, adam_g)
