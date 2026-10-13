"""L0 · 数值内核 —— 一切可信度的地基。

为什么需要独立的内核层：
    混合架构（Gated DeltaNet + MLA + MoE）里，数值失效不是「结果差一点」，
    而是「梯度静默变 NaN / 递推状态爆掉 / 量化后困惑度崩塌」。
    把稳定原语集中在一层，上层模块就不允许各自发明 exp / 归一化 / 正交化。

站在谁之上（零重写）：
    - logsumexp / softmax 稳定化      —— 经典 LAPACK 数值习惯，numpy 同款
    - Newton-Schulz 正交化            —— Keller Jordan Muon (2024) 的五步五次迭代
                                         系数 (3.4445, -4.7750, 2.0315) 取自其公开实现
    - BLAKE2b 派生的子流种子           —— 天枢 tianshu-ai L0 已验证的做法：
                                         相邻整数种子存在相关结构，会虚高多样性估计
    - 谱范数 via 幂迭代                —— Golub & Van Loan, Matrix Computations

不变量（L10 门禁会在每次交付时复检）：
    N1 softmax 行和恒为 1，且平移不变：softmax(x + c) == softmax(x)
    N2 logsumexp 与 naive log(sum(exp)) 在安全区一致到 1e-12
    N3 Newton-Schulz 输出的奇异值落在 (0.5, 1.5) 区间且接近 1
    N4 spawn_seed 派生的子流两两不相关（|Pearson r| < 0.02）
    N5 幂迭代谱范数与 numpy.linalg.svd 完全一致（1e-10）
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "stable_hash",
    "spawn_seed",
    "rng",
    "logsumexp",
    "softmax",
    "log_softmax",
    "spectral_norm",
    "newton_schulz",
    "rms",
    "topk",
    "topk_mask",
    "safe_div",
    "clip_grad_norm",
    "count_params",
]

# Muon 公开的五次 Newton-Schulz 系数（最大化 0 点斜率）
NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


# ---------------------------------------------------------------- 确定性

def stable_hash(obj: object) -> int:
    """跨进程稳定的哈希。

    为什么不用内置 hash()：Python 对 str 的 hash 受 PYTHONHASHSEED 随机化，
    跨进程不一致，绝不能用作持久化缓存键或种子派生源。
    """
    if isinstance(obj, bytes):
        payload = obj
    elif isinstance(obj, str):
        payload = obj.encode("utf-8")
    else:
        payload = repr(obj).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def spawn_seed(master: int, *tags: object) -> int:
    """从主种子派生互不相关的子流种子。

    为什么不用 master + i：相邻整数种子的 RNG 存在已知相关结构，
    会让 HPO / 多 seed 基准的不同 trial 共享伪随机结构，虚高结论可信度。
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(int(master).to_bytes(8, "big"))
    for t in tags:
        h.update(b"|")
        h.update(stable_hash(t).to_bytes(8, "big"))
    return int.from_bytes(h.digest(), "big")


def rng(seed: int) -> np.random.Generator:
    """PCG64 —— 比 Mersenne Twister 统计性质更好且更快，numpy 官方推荐默认。"""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- 稳定函数

def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """数值稳定的 log-sum-exp。

    log Σ exp(xᵢ) = m + log Σ exp(xᵢ - m)，m = max(xᵢ)。
    不做这个平移，x 里出现 800 就会 inf，梯度全 NaN。
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    out = m_safe + np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """平移不变、行和恒为 1 的 softmax。"""
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x - logsumexp(x, axis=axis, keepdims=True)


def safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return num / (den + eps)


# ---------------------------------------------------------------- 线性代数

def spectral_norm(A: np.ndarray, iters: int = 16, eps: float = 1e-12) -> float:
    """幂迭代求谱范数（最大奇异值）。

    为什么不用完整 SVD：Newton-Schulz 每步都调用，SVD 的 O(min(n,m)·nm) 太贵；
    幂迭代 16 步在 float64 下已收敛到机器精度量级（N5 门禁验证）。
    """
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return 0.0
    m, n = A.shape
    b = np.ones(n, dtype=np.float64) / max(n, 1) ** 0.5
    for _ in range(iters):
        v = A @ b
        nv = np.linalg.norm(v)
        if nv < eps:
            return 0.0
        v = v / nv
        b = A.T @ v
        nb = np.linalg.norm(b)
        if nb < eps:
            return 0.0
        b = b / nb
    return float(np.linalg.norm(A @ b))


def newton_schulz(
    G: np.ndarray,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: Sequence[float] = NS_COEFFS,
) -> np.ndarray:
    """Newton-Schulz 正交化：把矩阵推向最近的半正交矩阵 UVᵀ。

    迭代式（五次/quintic，Keller Jordan Muon）：
        A = X Xᵀ ;  B = bA + cA·A ;  X = aX + B X
    收敛后 X 的奇异值趋近 1（实践上落在 0.5~1.5 已足够，N3 门禁守着）。

    为什么用它而不是 SVD 求极分解：
        SVD 在 float32/bfloat16 下不稳定，且 GPU 上慢；NS 只需要矩阵乘，
        可在低精度跑，FLOP 开销 < 1%（Moonshot 在 405B 规模实测 0.5%）。

    为什么转置：要求 size(-2) <= size(-1)，让 X Xᵀ 是小的那个 Gram 矩阵。
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects 2D, got {G.shape}")
    a, b, c = coeffs
    X = G.copy()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    # 归一化到谱范数 <= 1，否则迭代不收敛
    X = X / (spectral_norm(X) + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def rms(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Root-mean-square，RMSNorm 的分母。"""
    return np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)), axis=axis) + eps)


def topk(a: np.ndarray, k: int, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """返回 (values, indices)，按降序。k 超出维度时截断。"""
    a = np.asarray(a)
    k = int(min(k, a.shape[axis]))
    idx = np.argpartition(-a, k - 1, axis=axis)
    # argpartition 只保证前 k 个是候选，需再排序以保证「降序」这一契约
    top_idx = np.take(idx, np.arange(k), axis=axis)
    top_val = np.take_along_axis(a, top_idx, axis=axis)
    order = np.argsort(-top_val, axis=axis)
    top_val = np.take_along_axis(top_val, order, axis=axis)
    top_idx = np.take_along_axis(top_idx, order, axis=axis)
    return top_val, top_idx


def topk_mask(scores: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Top-K 的 0/1 掩码（MoE 路由用）。"""
    _, idx = topk(scores, k, axis=axis)
    mask = np.zeros_like(scores, dtype=np.float64)
    np.put_along_axis(mask, idx, 1.0, axis=axis)
    return mask


def clip_grad_norm(grads: dict[str, np.ndarray], max_norm: float) -> tuple[dict[str, np.ndarray], float]:
    """全局梯度范数裁剪。

    为什么必须裁剪：MoE + 深层混合架构的梯度分布是重尾的，
    一步尖峰能把 RMSNorm 增益和 router 偏置同时打飞，损失再也回不来。
    """
    total = 0.0
    for g in grads.values():
        if g is not None:
            total += float(np.sum(np.square(g)))
    total = float(np.sqrt(total))
    if max_norm and total > max_norm:
        scale = max_norm / (total + 1e-12)
        return {k: (v * scale if v is not None else None) for k, v in grads.items()}, total
    return grads, total


def count_params(params: Iterable[tuple[str, np.ndarray]]) -> int:
    return int(sum(np.asarray(p).size for _, p in params))
