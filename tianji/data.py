"""内置语料 —— 全部离线，不依赖任何下载。

为什么自带语料而不是去下载 TinyShakespeare / WikiText：
    「可复现」是硬约束。一旦依赖外部下载，镜像变更、网络失败、
    许可证变化都会让 `make test` 在某天突然跑不起来。
    自带语料让仓库在完全断网的机器上也能 100% 跑通。

自带两份：
    1. TINY_TALE  —— 极小的英文/中文混合文本，用于冒烟与教学
    2. 合成马尔可夫语料生成器 —— 用于基准（有明确的熵下界，可判定收敛）
"""

from __future__ import annotations

import numpy as np

from .l0_numerics import rng, spawn_seed

__all__ = ["TINY_TALE", "default_corpus", "markov_tokens"]


TINY_TALE = (
    "the quick brown fox jumps over the lazy dog. "
    "the lazy dog barks at the quick brown fox. "
    "to be or not to be, that is the question. "
    "all that glitters is not gold, but gold glitters always. "
    "a journey of a thousand miles begins with a single step. "
    "天玑是世界上最好的东西，晨星在深圳写代码。 "
    "启明星是天亮前最后还亮着的那颗星。 "
    "零重写，集大成，闭环交付。"
) * 40


def default_corpus() -> str:
    """默认训练语料：TINY_TALE 重复 6 次，约 90KB（71280 字符，UTF-8 93.5KB）。

    早期 docstring 写「约 20KB」是拍脑袋的，实测 93.5KB（G20 门禁机检
    大小与 roundtrip，防止文档与实现悄悄漂移）。
    """
    return TINY_TALE * 6


def markov_tokens(n: int, vocab: int, seed: int = 0, order: int = 2) -> np.ndarray:
    """确定性合成的 order 阶马尔可夫 token 序列。

    为什么基准用它：有明确的熵下界，因此「loss 降到多少才算学会」是可判定的。
    真实文本没有这个性质（你无法事先知道最优 loss 是多少）。

    熵下界由 :func:`markov_entropy_rate` 算出并机检（门禁 G24）。
    """
    g = rng(spawn_seed(seed, "synth"))
    if order == 2:
        trans = g.dirichlet(np.ones(vocab) * 0.25, size=(vocab, vocab))
    else:
        trans = g.dirichlet(np.ones(vocab) * 0.5, size=(vocab,))
    out = np.zeros(n, dtype=np.int64)
    a, b = 0, 1
    for i in range(n):
        p = trans[a, b] if order == 2 else trans[b]
        c = g.choice(vocab, p=p)
        out[i] = c
        a, b = b, c
    return out


def markov_entropy_rate(vocab: int = 64, seed: int = 0, order: int = 2,
                       n_sample: int = 200_000, tol: float = 5e-3) -> dict:
    """计算 order 阶马尔可夫合成源的熵率 H，并用两套独立算法交叉验证。

    熵率 ``H = H(X_t | X_{t-1}, ..., X_{t-order+1})`` 是「模型降到多少才算学会」
    的可判定下界：任何模型在该源上的交叉熵都不可能低于 ``exp(H)``（Gibbs 不等式）。

    两套独立算法（门禁 G24 守着一致性）：

    * 方法 A · 解析平稳分布 ——
      ``order=2``：把 2 阶链视为 ``V²`` 个状态 ``(a,b)`` 上的 1 阶链，
      功率迭代求平稳分布 ``π``，``H_A = Σ_{a,b} π(a,b)·H(trans[a,b,:])``；
      ``order=1``：``V`` 个状态上的 1 阶链，``H_A = Σ_b π(b)·H(trans[b,:])``。
    * 方法 B · oracle 交叉熵 ——
      用「已知真实转移」在长样本上算交叉熵
      ``H_B = -mean log P(next | context)``，其期望恰为 ``H``。

    二者一致 ⇒ H 是被正确算出来的（而非编造）；且恒有 ``0 < H < log(vocab)``。

    返回 dict：``H, H_analytic, H_oracle, abs_diff, vocab, order, log_vocab,
    consistent, tol``。
    """
    V = int(vocab)
    g = rng(spawn_seed(seed, "synth"))
    if order == 2:
        trans = g.dirichlet(np.ones(V) * 0.25, size=(V, V))   # (V,V,V)
    elif order == 1:
        trans = g.dirichlet(np.ones(V) * 0.5, size=(V,))       # (V,V)
    else:
        raise ValueError(f"markov_entropy_rate 仅支持 order ∈ {{1,2}}，得到 {order}")

    with np.errstate(divide="ignore"):
        logp = np.log(trans + 1e-300)

    if order == 2:
        Hcond = -np.sum(trans * logp, axis=2)                  # (V,V)
        # 功率迭代求 (a,b) 上的平稳分布（链不可约且非周期 ⇒ 唯一平稳分布）
        pi = np.full((V, V), 1.0 / (V * V), dtype=np.float64)
        for _ in range(2000):
            pi_next = np.einsum("ab,abc->bc", pi, trans)
            s = pi_next.sum()
            if s > 0:
                pi_next /= s
            if np.max(np.abs(pi_next - pi)) < 1e-13:
                pi = pi_next
                break
            pi = pi_next
        H_A = float(np.sum(pi * Hcond))
    else:  # order == 1
        Hcond = -np.sum(trans * logp, axis=1)                  # (V,)
        pi = np.full(V, 1.0 / V, dtype=np.float64)
        for _ in range(2000):
            pi_next = trans.T @ pi                             # π' = Mᵀ π
            s = pi_next.sum()
            if s > 0:
                pi_next /= s
            if np.max(np.abs(pi_next - pi)) < 1e-13:
                pi = pi_next
                break
            pi = pi_next
        H_A = float(np.sum(pi * Hcond))

    # 方法 B：oracle 交叉熵（长样本，期望恰为 H）
    sample = markov_tokens(n_sample, V, seed, order)
    if order == 2:
        a = sample[:-2]; b = sample[1:-1]; c = sample[2:]
        lc = np.log(trans[a, b, c] + 1e-300)
    else:
        b = sample[:-1]; c = sample[1:]
        lc = np.log(trans[b, c] + 1e-300)
    H_B = float(-lc.mean())

    log_vocab = float(np.log(V))
    abs_diff = abs(H_A - H_B)
    consistent = bool(abs_diff < tol and 0.0 < H_A < log_vocab and 0.0 < H_B < log_vocab)
    return {
        "H": (H_A + H_B) / 2.0,
        "H_analytic": H_A,
        "H_oracle": H_B,
        "abs_diff": abs_diff,
        "vocab": V,
        "order": order,
        "log_vocab": log_vocab,
        "consistent": consistent,
        "tol": float(tol),
    }
