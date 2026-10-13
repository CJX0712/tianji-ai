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
    """默认训练语料：TINY_TALE 重复若干次，约 20KB。"""
    return TINY_TALE * 6


def markov_tokens(n: int, vocab: int, seed: int = 0, order: int = 2) -> np.ndarray:
    """确定性合成的 order 阶马尔可夫 token 序列。

    为什么基准用它：有明确的熵下界，因此「loss 降到多少才算学会」是可判定的。
    真实文本没有这个性质（你无法事先知道最优 loss 是多少）。
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
