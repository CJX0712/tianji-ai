"""快速冒烟：对每个原子模块做中心差分梯度检验。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tianji.l1_contract import gradcheck
from tianji.l3_primitives import Linear, RMSNorm, RoPE, SwiGLU, CausalAttention, CrossEntropy, rope_tables
from tianji.l4_deltanet import GatedDeltaNet, CausalConv1D, L2Norm
from tianji.l5_moe import MoELayer

g = np.random.default_rng(0)


def T(name, module, args, tol=1e-6):
    rep, _, _ = gradcheck(module, args, seed=1, tol=tol, name=name)
    print(rep)
    return rep.passed


ok = []
B, Tn, d, H = 2, 5, 8, 2

# Linear
x = g.standard_normal((B, Tn, d)); W = g.standard_normal((6, d)); b = g.standard_normal(6)
ok.append(T("Linear", Linear(), [x, W, b]))

# RMSNorm
x = g.standard_normal((B, Tn, d)); w = g.standard_normal(d) + 1.0
ok.append(T("RMSNorm", RMSNorm(), [x, w]))

# RoPE
D = 8
cos, sin = rope_tables(D, 16)
x = g.standard_normal((B, H, Tn, D))
ok.append(T("RoPE", RoPE(), [x, cos[:Tn], sin[:Tn]]))

# SwiGLU
dff = 10
x = g.standard_normal((B, Tn, d)); Wg = g.standard_normal((dff, d)); Wu = g.standard_normal((dff, d)); Wd = g.standard_normal((d, dff))
ok.append(T("SwiGLU", SwiGLU(), [x, Wg, Wu, Wd]))

# CausalAttention
D = 8
q = g.standard_normal((B, H, Tn, D)); k = g.standard_normal((B, H, Tn, D)); v = g.standard_normal((B, H, Tn, D))
ok.append(T("CausalAttn-GQA", CausalAttention(), [q, k, v]))

# CrossEntropy
V = 7
logits = g.standard_normal((B, Tn, V)); tgt = g.integers(0, V, (B, Tn))
ok.append(T("CrossEntropy", CrossEntropy(), [logits, tgt]))

# L2Norm
x = g.standard_normal((B, H, Tn, D))
ok.append(T("L2Norm", L2Norm(), [x]))

# CausalConv1D
x = g.standard_normal((2, 6, 4)); w = g.standard_normal((4, 3))
ok.append(T("CausalConv1D", CausalConv1D(3), [x, w]))

# GatedDeltaNet
B2, H2, T2, D2 = 2, 2, 6, 4
q = g.standard_normal((B2, H2, T2, D2)); k = g.standard_normal((B2, H2, T2, D2)); v = g.standard_normal((B2, H2, T2, D2))
beta = sigmoid_rand = 1 / (1 + np.exp(-g.standard_normal((B2, H2, T2))))
alpha = 1 / (1 + np.exp(-g.standard_normal((B2, H2, T2))))
ok.append(T("GatedDeltaNet", GatedDeltaNet(), [q, k, v, beta, alpha]))

# MoE
E, K, dm, dffm, N = 4, 2, 6, 8, 7
x = g.standard_normal((N, dm)); router = g.standard_normal((E, dm))
Wg = g.standard_normal((E, dffm, dm)); Wu = g.standard_normal((E, dffm, dm)); Wd = g.standard_normal((E, dm, dffm))
bias = np.zeros(E)
sWg = g.standard_normal((dffm, dm)); sWu = g.standard_normal((dffm, dm)); sWd = g.standard_normal((dm, dffm))
ok.append(T("MoELayer", MoELayer(E, K, True), [x, router, Wg, Wu, Wd, bias, sWg, sWu, sWd]))

print(f"\n{sum(ok)}/{len(ok)} passed")
sys.exit(0 if all(ok) else 1)
