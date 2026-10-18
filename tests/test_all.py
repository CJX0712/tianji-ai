"""pytest 测试套件 —— 分层断言，全部可独立运行。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tianji.l0_numerics import (        # noqa: E402
    logsumexp, softmax, newton_schulz, spectral_norm, spawn_seed, rng)
from tianji.l1_contract import gradcheck, Ctx, Module       # noqa: E402
from tianji.l2_tokenizer import BPE, pretokenize            # noqa: E402
from tianji.l3_primitives import (                          # noqa: E402
    Linear, RMSNorm, RoPE, SwiGLU, CausalAttention, CrossEntropy,
    rope_tables, causal_mask_rect, causal_mask)
from tianji.l4_deltanet import (        # noqa: E402
    GatedDeltaNet, CausalConv1D, L2Norm, gdn_parallel, gdn_chunkwise,
    gdn_chunkwise_backward, GatedDeltaNetChunkwise)
from tianji.l5_moe import (                                 # noqa: E402
    MoELayer, LoadBalancer, sigmoid)
from tianji.l6_optimizer import Muon, Muon2, orthogonalization_quality  # noqa: E402
from tianji.l7_train import cosine_schedule, wsd_schedule, DataLoader   # noqa: E402
from tianji.l8_infer import (                               # noqa: E402
    quantize_per_channel, quantize_groupwise, dequantize, speculative_decode)
from tianji.l10_microops import MICROOPS, FAMILIES, run_all as micro_all  # noqa: E402
from tianji.model import TianjiConfig, TianjiModel, init_params, build_layer_layout  # noqa: E402
from tianji.verify import run_all_gates                     # noqa: E402


# ============================================================ L0 数值

def test_softmax_row_sum_and_shift_invariance():
    x = rng(0).standard_normal((3, 7)) * 20
    p = softmax(x, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-12)
    assert np.allclose(p, softmax(x + 123.0, axis=-1), atol=1e-12)


def test_logsumexp_matches_naive_in_safe_region():
    x = rng(1).standard_normal((4, 5))
    naive = np.log(np.sum(np.exp(x), axis=-1))
    assert np.allclose(logsumexp(x, axis=-1), naive, atol=1e-12)


def test_logsumexp_no_overflow():
    x = np.full((2, 3), 800.0)
    assert np.all(np.isfinite(logsumexp(x, axis=-1)))


def test_newton_schulz_singular_values_in_band():
    g = rng(2).standard_normal((16, 12))
    O = newton_schulz(g, steps=5)
    sv = np.linalg.svd(O, compute_uv=False)
    assert 0.5 <= sv.min() and sv.max() <= 1.5


def test_spawn_seed_streams_uncorrelated():
    a = rng(spawn_seed(0, "a")).standard_normal(2000)
    b = rng(spawn_seed(0, "b")).standard_normal(2000)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.05


# ============================================================ L1 反向契约

def _mk_inputs(seed, D=8):
    g = rng(seed)
    return (g.standard_normal((2, 5, D)), g.standard_normal((6, D)),
            g.standard_normal(6))


@pytest.mark.parametrize("mod,args", [
    (Linear(), None),
])
def test_gradcheck_linear(mod, args):
    rep, _, _ = gradcheck(Linear(), list(_mk_inputs(3)), seed=1, name="Linear")
    assert rep.passed, rep


def test_gradcheck_all_primitives():
    g = rng(7)
    D = 8
    cos, sin = rope_tables(D, 16)
    cases = [
        (RMSNorm(), [g.standard_normal((2, 5, D)), g.standard_normal(D) + 1]),
        (RoPE(), [g.standard_normal((2, 2, 5, D)), cos[:5], sin[:5]]),
        (SwiGLU(), [g.standard_normal((2, 5, D)), g.standard_normal((10, D)),
                    g.standard_normal((10, D)), g.standard_normal((D, 10))]),
        (CausalAttention(), [g.standard_normal((2, 4, 5, D)),
                             g.standard_normal((2, 2, 5, D)),
                             g.standard_normal((2, 2, 5, D))]),
        (CausalConv1D(3), [g.standard_normal((2, 6, 4)), g.standard_normal((4, 3))]),
        (L2Norm(), [g.standard_normal((2, 2, 5, D))]),
        (GatedDeltaNet(), [g.standard_normal((2, 2, 6, 4)), g.standard_normal((2, 2, 6, 4)),
                           g.standard_normal((2, 2, 6, 4)),
                           sigmoid(g.standard_normal((2, 2, 6))),
                           sigmoid(g.standard_normal((2, 2, 6)))]),
    ]
    for m, a in cases:
        rep, _, _ = gradcheck(m, a, seed=1, name=type(m).__name__)
        assert rep.passed, f"{type(m).__name__}: {rep}"


def test_gradcheck_moe():
    g = rng(11)
    E, dm, dff, N = 4, 6, 8, 7
    args = [g.standard_normal((N, dm)), g.standard_normal((E, dm)),
            g.standard_normal((E, dff, dm)), g.standard_normal((E, dff, dm)),
            g.standard_normal((E, dm, dff)), np.zeros(E),
            g.standard_normal((dff, dm)), g.standard_normal((dff, dm)),
            g.standard_normal((dm, dff))]
    rep, _, _ = gradcheck(MoELayer(E, 2, True), args, seed=1, name="MoELayer")
    assert rep.passed, rep


# ============================================================ L2 分词

def test_bpe_roundtrip():
    corpus = ("the quick brown fox jumps over the lazy dog. " * 30 +
              "to be or not to be, that is the question. " * 30 +
              "天玑是世界上最好的东西。" * 20)
    bpe = BPE(vocab_size=400).train(corpus)
    for s in ["the quick brown fox", "to be or not to be", "Hello, world! 123",
              "天玑是世界上最好的", "a  b\tc\n d", "  lead", "trail  ", "don't stop"]:
        assert bpe.decode(bpe.encode(s)) == s


def test_bpe_determinism():
    corpus = "abc abcd abcde " * 100
    b1 = BPE(vocab_size=300).train(corpus)
    b2 = BPE(vocab_size=300).train(corpus)
    assert b1.merges == b2.merges


def test_bpe_compression():
    corpus = "the quick brown fox " * 200
    bpe = BPE(vocab_size=300).train(corpus)
    ids = bpe.encode(corpus)
    assert len(ids) < len(corpus)


# ============================================================ L3 原语

def test_rope_preserves_norm():
    g = rng(5)
    cos, sin = rope_tables(8, 16)
    x = g.standard_normal((2, 3, 6, 8))
    from tianji.l3_primitives import RoPE
    y = RoPE().forward(Ctx(), x, cos[:6][None, :, :], sin[:6][None, :, :])
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1), atol=1e-12)


def test_rope_relative_position():
    """⟨RoPE(q,i), RoPE(k,j)⟩ 只依赖 (i − j)。"""
    g = rng(6)
    D = 8
    cos, sin = rope_tables(D, 32)
    q = g.standard_normal(D)
    k = g.standard_normal(D)
    from tianji.l3_primitives import RoPE
    m = RoPE()
    def dot(i, j):
        qi = m.forward(Ctx(), q[None, None], cos[i:i + 1], sin[i:i + 1])[0, 0]
        kj = m.forward(Ctx(), k[None, None], cos[j:j + 1], sin[j:j + 1])[0, 0]
        return float(qi @ kj)
    a = [dot(0, 0), dot(1, 1), dot(5, 5)]
    b = [dot(3, 3), dot(4, 4), dot(8, 8)]
    assert np.allclose(a, b, atol=1e-12)


def test_rmsnorm_output_rms_is_one():
    g = rng(8)
    x = g.standard_normal((4, 6, 16))
    y = RMSNorm().forward(Ctx(), x, np.ones(16))
    r = np.sqrt(np.mean(y ** 2, axis=-1))
    assert np.allclose(r, 1.0, atol=1e-8)


def test_causal_attention_ignores_future():
    g = rng(9)
    q, k, v = (g.standard_normal((1, 2, 6, 8)) for _ in range(3))
    m = CausalAttention()
    o1 = m.forward(Ctx(), q, k, v)
    v2 = v.copy()
    v2[:, :, 4:, :] += 100.0          # 扰动「未来」
    o2 = m.forward(Ctx(), q, k, v2)
    assert np.allclose(o1[:, :, :4, :], o2[:, :, :4, :], atol=1e-12)


def test_rect_causal_mask_allows_all_history():
    m = causal_mask_rect(1, 5)
    assert np.all(np.isfinite(m))          # 单 query 能看到全部 5 个 key
    m2 = causal_mask_rect(3, 3)
    assert np.array_equal(m2, causal_mask(3))


# ============================================================ L4 GDN

def test_gdn_matches_naive_definition():
    """递推实现 vs 展开的朴素定义式，逐位一致。"""
    g = rng(12)
    B, H, T, D = 2, 2, 7, 4
    q, k, v = (g.standard_normal((B, H, T, D)) for _ in range(3))
    beta = sigmoid(g.standard_normal((B, H, T)))
    alpha = sigmoid(g.standard_normal((B, H, T)))
    kh = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    o = GatedDeltaNet().forward(Ctx(), q, k, v, beta, alpha)
    S = np.zeros((B, H, D, D))
    for t in range(T):
        kt, vt = kh[:, :, t, :], v[:, :, t, :]
        bt, at = beta[:, :, t][..., None, None], alpha[:, :, t][..., None, None]
        P = np.einsum("bhvk,bhk->bhv", S, kt)
        S = at * (S - bt * P[..., :, None] * kt[..., None, :]) \
            + bt * vt[..., :, None] * kt[..., None, :]
        assert np.allclose(o[:, :, t, :], np.einsum("bhvk,bhk->bhv", S, q[:, :, t, :]),
                           atol=1e-12)


def test_gdn_parallel_equals_recurrent():
    """Gated DeltaNet 的并行前缀扫描形式 vs 递推形式，逐位一致（门禁 G11 的单元级复检）。"""
    g = rng(120)
    B, H, T, D = 2, 3, 11, 6
    q, k, v = (g.standard_normal((B, H, T, D)) for _ in range(3))
    beta = g.random((B, H, T))
    alpha = g.random((B, H, T))
    rec = GatedDeltaNet().forward(Ctx(), q, k, v, beta, alpha)
    par = gdn_parallel(q, k, v, beta, alpha)
    assert np.allclose(rec, par, atol=1e-9)


def test_gdn_chunkwise_equals_recurrent():
    """chunkwise（分块并行）形式 vs 递推形式逐位一致（门禁 G12 的单元级复检）。

    T=11 故意不被 chunk 整除，覆盖尾块长度不足的路径 —— 只在整除时成立的实现会在这里暴露。
    """
    g = rng(121)
    B, H, T, D = 2, 3, 11, 6
    q, k, v = (g.standard_normal((B, H, T, D)) for _ in range(3))
    beta, alpha = g.random((B, H, T)), g.random((B, H, T))
    rec = GatedDeltaNet().forward(Ctx(), q, k, v, beta, alpha)
    for chunk in (1, 2, 4, 5, 16):
        assert np.allclose(rec, gdn_chunkwise(q, k, v, beta, alpha, chunk=chunk),
                           atol=1e-9), f"chunk={chunk} 不等价"


def test_gdn_chunkwise_backward_equals_materialized():
    """chunkwise 重计算反向 vs 物化全量状态反向逐位一致（门禁 G13 的单元级复检）。

    两条不同的资源/算法路径（物化 S_all vs 只存 O(T/C) 块入口状态）必须给出同一个
    解析梯度；T=23 故意不被 chunk 整除，覆盖尾块长度不足的路径。
    """
    g = rng(122)
    B, H, T, D = 2, 3, 23, 6
    q, k, v = (g.standard_normal((B, H, T, D)) for _ in range(3))
    beta, alpha = g.random((B, H, T)), g.random((B, H, T))
    gout = g.standard_normal((B, H, T, D))
    ctx = Ctx()
    GatedDeltaNet().forward(ctx, q, k, v, beta, alpha)
    dmat = GatedDeltaNet().backward(ctx, gout)
    for chunk in (1, 2, 3, 4, 8, 16, 23):
        dcw = gdn_chunkwise_backward(q, k, v, beta, alpha, gout, chunk=chunk)
        for a, b in zip(dmat, dcw):
            assert np.allclose(a, b, atol=1e-9), f"chunk={chunk} 反向不等价"


def test_gdn_chunkwise_backward_gradcheck():
    """用完全独立的中心差分（gradcheck）裁判 chunkwise 反向 —— 确保不是「两条手写反向互抄」。"""
    g = rng(123)
    B, H, T, D = 2, 2, 9, 5
    q, k, v = (g.standard_normal((B, H, T, D)) for _ in range(3))
    beta = g.random((B, H, T))
    alpha = g.random((B, H, T))
    rep, _, _ = gradcheck(GatedDeltaNetChunkwise(chunk=4), [q, k, v, beta, alpha],
                          seed=3, tol=1e-6, name="GDNChunkwise")
    assert rep.passed, f"gradcheck 失败: {rep}"


def test_gdn_state_bounded():
    g = rng(13)
    B, H, T, D = 1, 2, 200, 6
    k = g.standard_normal((B, H, T, D))
    v = g.standard_normal((B, H, T, D))
    q = g.standard_normal((B, H, T, D))
    beta = np.full((B, H, T), 0.5)
    alpha = np.full((B, H, T), 0.9)
    ctx = Ctx()
    GatedDeltaNet().forward(ctx, q, k, v, beta, alpha)
    n = np.linalg.norm(ctx["S_all"], axis=(3, 4))
    assert np.all(np.isfinite(n)) and n.max() < 1e6


def test_gdn_causality():
    g = rng(14)
    B, H, T, D = 1, 2, 8, 4
    q, k, v = (g.standard_normal((B, H, T, D)) for _ in range(3))
    beta = np.full((B, H, T), 0.5)
    alpha = np.full((B, H, T), 0.9)
    o1 = GatedDeltaNet().forward(Ctx(), q, k, v, beta, alpha)
    v2 = v.copy()
    v2[:, :, 5:, :] += 50.0
    o2 = GatedDeltaNet().forward(Ctx(), q, k, v2, beta, alpha)
    assert np.allclose(o1[:, :, :5, :], o2[:, :, :5, :], atol=1e-12)


# ============================================================ L5 MoE

def test_moe_gate_normalized():
    g = rng(15)
    E, dm, dff, N = 4, 6, 8, 9
    args = [g.standard_normal((N, dm)), g.standard_normal((E, dm)),
            g.standard_normal((E, dff, dm)), g.standard_normal((E, dff, dm)),
            g.standard_normal((E, dm, dff)), np.zeros(E), None, None, None]
    c = Ctx()
    MoELayer(E, 2, False).forward(c, *args)
    assert np.allclose(c["g"].sum(axis=-1), 1.0, atol=1e-12)


def test_moe_topk_sparsity():
    g = rng(16)
    E, dm, dff, N = 8, 6, 8, 12
    args = [g.standard_normal((N, dm)), g.standard_normal((E, dm)),
            g.standard_normal((E, dff, dm)), g.standard_normal((E, dff, dm)),
            g.standard_normal((E, dm, dff)), np.zeros(E), None, None, None]
    c = Ctx()
    MoELayer(E, 2, False).forward(c, *args)
    assert np.array_equal(np.count_nonzero(c["mask"], axis=-1), np.full(N, 2))


def test_load_balancer_pushes_overloaded_down():
    lb = LoadBalancer(8, gamma=1e-3)
    counts = np.array([40.0] + [2.0] * 7)
    for _ in range(50):
        lb.update(counts)
    assert lb.bias[0] < lb.bias[1]


def test_moe_no_token_dropped():
    g = rng(17)
    E, dm, dff, N = 4, 6, 8, 10
    args = [g.standard_normal((N, dm)), g.standard_normal((E, dm)),
            g.standard_normal((E, dff, dm)), g.standard_normal((E, dff, dm)),
            g.standard_normal((E, dm, dff)), np.zeros(E), None, None, None]
    c = Ctx()
    out = MoELayer(E, 2, False).forward(c, *args)
    assert np.all(np.isfinite(out)) and np.any(out != 0)


def test_moe_gate_normalized_under_extreme_inputs():
    """G17 回归：极端输入（大激活×大 router 权重×大 bias）下门控行和仍恒为 1。

    曾抓出的真实缺陷：Z = Σ + 1e-12 的加法偏置在选中亲和度极小
    （σ 低至 8.8e-27）时主导归一化，门控行和塌陷到 0（偏差 1.0）。
    修复为无偏置归一化（Z=Σ，仅 Σ==0 时兜底）。
    """
    g = rng(91)
    for scale in (0.01, 1.0, 100.0):
        for rscale in (0.001, 1.0, 50.0):
            E, dm, dff, N = 4, 6, 8, 16
            x = g.standard_normal((N, dm)) * scale
            router = g.standard_normal((E, dm)) * rscale
            Wg = g.standard_normal((E, dff, dm)) * 0.3
            Wu = g.standard_normal((E, dff, dm)) * 0.3
            Wd = g.standard_normal((E, dm, dff)) * 0.3
            bias = g.standard_normal(E) * 1000.0        # bias 主导选择
            c = Ctx()
            MoELayer(E, 2, False).forward(c, x, router, Wg, Wu, Wd, bias)
            rs = c["g"].sum(axis=-1)
            assert np.allclose(rs, 1.0, atol=1e-12), \
                f"scale={scale} rscale={rscale} 行和偏差 {np.abs(rs - 1).max():.2e}"
            assert np.array_equal(np.count_nonzero(c["mask"], axis=-1),
                                  np.full(N, 2))


def test_moe_expert_flops_equals_N_times_K():
    """G17：专家计算总量恒 == N·K（FLOPs 比 K/E 的数值体现，无隐性稠密计算）。"""
    g = rng(92)
    N, E, K, dm, dff = 60, 8, 3, 5, 7
    args = [g.standard_normal((N, dm)), g.standard_normal((E, dm)),
            g.standard_normal((E, dff, dm)), g.standard_normal((E, dff, dm)),
            g.standard_normal((E, dm, dff)), np.zeros(E), None, None, None]
    c = Ctx()
    MoELayer(E, K, False).forward(c, *args)
    assert sum(len(t) for t in c["token_ids"]) == N * K


def test_load_balancer_clip_and_empty_maxcv():
    """G17：偏置 clip 边界 + maxcv 空历史必须显式为 inf（不假装均衡）。"""
    lb = LoadBalancer(4, gamma=1e-2, bias_cap=0.2)
    for _ in range(50):
        lb.update(np.array([500.0, 0.0, 0.0, 0.0]))
    assert np.all(np.abs(lb.bias) <= 0.2 + 1e-12)
    lb.reset()
    assert lb.bias.tolist() == [0.0] * 4 and lb.history == []
    assert LoadBalancer(4).maxcv() == float("inf")


# ============================================================ L6 优化器

def test_muon2_quality_at_least_muon_90pct():
    g = rng(18)
    G = g.standard_normal((24, 16))
    G = G @ G.T @ G
    q5 = orthogonalization_quality(G, newton_schulz(G, steps=5))
    q3 = orthogonalization_quality(G, newton_schulz(G / (np.sqrt(np.mean(G ** 2)) + 1e-8),
                                                    steps=3))
    assert q3 >= 0.9 * q5


def test_muon_handles_3d_params():
    """MoE 专家权重是 3D，优化器必须能处理。"""
    P = {"W": rng(19).standard_normal((4, 8, 6))}
    Gd = {"W": rng(20).standard_normal((4, 8, 6))}
    before = P["W"].copy()
    Muon2(ns_steps=3).step(P, Gd)
    assert not np.allclose(before, P["W"])


def test_cosine_schedule_monotone_and_endpoints():
    vals = [cosine_schedule(s, 100, 10, 1.0, 0.0) for s in range(100)]
    assert abs(vals[0] - 0.1) < 1e-12
    # 端点必须精确落在 lr_min：调度器对 step >= total 有专门分支
    assert cosine_schedule(100, 100, 10, 1.0, 0.0) == 0.0
    tail = vals[10:]
    assert all(tail[i] >= tail[i + 1] - 1e-12 for i in range(len(tail) - 1))


def test_wsd_schedule_has_stable_phase():
    vals = [wsd_schedule(s, 100, 10, 1.0) for s in range(100)]
    assert abs(vals[50] - 1.0) < 1e-12      # 稳定期保持 lr_max
    assert wsd_schedule(100, 100, 10, 1.0, 0.0) < 1e-12


# ============================================================ L7 训练

def test_model_gradcheck():
    cfg = TianjiConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                       n_kv_heads=2, d_latent_q=16, d_latent_kv=12, d_rope=4,
                       d_ff=32, n_experts=1, mtp_depth=0, hybrid_ratio=1, max_len=32)
    P = init_params(cfg, seed=0)
    tok = rng(1).integers(0, 32, (2, 6))
    tgt = rng(2).integers(0, 32, (2, 6))

    def loss_of(Pd):
        return TianjiModel(cfg, Pd).forward(tok, tgt)["loss"]

    m = TianjiModel(cfg, P)
    m.forward(tok, tgt)
    G = m.backward()
    gscale = max(float(np.abs(v).max()) for v in G.values())
    atol = 1e-7 * gscale
    eps, tol = 1e-5, 1e-5
    worst = 0.0
    rs = rng(3)
    for nm in sorted(P):
        if nm not in G:
            continue
        arr = P[nm]
        ga = G[nm].reshape(-1)
        fl = arr.reshape(-1)
        for j in rs.choice(arr.size, size=min(3, arr.size), replace=False):
            o = fl[j]
            fl[j] = o + eps; a = loss_of(P)
            fl[j] = o - eps; b = loss_of(P)
            fl[j] = o
            num = (a - b) / (2 * eps)
            worst = max(worst, abs(ga[j] - num) / (atol + tol * abs(num)))
    assert worst <= 1.0, f"模型梯度检验 worst={worst}"


def test_training_reduces_loss():
    from tianji.l7_train import Trainer
    from tianji.l6_optimizer import HybridOptimizer
    cfg = TianjiConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                       d_latent_q=16, d_latent_kv=12, d_rope=4, d_ff=32,
                       n_experts=1, mtp_depth=0, max_len=32)
    m = TianjiModel(cfg, init_params(cfg, seed=0))
    tok = rng(4).integers(0, 32, 400)
    ld = DataLoader(tok, seq_len=10, batch_size=2, seed=0)
    tr = Trainer(m, HybridOptimizer(), ld, total_steps=40, warmup=5, lr_max=0.05)
    log = tr.run(40, verbose=False)
    assert np.mean(log.losses[-3:]) < np.mean(log.losses[:3])


def test_determinism_same_seed():
    cfg = TianjiConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                       d_latent_q=16, d_latent_kv=12, d_rope=4, d_ff=32,
                       n_experts=1, mtp_depth=0, max_len=32)
    from tianji.l7_train import Trainer
    from tianji.l6_optimizer import HybridOptimizer

    def run(s):
        mm = TianjiModel(cfg, init_params(cfg, seed=s))
        ld = DataLoader(rng(s).integers(0, 32, 200), 8, 2, seed=s)
        Trainer(mm, HybridOptimizer(), ld, total_steps=8, warmup=2).run(8, verbose=False)
        return float(np.sum([np.sum(np.asarray(v)) for v in mm.P.values()]))

    assert run(0) == run(0)


# ============================================================ L8 推理

def test_quant_error_bounded_and_monotone():
    W = rng(21).standard_normal((32, 32))
    W8, s8 = quantize_per_channel(W, 8)
    W4, s4, g = quantize_groupwise(W, 4, group=32)
    e8 = np.linalg.norm(W - dequantize(W8, s8)) / np.linalg.norm(W)
    e4 = np.linalg.norm(W - dequantize(W4, s4, g)) / np.linalg.norm(W)
    assert e8 < 1e-2 and e4 < 2e-1 and e8 <= e4


def test_kv_cache_equivalence():
    cfg = TianjiConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                       d_latent_q=16, d_latent_kv=12, d_rope=4, d_ff=32,
                       n_experts=1, mtp_depth=0, hybrid_ratio=0, max_len=32)
    m = TianjiModel(cfg, init_params(cfg, seed=0))
    seq = rng(9).integers(0, 32, (1, 8))
    full = m.forward(seq)["logits"]
    cache, steps = {}, []
    for t in range(seq.shape[1]):
        steps.append(m.forward(seq[:, t:t + 1], cache=cache, pos=t)["logits"][0, -1])
    assert np.max(np.abs(full - np.stack(steps)[None])) < 1e-10


# ============================================================ L9 对齐

def test_dpo_loss_at_zero_preference():
    from tianji.l9_align import dpo_terms
    loss, coef = dpo_terms(1.0, 1.0, 1.0, 1.0, beta=0.1)
    assert abs(loss - np.log(2.0)) < 1e-9   # sigmoid 内有 1e-12 数值保护项
    assert coef < 0            # 仍应推动 chosen 上升


def test_ipo_optimum_is_finite():
    from tianji.l9_align import ipo_terms
    beta, tau = 0.1, 0.1
    target = tau / (2 * beta)
    loss_at_opt, _ = ipo_terms(target, 0.0, 0.0, 0.0, beta=beta, tau=tau)
    loss_far, _ = ipo_terms(1000.0, 0.0, 0.0, 0.0, beta=beta, tau=tau)
    assert loss_at_opt < loss_far


def test_simpo_length_normalization():
    from tianji.l9_align import seq_logprob, mean_logprob
    logits = rng(22).standard_normal((1, 8, 16))
    tgt = rng(23).integers(0, 16, (1, 8))
    a = mean_logprob(logits, tgt)
    tgt2 = np.concatenate([tgt, tgt], axis=1)
    logits2 = np.concatenate([logits, logits], axis=1)
    b = mean_logprob(logits2, tgt2)
    assert abs(a - b) < 1e-12      # 复制一倍，平均 logprob 不变


def test_reference_model_frozen():
    from tianji.l9_align import freeze_copy, Aligner
    cfg = TianjiConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                       d_latent_q=16, d_latent_kv=12, d_rope=4, d_ff=32,
                       n_experts=1, mtp_depth=0, max_len=32)
    P = init_params(cfg, seed=0)
    ref = freeze_copy(P)
    al = Aligner(TianjiModel(cfg, P), ref, method="dpo")
    x = rng(1).integers(0, 32, (1, 5))
    y = rng(2).integers(0, 32, (1, 5))
    _, G = al.loss_and_grads(x, y, x, y)
    assert np.array_equal(ref["tok_emb"], freeze_copy(ref)["tok_emb"])
    assert set(G.keys()) <= set(P.keys())


def test_dpo_end_to_end_gradcheck_and_direction():
    """G16 核心：DPO 全链路（chosen+rejected 双向反传+梯度合并）的参数梯度
    用中心差分独立裁判；一步更新后 chosen logprob 上升、rejected 下降。"""
    from tianji.l9_align import Aligner, freeze_copy, seq_logprob
    cfg = TianjiConfig(vocab_size=24, d_model=16, n_layers=2, n_heads=2,
                       n_kv_heads=1, d_latent_q=8, d_latent_kv=6, d_rope=4,
                       d_ff=16, d_ff_expert=12, n_experts=2, moe_top_k=1,
                       max_len=32, mtp_depth=0)
    g = rng(77)
    P = init_params(cfg, seed=3)
    pol = TianjiModel(cfg, P)
    al = Aligner(pol, freeze_copy(init_params(cfg, seed=9)), method="dpo", beta=0.1)
    xc, yc = g.integers(0, 24, (1, 6)), g.integers(0, 24, (1, 6))
    xr, yr = g.integers(0, 24, (1, 5)), g.integers(0, 24, (1, 5))
    _, G = al.loss_and_grads(xc, yc, xr, yr)

    eps = 1e-5
    for k in ("tok_emb", "final_norm_w"):
        arr = P[k]
        idx = tuple(0 for _ in arr.shape)
        o = arr[idx]
        arr[idx] = o + eps; fp = al.loss_and_grads(xc, yc, xr, yr)[0]
        arr[idx] = o - eps; fm = al.loss_and_grads(xc, yc, xr, yr)[0]
        arr[idx] = o
        num = (fp - fm) / (2 * eps)
        assert abs(num - G[k][idx]) < 1e-6, f"{k} 端到端梯度失配: {num} vs {G[k][idx]}"

    lp_c0 = seq_logprob(pol.forward(xc, yc)["logits"], yc)
    lp_r0 = seq_logprob(pol.forward(xr, yr)["logits"], yr)
    for k in P:
        if k in G:
            P[k] -= 0.5 * G[k]
    lp_c1 = seq_logprob(pol.forward(xc, yc)["logits"], yc)
    lp_r1 = seq_logprob(pol.forward(xr, yr)["logits"], yr)
    assert lp_c1 > lp_c0, "chosen logprob 应上升"
    assert lp_r1 < lp_r0, "rejected logprob 应下降"


# ============================================================ L10 微指令

def test_microop_count():
    assert len(FAMILIES) == 32
    assert len(MICROOPS) == 1024


def test_microops_all_runnable():
    r = micro_all(seed=0)
    assert r["failed"] == [], r["failed"][:5]
    assert r["ok"] == 1024


# ============================================================ L8 投机解码

def test_spec_decode_residual_identity():
    """单步拒绝采样恒等：输出分布逐点 == target，与 draft 好坏无关（64 组对抗分布）。"""
    g = rng(spawn_seed(7, "spec_id"))
    for trial in range(64):
        V = int(g.integers(2, 25))
        lp = g.standard_normal(V) * g.uniform(0.5, 3.0)
        lq = g.standard_normal(V) * g.uniform(0.5, 3.0)
        if trial % 4 == 1:
            lq = lp.copy()                                   # draft == target（恒接受）
        elif trial % 4 == 2:
            lp[int(g.integers(0, V))] = -40.0                # target 含近零质量
        elif trial % 4 == 3:
            lq = g.standard_normal(V) + 3.0                  # draft 高度集中
        p, q = softmax(lp), softmax(lq)
        acc = q * np.minimum(1.0, p / np.maximum(q, 1e-300))
        rej = 1.0 - acc.sum()
        r = np.maximum(p - q, 0.0)
        rs = r.sum()
        out = acc + (rej * r / rs if rs > 1e-12 else 0.0)
        assert np.max(np.abs(out - p)) < 1e-12, f"trial={trial} 恒等被破坏"


def test_spec_decode_statistical_lossless():
    """端到端统计无损：mock 双分布模型 6000 token，经验边际 TV<0.05；接受率 ∈ (0,1)。"""
    from types import SimpleNamespace
    V = 12
    pb = np.arange(V, dtype=np.float64) * 0.7 - 3.0
    pq = np.concatenate([np.full(3, 2.0), np.full(V - 3, -2.0)])   # draft 偏斜
    p = softmax(pb)

    class _Mock:
        cfg = SimpleNamespace(mtp_depth=1)

        def forward(self, x, **kw):
            L = x.shape[1]
            return {"logits": np.broadcast_to(pb, (1, L, V)).copy(),
                    "mtp_logits": np.broadcast_to(pq, (1, L, V)).copy(), "aux": {}}

    ids, st = speculative_decode(_Mock(), np.asarray([[0]], dtype=np.int64),
                                 6000, gamma=1, seed=11)
    gen = np.asarray(ids[1:], dtype=np.int64)
    assert gen.size == 6000
    tv = 0.5 * float(np.abs(np.bincount(gen, minlength=V) / gen.size - p).sum())
    assert tv < 0.05, f"经验边际偏离 target：TV={tv:.4f}"
    assert 0.0 < st.accept_rate < 1.0, f"接受率无效：{st.accept_rate}"
    assert st.accepted >= 1


def test_spec_decode_gamma_fingerprint_and_algebra():
    """γ>1 并行验证：位置指纹逐位一致（钉死 off-by-one 与状态回滚）+ 代数条件。

    target/draft 同指纹（q=p）→ 全接受 + bonus，产出完全确定：
    产出序列必须恰为指纹 1..9；每轮主干 forward 恰 1 次；tokens/forward = 3 > 1。
    """
    from types import SimpleNamespace
    from tianji.l8_infer import TinyDraftGDN
    V = 12

    def finger(t):
        z = np.zeros(V)
        z[t % V] = 40.0
        return z

    class _T:
        cfg = SimpleNamespace(mtp_depth=0)
        calls = 0

        def forward(self, x, **kw):
            self.calls += 1
            L = x.shape[1]
            return {"logits": np.stack([finger(t + 1) for t in range(L)])[None],
                    "aux": {}}

    class _D:
        # S = 已喂 token 栈：dm.S = 快照 赋值即回滚（与真实 GDN 状态语义一致）
        def __init__(self):
            self.S = []
            self.last_logits = finger(0)

        def warmup(self, tokens):
            self.S = list(np.asarray(tokens).reshape(-1))
            self.last_logits = finger(len(self.S))

        def step(self, token):
            self.S = [*self.S, int(token)]
            return finger(len(self.S))

    m = _T()
    ids, st = speculative_decode(m, np.asarray([[0]], dtype=np.int64),
                                 9, gamma=3, seed=5, draft_model=_D())
    assert list(ids[1:]) == list(range(1, 10)), f"指纹序列破坏：{ids[1:]}"
    assert m.calls == st.steps == 3, f"forward 次数错误：{m.calls} vs steps={st.steps}"
    assert st.produced == 9 and st.speedup > 1.0, \
        f"代数条件失败：produced={st.produced} steps={st.steps}"


def test_spec_decode_gamma_statistical_lossless():
    """γ>1 混合接受/拒绝路径统计无损 + 真实 TinyDraftGDN（已门禁 gdn_step）冒烟。"""
    from types import SimpleNamespace
    from tianji.l8_infer import TinyDraftGDN
    V = 12
    pb = np.arange(V, dtype=np.float64) * 0.7 - 3.0
    pq = np.concatenate([np.full(3, 2.0), np.full(V - 3, -2.0)])
    p = softmax(pb)

    class _T:
        cfg = SimpleNamespace(mtp_depth=0)

        def forward(self, x, **kw):
            L = x.shape[1]
            return {"logits": np.broadcast_to(pb, (1, L, V)).copy(), "aux": {}}

    class _D:
        def __init__(self):
            self.S = np.zeros((1, 1, 1, 1))
            self.last_logits = pq.copy()

        def warmup(self, tokens):
            pass

        def step(self, token):
            return pq

    ids, st = speculative_decode(_T(), np.asarray([[0]], dtype=np.int64),
                                 4000, gamma=3, seed=13, draft_model=_D())
    gen = np.asarray(ids[1:], dtype=np.int64)
    assert gen.size == 4000
    tv = 0.5 * float(np.abs(np.bincount(gen, minlength=V) / gen.size - p).sum())
    assert tv < 0.05, f"γ>1 经验边际偏离 target：TV={tv:.4f}"
    assert 0.0 < st.accept_rate < 1.0
    # ---- 真实 TinyDraftGDN（gdn_step 递推）冒烟：结构正确、输出有限、接口完整
    d = TinyDraftGDN(V, d=16, seed=3)
    d.warmup([1, 2, 3])
    lg = d.step(4)
    assert lg.shape == (V,) and np.all(np.isfinite(lg))
    ids2, st2 = speculative_decode(_T(), np.asarray([[1, 2]], dtype=np.int64),
                                   20, gamma=2, seed=7, draft_model=d)
    assert len(ids2) == 22 and st2.produced == 20
    assert st2.steps >= 1 and np.all(np.isfinite(np.asarray(ids2)))


# ============================================================ 门禁

def test_all_gates():
    """完整门禁（较慢，默认跑轻量档）。"""
    rep = run_all_gates(seed=0, steps=20)
    assert rep.all_green, rep.table()
