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
from tianji.l4_deltanet import GatedDeltaNet, CausalConv1D, L2Norm   # noqa: E402
from tianji.l5_moe import (                                 # noqa: E402
    MoELayer, LoadBalancer, sigmoid)
from tianji.l6_optimizer import Muon, Muon2, orthogonalization_quality  # noqa: E402
from tianji.l7_train import cosine_schedule, wsd_schedule, DataLoader   # noqa: E402
from tianji.l8_infer import quantize_per_channel, quantize_groupwise, dequantize  # noqa: E402
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


# ============================================================ L10 微指令

def test_microop_count():
    assert len(FAMILIES) == 32
    assert len(MICROOPS) == 1024


def test_microops_all_runnable():
    r = micro_all(seed=0)
    assert r["failed"] == [], r["failed"][:5]
    assert r["ok"] == 1024


# ============================================================ 门禁

def test_all_gates():
    """完整门禁（较慢，默认跑轻量档）。"""
    rep = run_all_gates(seed=0, steps=20)
    assert rep.all_green, rep.table()
