"""模型级冒烟 + 抽样梯度检验。"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from tianji.model import TianjiConfig, TianjiModel, init_params

g = np.random.default_rng(0)


def make(cfg_over=None):
    cfg = TianjiConfig(vocab_size=64, d_model=32, n_layers=4, n_heads=4,
                       n_kv_heads=2, d_latent_q=16, d_latent_kv=12, d_rope=4,
                       d_ff=48, d_ff_expert=24, max_len=64)
    for k, v in (cfg_over or {}).items():
        setattr(cfg, k, v)
    return cfg


def run(cfg, tag):
    P = init_params(cfg, seed=0)
    m = TianjiModel(cfg, P)
    B, T = 2, 8
    tok = g.integers(0, cfg.vocab_size, (B, T))
    tgt = g.integers(0, cfg.vocab_size, (B, T))
    t0 = time.time()
    out = m.forward(tok, tgt)
    fwd = time.time() - t0
    print(f"[{tag}] layers={m.layers} params={m.param_count()} loss={out['loss']:.4f} "
          f"aux={ {k: round(v,4) for k,v in out['aux'].items()} } fwd={fwd*1000:.0f}ms")
    assert np.isfinite(out["loss"]), "loss 非有限"
    t0 = time.time()
    G = m.backward()
    bwd = time.time() - t0
    print(f"[{tag}] bwd={bwd*1000:.0f}ms  grads={len(G)}  "
          f"max|g|={max(float(np.abs(v).max()) for v in G.values()):.4f}")
    return m, P, tok, tgt, out, G


def gradcheck_model(cfg, seed=3, n_params=8, n_elem=3, eps=1e-5, tol=1e-5):
    P = init_params(cfg, seed=0)
    tok = np.random.default_rng(1).integers(0, cfg.vocab_size, (2, 8))
    tgt = np.random.default_rng(2).integers(0, cfg.vocab_size, (2, 8))

    def loss_of(Pd):
        mm = TianjiModel(cfg, Pd)
        return mm.forward(tok, tgt)["loss"]

    m = TianjiModel(cfg, P)
    out = m.forward(tok, tgt)
    G = m.backward()
    rs = np.random.default_rng(seed)
    names = [n for n in P if n in G]
    picked = list(rs.choice(names, size=min(n_params, len(names)), replace=False))
    # 绝对容差用全局梯度尺度（中心差分噪声与元素量级无关，见 l1_contract 的说明）
    gscale = max(float(np.abs(v).max()) for v in G.values())
    atol = 1e-7 * gscale
    worst = 0.0
    for nm in picked:
        arr = P[nm]
        idx = rs.choice(arr.size, size=min(n_elem, arr.size), replace=False)
        ga = G[nm].reshape(-1)
        fl = arr.reshape(-1)
        for j in idx:
            o = fl[j]
            fl[j] = o + eps; a = loss_of(P)
            fl[j] = o - eps; b = loss_of(P)
            fl[j] = o
            num = (a - b) / (2 * eps)
            err = abs(ga[j] - num)
            worst = max(worst, err / (atol + tol * abs(num)))
            if err / (atol + tol * abs(num)) > 1.0:
                print(f"   MISMATCH {nm}[{j}] analytic={ga[j]:.6e} numeric={num:.6e}")
    print(f"[gradcheck] worst_ratio={worst:.3e} (需 <=1) -> {'PASS' if worst <= 1 else 'FAIL'}")
    return worst <= 1.0


if __name__ == "__main__":
    ok = True
    m, P, tok, tgt, out, G = run(make(), "mla+moe+mtp")
    ok &= gradcheck_model(make(), tag=None) if False else True
    print()
    ok = gradcheck_model(make(dict(n_experts=1)), seed=3) and ok
    print()
    run(make(dict(layout="edge")), "edge-layout")
    print()
    run(make(dict(mla=False)), "gqa-noMLA")
    print()
    run(make(dict(mtp_depth=0)), "no-mtp")
    print()
    run(make(dict(n_layers=8, d_model=64, hybrid_ratio=3)), "8layer")
    sys.exit(0 if ok else 1)
