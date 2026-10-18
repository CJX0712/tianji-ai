"""基准 —— 所有数字由本机实跑产出，不填预想值。

运行： python bench/run_bench.py [--steps N] [--seed S] [--quick]

对比维度：
    B1 架构    Hybrid 3:1 (GDN+MLA) / 全注意力 / 全 GDN，等参数量
    B2 布局    interleave（Qwen3-Next） vs edge（Mamba-3）
    B3 优化器  AdamW / Muon / Muon2
    B4 量化    FP64 / INT8 / INT4 的困惑度退化
    B5 缓存    混合架构 vs 全注意力的 KV cache 字节数（随上下文长度）
    B6 投机解码 接受率与加速比

诚实声明：
    - 语料是**确定性合成**的 2 阶马尔可夫，不是真实文本。
      选择合成语料的理由：离线可复现（无需下载），且有明确的熵下界，
      因此「降到多少才算学会」是可判定的。
      因此本基准的绝对数字**不可与任何真实语料上的结果横向比较**，
      它只用于**同一实验内不同配置之间的相对比较**。
    - 模型规模在百万参数以下、训练步数在数百量级。
      这是 CPU + NumPy 后端的现实约束，不是「大模型」基准。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tianji.model import TianjiConfig, TianjiModel, init_params          # noqa: E402
from tianji.l0_numerics import spawn_seed, rng                            # noqa: E402
from tianji.l6_optimizer import HybridOptimizer, AdamW, Muon, Muon2       # noqa: E402
from tianji.l7_train import DataLoader, Trainer, evaluate                 # noqa: E402
from tianji.l8_infer import (quantize_per_channel, quantize_groupwise,    # noqa: E402
                             dequantize, speculative_decode, generate,
                             TinyDraftGDN)


def synth_tokens(n: int, vocab: int, seed: int = 0) -> np.ndarray:
    g = rng(spawn_seed(seed, "synth"))
    trans = g.dirichlet(np.ones(vocab) * 0.25, size=(vocab, vocab))
    out = np.zeros(n, dtype=np.int64)
    a, b = 0, 1
    for i in range(n):
        c = g.choice(vocab, p=trans[a, b])
        out[i] = c
        a, b = b, c
    return out


def base_cfg(**kw):
    c = TianjiConfig(vocab_size=64, d_model=64, n_layers=6, n_heads=4,
                     n_kv_heads=2, d_latent_q=32, d_latent_kv=24, d_rope=8,
                     d_ff=128, d_ff_expert=64, n_experts=4, moe_top_k=2,
                     max_len=256, mtp_depth=1)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def run_train(cfg, tok, val_tok, steps, seed, opt_kind="muon2", lr=0.03,
              seq=32, batch=8):
    P = init_params(cfg, seed=seed)
    m = TianjiModel(cfg, P)
    if opt_kind == "muon2":
        opt = HybridOptimizer(muon2=Muon2(ns_steps=3), adamw=AdamW(lr=lr))
    elif opt_kind == "muon":
        opt = HybridOptimizer(muon2=Muon(ns_steps=5), adamw=AdamW(lr=lr))
    else:
        # 纯 AdamW：把所有参数都交给 AdamW（关闭 Muon 路由）
        class _AO(HybridOptimizer):
            def is_muon(self, name, p):
                return False
        opt = _AO(muon2=Muon2(ns_steps=3), adamw=AdamW(lr=lr))
    ld = DataLoader(tok, seq_len=seq, batch_size=batch, seed=seed)
    vld = DataLoader(val_tok, seq_len=seq, batch_size=batch, seed=seed + 999)
    tr = Trainer(m, opt, ld, total_steps=steps, warmup=max(steps // 10, 5),
                 lr_max=lr, val_loader=vld, eval_every=max(steps // 4, 1))
    t0 = time.time()
    log = tr.run(steps, verbose=False)
    dt = time.time() - t0
    return {
        "params": m.param_count(),
        "layers": "".join("A" if l == "attn" else "L" for l in m.layers),
        "final_loss": float(np.mean(log.losses[-5:])),
        "best_loss": float(np.min(log.losses)),
        "val_loss": float(log.val[-1][1]) if log.val else float("nan"),
        "seconds": dt,
        "tokens_per_s": steps * batch * seq / max(dt, 1e-9),
        "ms_per_step": dt * 1000 / steps,
    }


def bench_architecture(steps, seed):
    tok = synth_tokens(6000, 64, seed)
    val = synth_tokens(1500, 64, seed + 7)
    rows = []
    variants = [
        ("Hybrid 3:1 (GDN+MLA)", dict(hybrid_ratio=3)),
        ("全注意力 (MLA)", dict(hybrid_ratio=0)),
        ("全 GDN (无注意力)", dict(hybrid_ratio=10 ** 6)),
    ]
    for name, kw in variants:
        cfg = base_cfg(**kw)
        r = run_train(cfg, tok, val, steps, seed)
        r["name"] = name
        rows.append(r)
        print(f"  {name:<24} loss={r['final_loss']:.4f} val={r['val_loss']:.4f} "
              f"params={r['params']} {r['ms_per_step']:.0f}ms/step", flush=True)
    return rows


def bench_layout(steps, seed):
    tok = synth_tokens(6000, 64, seed)
    val = synth_tokens(1500, 64, seed + 7)
    rows = []
    for lay in ("interleave", "edge"):
        cfg = base_cfg(layout=lay)
        r = run_train(cfg, tok, val, steps, seed)
        r["name"] = f"layout={lay}"
        r["layout"] = lay
        rows.append(r)
        print(f"  layout={lay:<16} loss={r['final_loss']:.4f} val={r['val_loss']:.4f}", flush=True)
    return rows


def bench_optimizer(steps, seed):
    tok = synth_tokens(6000, 64, seed)
    val = synth_tokens(1500, 64, seed + 7)
    rows = []
    for kind, lr in (("adamw", 3e-3), ("muon", 0.03), ("muon2", 0.03)):
        cfg = base_cfg()
        r = run_train(cfg, tok, val, steps, seed, opt_kind=kind, lr=lr)
        r["name"] = kind
        r["lr"] = lr
        rows.append(r)
        print(f"  optimizer={kind:<8} lr={lr:<7} loss={r['final_loss']:.4f} "
              f"val={r['val_loss']:.4f}", flush=True)
    return rows


def bench_quantization(seed):
    cfg = base_cfg(n_layers=2, n_experts=1)
    P = init_params(cfg, seed=seed)
    m = TianjiModel(cfg, P)
    tok = synth_tokens(400, cfg.vocab_size, seed)
    ld = DataLoader(tok, seq_len=24, batch_size=4, seed=seed)

    def eval_model(params):
        mm = TianjiModel(cfg, params)
        return evaluate(mm, ld, n_batch=6)

    base = eval_model(P)
    rows = [{"name": "FP64", "loss": float(base), "ppl": float(np.exp(base)),
             "rel_err": 0.0}]
    for bits, tag in ((8, "INT8"), (4, "INT4")):
        Q = {}
        errs = []
        for k, v in P.items():
            if v.ndim < 2:
                Q[k] = v.copy()
                continue
            if bits == 8:
                Wq, sc = quantize_per_channel(v, 8)
                Q[k] = dequantize(Wq, sc)
            else:
                Wq, sc, g = quantize_groupwise(v, 4, group=32)
                Q[k] = dequantize(Wq, sc, g)
            errs.append(np.linalg.norm(v - Q[k]) / (np.linalg.norm(v) + 1e-12))
        loss = eval_model(Q)
        rows.append({"name": tag, "loss": float(loss), "ppl": float(np.exp(loss)),
                     "rel_err": float(np.mean(errs))})
        print(f"  {tag}: loss {base:.4f} -> {loss:.4f} (ppl {np.exp(base):.2f} -> "
              f"{np.exp(loss):.2f}), 权重量化相对误差 {np.mean(errs):.3e}", flush=True)
    return rows


def bench_cache(seed):
    """KV/状态缓存随上下文增长的字节数。

    混合架构里只有全注意力层需要 KV cache；GDN 层的状态是 O(d²) 常量。
    """
    rows = []
    for T in (128, 512, 2048, 8192):
        cfg_h = base_cfg(hybrid_ratio=3)
        cfg_a = base_cfg(hybrid_ratio=0)
        mh = TianjiModel(cfg_h, init_params(cfg_h, seed=seed))
        ma = TianjiModel(cfg_a, init_params(cfg_a, seed=seed))

        def cache_bytes(m):
            n_attn = sum(1 for l in m.layers if l == "attn")
            H, dh, dr = m.cfg.n_heads, m.cfg.d_head, m.cfg.d_rope
            per_token = n_attn * H * (dh + dr) * 8 * 2      # K 与 V，float64
            return T * per_token
        bh, ba = cache_bytes(mh), cache_bytes(ma)
        rows.append({"T": T, "hybrid_bytes": bh, "full_attn_bytes": ba,
                     "ratio": bh / max(ba, 1)})
        print(f"  T={T:<6} hybrid={bh / 1024:.1f}KB  full={ba / 1024:.1f}KB  "
              f"节省 {(1 - bh / ba) * 100:.1f}%", flush=True)
    return rows


def bench_speculative(seed, max_new=16):
    cfg = base_cfg(n_layers=2, n_experts=1, mtp_depth=1)
    P = init_params(cfg, seed=seed)
    m = TianjiModel(cfg, P)
    tok = synth_tokens(6000, 64, seed)
    # 先训练几步让 MTP 头有意义
    ld = DataLoader(tok, seq_len=24, batch_size=4, seed=seed)
    opt = HybridOptimizer(muon2=Muon2(ns_steps=3), adamw=AdamW(lr=0.03))
    Trainer(m, opt, ld, total_steps=25, warmup=5, lr_max=0.03).run(25, verbose=False)
    prompt = synth_tokens(6, cfg.vocab_size, seed + 3)
    ids, st = speculative_decode(m, prompt, max_new=max_new, gamma=1, seed=seed)
    print(f"  MTP 搭车(γ=1): drafted={st.drafted} accepted={st.accepted} "
          f"接受率={st.accept_rate:.3f} tokens/forward={st.speedup:.3f}", flush=True)
    # 独立轻量 draft + γ=3 一次 batch 并行验证：每轮 1 次主干 forward
    dm = TinyDraftGDN(cfg.vocab_size, d=32, seed=seed)
    ids2, st2 = speculative_decode(m, prompt, max_new=max_new, gamma=3,
                                   seed=seed, draft_model=dm)
    print(f"  独立draft(γ=3): drafted={st2.drafted} accepted={st2.accepted} "
          f"接受率={st2.accept_rate:.3f} tokens/forward={st2.speedup:.3f} "
          f"(随机 draft 接受率低属预期——无损性由 G15 机检；"
          f"训练 draft 后接受率↑→加速趋近 1+α)", flush=True)
    return {"drafted": st.drafted, "accepted": st.accepted,
            "accept_rate": st.accept_rate, "tokens_per_verify": st.speedup,
            "gamma3_tokens_per_forward": st2.speedup}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps = 40 if a.quick else a.steps

    print("=" * 68)
    print("B1 架构对比")
    print("=" * 68)
    b1 = bench_architecture(steps, a.seed)
    print()
    print("=" * 68)
    print("B2 布局对比")
    print("=" * 68)
    b2 = bench_layout(steps, a.seed)
    print()
    print("=" * 68)
    print("B3 优化器对比")
    print("=" * 68)
    b3 = bench_optimizer(steps, a.seed)
    print()
    print("=" * 68)
    print("B4 量化")
    print("=" * 68)
    b4 = bench_quantization(a.seed)
    print()
    print("=" * 68)
    print("B5 KV cache")
    print("=" * 68)
    b5 = bench_cache(a.seed)
    print()
    print("=" * 68)
    print("B6 投机解码")
    print("=" * 68)
    b6 = bench_speculative(a.seed)
    print()

    out = {"seed": a.seed, "steps": steps,
           "architecture": b1, "layout": b2, "optimizer": b3,
           "quantization": b4, "cache": b5, "speculative": b6,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "results", "bench.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {path}")


if __name__ == "__main__":
    main()
