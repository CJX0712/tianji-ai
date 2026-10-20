"""基准 —— 所有数字由本机实跑产出，不填预想值。

运行： python bench/run_bench.py [--steps N] [--seed S] [--quick]

对比维度：
    B0 熵下界  H（可机检，门禁 G24）—— 任何模型交叉熵 ≥ H
    B1 架构    Hybrid 3:1 (GDN+MLA) / 全注意力 / 全 GDN，等参数量
    B2 布局    interleave（Qwen3-Next） vs edge（Mamba-3）
    B3 优化器  AdamW / Muon / Muon2
    B4 量化    FP64 / INT8 / INT4 的困惑度退化
    B5 缓存    混合架构 vs 全注意力的 KV cache 字节数（随上下文长度）
    B6 投机解码 接受率与加速比
    B7 可学性对照臂  同模型规模换「规模内可学」语料（1 阶 vocab=8），
                     证明基准能测出「学会」（learned fraction 转正 ≈1），
                     不是永远学不会的钝器；B1 的三架构在同一可学任务上重跑

诚实声明：
    - 语料是**确定性合成**的 2 阶马尔可夫（:func:`tianji.markov_tokens`），不是真实文本。
      选择合成语料的理由：离线可复现（无需下载），且有明确的熵下界，
      因此「降到多少才算学会」是可判定的 —— 该下界 H 由
      :func:`tianji.markov_entropy_rate` 算出并由门禁 G24 机检（见 B0）。
      因此本基准的绝对数字**不可与任何真实语料上的结果横向比较**，
      它只用于**同一实验内不同配置之间的相对比较**。
    - train 与 val 由 :func:`tianji.markov_train_val` 从**同一转移矩阵**采样两条
      独立序列得到（同源留出）。历史上曾用 `seed` 与 `seed+7` 分别生成，而
      markov_tokens 每次调用都重抽转移矩阵 ⇒ train/val 竟是不同源，val 指标失效
      （把「能学会」误判成「学不会」）；已修复并由门禁 G26 机检同源性防回归。
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
from tianji.data import (markov_tokens as synth_tokens,                  # noqa: E402
                         markov_entropy_rate, markov_train_val,
                         markov_transitions)
from tianji.l6_optimizer import HybridOptimizer, AdamW, Muon, Muon2       # noqa: E402
from tianji.l7_train import DataLoader, Trainer, evaluate                 # noqa: E402
from tianji.l8_infer import (quantize_per_channel, quantize_groupwise,    # noqa: E402
                             dequantize, speculative_decode, generate,
                             TinyDraftGDN)


def base_cfg(**kw):
    c = TianjiConfig(vocab_size=64, d_model=64, n_layers=6, n_heads=4,
                     n_kv_heads=2, d_latent_q=32, d_latent_kv=24, d_rope=8,
                     d_ff=128, d_ff_expert=64, n_experts=4, moe_top_k=2,
                     max_len=256, mtp_depth=1)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def run_train(cfg, tok, val_tok, steps, seed, opt_kind="muon2", lr=0.03,
              seq=32, batch=8, full_val_batches=0, oracle_trans=None):
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
    r = {
        "params": m.param_count(),
        "layers": "".join("A" if l == "attn" else "L" for l in m.layers),
        "final_loss": float(np.mean(log.losses[-5:])),
        "best_loss": float(np.min(log.losses)),
        "val_loss": float(log.val[-1][1]) if log.val else float("nan"),
        "seconds": dt,
        "tokens_per_s": steps * batch * seq / max(dt, 1e-9),
        "ms_per_step": dt * 1000 / steps,
    }
    # 可选的「全量 val + oracle 参照」（B7 对照臂用）：
    # Trainer 的 val 只抽 4 个 batch（1024 token）—— 与 B1-B3 同协议、可比，
    # 但小样本上 lf 有 ±0.05 量级波动；full_val_batches>0 时补一个覆盖几乎全部
    # val 的低噪声数字，oracle_trans 给定则再报真实转移矩阵在同一 val 序列上的
    # 交叉熵 —— 模型 vs oracle 的差距才是「学会了没有」的精确判据。
    if full_val_batches:
        r["val_loss_full"] = float(evaluate(m, vld, n_batch=full_val_batches))
    if oracle_trans is not None:
        a, b = val_tok[:-1], val_tok[1:]      # 一阶源：上一个 token 即全部上下文
        r["oracle_nll_full"] = float(-np.log(oracle_trans[a, b] + 1e-300).mean())
    return r


def bench_architecture(steps, seed):
    tok, val = markov_train_val(6000, 1500, 64, seed)
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
    tok, val = markov_train_val(6000, 1500, 64, seed)
    # 两条等参数量布局：均 2 注意力 + 4 GDN，且注意力层落在 MoE 位置（1/3/5）的
    # 个数一致（各 1 个），故 FFN/MoE 混合结构也逐层相同 ⇒ 参数量严格相等。
    # 这样 B2 才只比较「空间排布」本身，不混杂「注意力层数」这一变量。
    #   interleave（Qwen3-Next）：注意力散布于中部 [1,4]
    #   edge（Mamba-3）        ：注意力集中在两端   [0,5]
    layouts = {
        "interleave": ["gdn", "attn", "gdn", "gdn", "attn", "gdn"],
        "edge":       ["attn", "gdn", "gdn", "gdn", "gdn", "attn"],
    }
    rows = []
    params_seen = {}
    for lay, lst in layouts.items():
        cfg = base_cfg(layers=lst)
        r = run_train(cfg, tok, val, steps, seed)
        r["name"] = f"layout={lay}"
        r["layout"] = lay
        r["layers_list"] = lst
        params_seen[lay] = r["params"]
        rows.append(r)
        print(f"  layout={lay:<16} loss={r['final_loss']:.4f} val={r['val_loss']:.4f} "
              f"params={r['params']}", flush=True)
    # 硬断言：两布局参数量必须相等，否则本对比不成立（G27 修复的混杂）
    assert params_seen["interleave"] == params_seen["edge"], (
        f"B2 两布局参数量不等：interleave={params_seen['interleave']} "
        f"edge={params_seen['edge']}（layout 对比必须等参数，否则混杂注意力层数）")
    return rows


def bench_optimizer(steps, seed):
    tok, val = markov_train_val(6000, 1500, 64, seed)
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


def bench_learnable(steps, seed):
    """B7 可学性对照臂 —— 证明基准不是「永远学不会」的钝器。

    主基准（B1-B3）的 2 阶 vocab=64 任务在 6k token 规模下真实数据稀缺
    （4096 个上下文状态 × 每状态仅 ~1.5 次观测），任何配置都学不会
    （learned fraction 全负、全员贴在 uniform 基线）。这留下一个合理质疑：
    「是架构学不会，还是基准本身就是钝器？」本臂用**同一模型规模、同一训练
    协议、同一 seed**，只把语料换成规模内可学的 1 阶马尔可夫（vocab=8、
    60k train / 15k val 同源留出，H=1.5533）：
      - learned fraction 应转正到 ≈1（模型交叉熵逼近真实转移矩阵的 oracle 值）；
      - 三架构配置在同一可学任务上的差异，才有统计意义（不再全员贴 uniform）。
    步数取 600（探针实测 Hybrid 在 step≈60 即达 lf≈1.0，600 步确保三配置
    均已收敛，比较的是容量而非优化速度）。
    """
    V, ORDER, N_TR, N_VA = 8, 1, 60_000, 15_000
    tok, val = markov_train_val(N_TR, N_VA, V, seed, order=ORDER)
    trans = markov_transitions(V, seed, ORDER)
    er = markov_entropy_rate(vocab=V, seed=seed, order=ORDER)
    H, uniform = float(er["H"]), float(er["log_vocab"])
    n_full = N_VA // (8 * 32) + 2          # 覆盖几乎全部 val 的 batch 数
    rows = []
    variants = [
        ("Hybrid 3:1 (GDN+MLA)", dict(hybrid_ratio=3)),
        ("全注意力 (MLA)", dict(hybrid_ratio=0)),
        ("全 GDN (无注意力)", dict(hybrid_ratio=10 ** 6)),
    ]
    for name, kw in variants:
        cfg = base_cfg(vocab_size=V, **kw)
        r = run_train(cfg, tok, val, steps, seed,
                      full_val_batches=n_full, oracle_trans=trans)
        r["name"] = name
        r["learned_fraction"] = (uniform - r["val_loss"]) / (uniform - H)
        r["learned_fraction_full"] = (uniform - r["val_loss_full"]) / (uniform - H)
        rows.append(r)
        print(f"  {name:<24} loss={r['final_loss']:.4f} "
              f"val={r['val_loss']:.4f}(lf={r['learned_fraction']:+.3f}) "
              f"full_val={r['val_loss_full']:.4f}"
              f"(lf={r['learned_fraction_full']:+.3f}) "
              f"oracle={r['oracle_nll_full']:.4f} H={H:.4f} "
              f"{r['ms_per_step']:.0f}ms/step", flush=True)
    return {"corpus": {"order": ORDER, "vocab": V, "n_train": N_TR,
                       "n_val": N_VA, "same_source": True,
                       "note": "markov_train_val 单源双采样（同源留出，门禁 G26 同款）"},
            "H": H, "uniform": uniform, "steps": steps, "rows": rows}


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
    print("B0 熵下界（可机检 · 门禁 G24）")
    print("=" * 68)
    er = markov_entropy_rate(vocab=64, seed=a.seed)
    print(f"  2 阶马尔可夫源熵率 H = {er['H']:.4f} nats "
          f"（解析 {er['H_analytic']:.4f} / oracle {er['H_oracle']:.4f}，"
          f"差 {er['abs_diff']:.2e} < {er['tol']:.0e}）")
    print(f"  uniform 基线 = log(64) = {er['log_vocab']:.4f} nats；"
          f"任何模型在该源上的交叉熵 ≥ H（Gibbs 不等式）")
    print(f"  learned fraction 上限 = (uniform − loss)/(uniform − H) = 1.0")
    print()

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
    print("=" * 68)
    print("B7 可学性对照臂（同规模模型 · 规模内可学语料）")
    print("=" * 68)
    b7 = bench_learnable(600, a.seed)
    print()

    out = {"seed": a.seed, "steps": steps,
           "entropy": er,
           "architecture": b1, "layout": b2, "optimizer": b3,
           "quantization": b4, "cache": b5, "speculative": b6,
           "learnable": b7,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "results", "bench.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {path}")


if __name__ == "__main__":
    main()
