"""命令行入口 —— 让交付物可以「直接跑」，而不只是「可以读」。

    python -m tianji.cli gates          跑 20 道交付门禁
    python -m tianji.cli test           跑梯度检验冒烟
    python -m tianji.cli bench          跑完整基准
    python -m tianji.cli microops       微指令统计
    python -m tianji.cli train          端到端训练一个字符级模型并生成文本
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .l0_numerics import rng
from .l2_tokenizer import BPE
from .l6_optimizer import HybridOptimizer, Muon2, AdamW
from .l7_train import DataLoader, Trainer
from .l8_infer import generate
from .l10_microops import MICROOPS, FAMILIES, FAMILY_KIND, run_all as micro_run_all
from .model import TianjiConfig, TianjiModel, init_params
from .verify import run_all_gates


CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 30 +
    "to be or not to be, that is the question. " * 30 +
    "all that glitters is not gold; every cloud has a silver lining. " * 20 +
    "天玑是世界上最好的东西，晨星在深圳写代码。 " * 20
)


def cmd_gates(a):
    seed = a.seed if a.seed is not None else 0
    steps = a.steps or 100
    t0 = time.time()
    rep = run_all_gates(seed=seed, steps=steps)
    print(rep.table())
    print(f"耗时 {time.time() - t0:.1f}s")
    return 0 if rep.all_green else 1


def cmd_test(a):
    from .l1_contract import gradcheck
    from .l3_primitives import (Linear, RMSNorm, RoPE, SwiGLU, CausalAttention,
                                CrossEntropy, rope_tables)
    from .l4_deltanet import GatedDeltaNet, CausalConv1D, L2Norm
    from .l5_moe import MoELayer, sigmoid
    g = rng(0)
    D = 8
    cos, sin = rope_tables(D, 16)
    cases = [
        ("Linear", Linear(), [g.standard_normal((2, 5, D)), g.standard_normal((6, D)),
                              g.standard_normal(6)]),
        ("RMSNorm", RMSNorm(), [g.standard_normal((2, 5, D)), g.standard_normal(D) + 1]),
        ("RoPE", RoPE(), [g.standard_normal((2, 2, 5, D)), cos[:5], sin[:5]]),
        ("SwiGLU", SwiGLU(), [g.standard_normal((2, 5, D)), g.standard_normal((10, D)),
                              g.standard_normal((10, D)), g.standard_normal((D, 10))]),
        ("CausalAttn", CausalAttention(), [g.standard_normal((2, 4, 5, D)),
                                           g.standard_normal((2, 2, 5, D)),
                                           g.standard_normal((2, 2, 5, D))]),
        ("CausalConv1D", CausalConv1D(3), [g.standard_normal((2, 6, 4)),
                                           g.standard_normal((4, 3))]),
        ("L2Norm", L2Norm(), [g.standard_normal((2, 2, 5, D))]),
        ("GatedDeltaNet", GatedDeltaNet(), [g.standard_normal((2, 2, 6, 4)),
                                            g.standard_normal((2, 2, 6, 4)),
                                            g.standard_normal((2, 2, 6, 4)),
                                            sigmoid(g.standard_normal((2, 2, 6))),
                                            sigmoid(g.standard_normal((2, 2, 6)))]),
    ]
    E, dm, dff, N = 4, 6, 8, 7
    cases.append(("MoELayer", MoELayer(E, 2, True),
                  [g.standard_normal((N, dm)), g.standard_normal((E, dm)),
                   g.standard_normal((E, dff, dm)), g.standard_normal((E, dff, dm)),
                   g.standard_normal((E, dm, dff)), np.zeros(E),
                   g.standard_normal((dff, dm)), g.standard_normal((dff, dm)),
                   g.standard_normal((dm, dff))]))
    n_ok = 0
    for name, m, args in cases:
        rep, _, _ = gradcheck(m, args, seed=1, name=name)
        print(rep)
        n_ok += rep.passed
    print(f"\n{n_ok}/{len(cases)} 模块通过梯度检验")
    return 0 if n_ok == len(cases) else 1


def cmd_bench(a):
    from bench.run_bench import main as bench_main
    sys.argv = ["run_bench.py", "--steps", str(a.steps or 120), "--seed",
                str(a.seed if a.seed is not None else 0)]
    bench_main()
    return 0


def cmd_microops(a):
    rep = micro_run_all(seed=0)
    print(f"微指令总数: {len(MICROOPS)}   家族数: {len(FAMILIES)}")
    print(f"可执行: {rep['ok']}/{rep['total']}   失败: {len(rep['failed'])}")
    core = [f for f, k in FAMILY_KIND.items() if k == "core"]
    sweep = [f for f, k in FAMILY_KIND.items() if k == "sweep"]
    print(f"\ncore 家族 ({len(core)}，成员语义独立): {', '.join(sorted(core))}")
    print(f"\nsweep 家族 ({len(sweep)}，成员为参数扫描): {', '.join(sorted(sweep))}")
    if a.show:
        for f in sorted(FAMILIES):
            print(f"  {f:<8} {len(FAMILIES[f]):>2} 条  [{FAMILY_KIND.get(f, 'core')}]")
    return 0


def cmd_train(a):
    print("训练分词器 ...")
    bpe = BPE(vocab_size=a.vocab).train(CORPUS)
    ids = np.asarray(bpe.encode(CORPUS), dtype=np.int64)
    print(f"词表 {bpe.size}，语料 {len(ids)} token（原 {len(CORPUS)} 字符，"
          f"压缩比 {len(CORPUS) / len(ids):.2f}×）")

    cfg = TianjiConfig(
        vocab_size=max(bpe.size, a.vocab), d_model=a.d_model, n_layers=a.layers,
        n_heads=4, n_kv_heads=2, d_latent_q=a.d_model // 2,
        d_latent_kv=max(a.d_model // 3, 8), d_rope=8,
        d_ff=a.d_model * 2, d_ff_expert=a.d_model, n_experts=4, moe_top_k=2,
        max_len=a.seq * 4, mtp_depth=1, hybrid_ratio=3, layout=a.layout,
    )
    seed = a.seed if a.seed is not None else 0
    model = TianjiModel(cfg, init_params(cfg, seed=seed))
    print(f"模型: {cfg.n_layers} 层 布局={''.join('A' if l == 'attn' else 'L' for l in model.layers)} "
          f"（A=MLA 全注意力, L=GatedDeltaNet）, 参数 {model.param_count()}")

    n = int(len(ids) * 0.9)
    tr_tok, va_tok = ids[:n], ids[n:]
    ld = DataLoader(tr_tok, seq_len=a.seq, batch_size=a.batch, seed=seed)
    vld = DataLoader(va_tok, seq_len=a.seq, batch_size=a.batch, seed=seed + 1)
    opt = HybridOptimizer(muon2=Muon2(ns_steps=3), adamw=AdamW(lr=a.lr))
    steps = a.steps or 100
    tr = Trainer(model, opt, ld, total_steps=steps, warmup=max(steps // 10, 5),
                 lr_max=a.lr, val_loader=vld, eval_every=max(steps // 5, 1))
    log = tr.run(steps, verbose=True, print_every=max(steps // 10, 1))
    print(f"\n训练完成: {log.step} 步, {log.seconds:.1f}s, "
          f"最终 loss {log.losses[-1]:.4f} (ppl {np.exp(log.losses[-1]):.2f})")
    if log.val:
        print(f"验证 loss 轨迹: " + " -> ".join(f"{v:.4f}" for _, v in log.val))

    if a.generate:
        prompt = a.prompt or "the quick brown"
        pid = np.asarray(bpe.encode(prompt), dtype=np.int64)
        out = generate(model, pid, max_new=a.new_tokens, temperature=a.temperature,
                       top_p=a.top_p, seed=a.seed if a.seed is not None else 0)
        print(f"\n提示: {prompt!r}")
        print(f"生成: {bpe.decode(np.asarray(out))!r}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tianji", description="天玑 TIANJI · 混合架构语言模型全栈")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_g = sub.add_parser("gates", help="跑 20 道交付门禁")
    p_g.add_argument("--steps", type=int, default=None)
    p_g.add_argument("--seed", type=int, default=None)
    sub.add_parser("test", help="模块梯度检验冒烟")
    p_b = sub.add_parser("bench", help="跑完整基准")
    p_b.add_argument("--steps", type=int, default=None)
    p_b.add_argument("--seed", type=int, default=None)
    p_m = sub.add_parser("microops", help="微指令统计")
    p_m.add_argument("--show", action="store_true", help="列出全部家族")

    p_t = sub.add_parser("train", help="端到端训练 + 生成")
    p_t.add_argument("--steps", type=int, default=None)
    p_t.add_argument("--seed", type=int, default=None)
    p_t.add_argument("--vocab", type=int, default=400)
    p_t.add_argument("--d-model", type=int, default=64)
    p_t.add_argument("--layers", type=int, default=6)
    p_t.add_argument("--seq", type=int, default=32)
    p_t.add_argument("--batch", type=int, default=8)
    p_t.add_argument("--lr", type=float, default=0.03)
    p_t.add_argument("--layout", choices=["interleave", "edge"], default="interleave")
    p_t.add_argument("--generate", action="store_true", default=True)
    p_t.add_argument("--prompt", type=str, default="")
    p_t.add_argument("--new-tokens", type=int, default=40)
    p_t.add_argument("--temperature", type=float, default=0.8)
    p_t.add_argument("--top-p", type=float, default=0.9)

    a = ap.parse_args(argv)
    return {
        "gates": cmd_gates, "test": cmd_test, "bench": cmd_bench,
        "microops": cmd_microops, "train": cmd_train,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
