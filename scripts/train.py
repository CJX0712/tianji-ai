#!/usr/bin/env python
"""训练脚本 —— 端到端跑通「语料 → BPE → 模型 → checkpoint」。

用法：
    python scripts/train.py --steps 300                      # 用内置语料
    python scripts/train.py --data my.txt --steps 500        # 用自己的语料
    python scripts/train.py --steps 300 --out runs/exp1      # 指定输出目录

产出：
    <out>/model.npz            权重
    <out>/model.npz.meta.json  元数据（配置/参数表/校验和）
    <out>/tokenizer.txt        词表（BPE 合并表）
    <out>/train_log.json       训练曲线
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tianji.data import default_corpus                       # noqa: E402
from tianji.l2_tokenizer import BPE                          # noqa: E402
from tianji.l6_optimizer import HybridOptimizer, Muon2, Muon, AdamW  # noqa: E402
from tianji.l7_train import DataLoader, Trainer              # noqa: E402
from tianji.model import TianjiConfig, TianjiModel, init_params  # noqa: E402
from tianji.persist import save_checkpoint                   # noqa: E402


def build_cfg(vocab: int, **kw) -> TianjiConfig:
    d_model = kw.pop("d_model", 96)
    return TianjiConfig(
        vocab_size=vocab, d_model=d_model, n_layers=kw.pop("n_layers", 8),
        n_heads=4, n_kv_heads=2,
        d_latent_q=max(d_model // 2, 16), d_latent_kv=max(d_model // 3, 8),
        d_rope=8, d_ff=d_model * 2, d_ff_expert=d_model,
        n_experts=kw.pop("n_experts", 4), moe_top_k=2,
        max_len=kw.pop("max_len", 1024), mtp_depth=1,
        hybrid_ratio=kw.pop("hybrid_ratio", 3),
        layout=kw.pop("layout", "interleave"),
    )


def make_optimizer(kind: str, lr: float):
    if kind == "muon2":
        return HybridOptimizer(muon2=Muon2(ns_steps=3), adamw=AdamW(lr=lr))
    if kind == "muon":
        return HybridOptimizer(muon2=Muon(ns_steps=5), adamw=AdamW(lr=lr))
    if kind == "adamw":
        class _AO(HybridOptimizer):
            def is_muon(self, name, p):
                return False
        return _AO(muon2=Muon2(ns_steps=3), adamw=AdamW(lr=lr))
    raise ValueError(f"未知优化器: {kind}")


def main():
    ap = argparse.ArgumentParser(description="天玑 TIANJI 训练脚本")
    ap.add_argument("--data", type=str, default=None, help="语料文本路径（默认用内置语料）")
    ap.add_argument("--out", type=str, default="runs/default", help="输出目录")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", choices=["muon2", "muon", "adamw"], default="muon2")
    ap.add_argument("--layout", choices=["interleave", "edge"], default="interleave")
    ap.add_argument("--hybrid-ratio", type=int, default=3)
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--print-every", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # ---- 语料
    if a.data:
        with open(a.data, encoding="utf-8") as f:
            corpus = f.read()
        print(f"[data] {a.data}  {len(corpus)} 字符")
    else:
        corpus = default_corpus()
        print(f"[data] 内置语料  {len(corpus)} 字符")

    # ---- 分词器
    tb = time.time()
    bpe = BPE(vocab_size=a.vocab).train(corpus)
    tok_path = os.path.join(a.out, "tokenizer.txt")
    bpe.save(tok_path)
    ids = np.asarray(bpe.encode(corpus), dtype=np.int64)
    nbytes = len(corpus.encode("utf-8"))
    print(f"[bpe ] 词表 {bpe.size}  {len(ids)} token  字节压缩比 {nbytes/len(ids):.2f}x  "
          f"({time.time()-tb:.1f}s)")

    # ---- 模型
    cfg = build_cfg(max(bpe.size, a.vocab), d_model=a.d_model, n_layers=a.layers,
                    hybrid_ratio=a.hybrid_ratio, layout=a.layout,
                    n_experts=a.n_experts, max_len=a.seq * 4)
    model = TianjiModel(cfg, init_params(cfg, seed=a.seed))
    layout_str = "".join("A" if l == "attn" else "L" for l in model.layers)
    print(f"[model] {cfg.n_layers} 层 布局={layout_str} (A=MLA, L=GatedDeltaNet)  "
          f"参数 {model.param_count():,}  d_model={cfg.d_model}")

    # ---- 训练
    n = int(len(ids) * 0.9)
    tr_tok, va_tok = ids[:n], ids[n:]
    ld = DataLoader(tr_tok, seq_len=a.seq, batch_size=a.batch, seed=a.seed)
    vld = DataLoader(va_tok, seq_len=a.seq, batch_size=a.batch, seed=a.seed + 1)
    opt = make_optimizer(a.optim, a.lr)
    tr = Trainer(model, opt, ld, total_steps=a.steps, warmup=max(a.steps // 10, 5),
                 lr_max=a.lr, val_loader=vld, eval_every=max(a.steps // 5, 1))
    print(f"[train] {a.steps} 步  optim={a.optim}  lr={a.lr}  seq={a.seq}  batch={a.batch}")
    log = tr.run(a.steps, verbose=True,
                 print_every=a.print_every or max(a.steps // 10, 1))
    print(f"[train] 完成 {log.seconds:.1f}s  "
          f"loss {log.losses[0]:.4f} -> {log.losses[-1]:.4f}  "
          f"ppl {np.exp(log.losses[-1]):.2f}")
    if log.val:
        print("[train] 验证 loss: " + " -> ".join(f"{v:.4f}" for _, v in log.val))

    # ---- 保存
    ckpt = os.path.join(a.out, "model.npz")
    digest = save_checkpoint(
        ckpt, cfg, model.P,
        meta={"steps": a.steps, "optim": a.optim, "lr": a.lr, "seed": a.seed,
              "final_loss": float(log.losses[-1]),
              "val_loss": float(log.val[-1][1]) if log.val else None,
              "vocab_size": bpe.size, "corpus_chars": len(corpus),
              "seconds": log.seconds},
        tokenizer_path=tok_path,
    )
    with open(os.path.join(a.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"losses": [float(x) for x in log.losses],
                   "val": [[int(s), float(v)] for s, v in log.val],
                   "grad_norms": [float(x) for x in log.grad_norms],
                   "lr_trace": [float(x) for x in log.lr_trace],
                   "seconds": log.seconds}, f, indent=2)
    print(f"[save ] {ckpt}")
    print(f"[save ] sha256={digest[:32]}...  tokenizer={tok_path}")
    print(f"[done ] 总耗时 {time.time()-t0:.1f}s")
    print(f"\n下一步推理：\n  python scripts/generate.py --ckpt {ckpt} "
          f"--tokenizer {tok_path} --prompt 'the quick brown'")


if __name__ == "__main__":
    main()
