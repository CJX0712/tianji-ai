#!/usr/bin/env python
"""最小可运行示例 —— 60 秒内跑完，无需任何外部数据或网络。

这是「交付前自查」的最后一道：证明一个全新环境从 import 到生成文本
整条链路是通的。CI 与 Dockerfile 的默认 CMD 都可以用它。

    python scripts/quickstart.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tianji import (
    BPE, DataLoader, HybridOptimizer, TianjiConfig, TianjiModel,
    init_params, Trainer, generate, run_all_gates,
)
from tianji.data import default_corpus
from tianji.persist import save_checkpoint, load_checkpoint


def main() -> int:
    t0 = time.time()
    print("=" * 62)
    print("天玑 TIANJI · 最小可运行示例")
    print("=" * 62)

    # 1. 分词
    corpus = default_corpus()
    bpe = BPE(vocab_size=512).train(corpus)
    ids = np.asarray(bpe.encode(corpus), dtype=np.int64)
    nbytes = len(corpus.encode("utf-8"))
    print(f"1. 分词   词表 {bpe.size}，{len(ids)} token，"
          f"字节压缩比 {nbytes/len(ids):.2f}x（字节级 BPE，中文按 UTF-8 字节计）")
    assert bpe.decode(bpe.encode(corpus[:200])) == corpus[:200], "分词往返失败"

    # 2. 模型
    cfg = TianjiConfig(vocab_size=max(bpe.size, 512), d_model=64, n_layers=4,
                       d_latent_q=32, d_latent_kv=24, d_rope=8,
                       d_ff=96, d_ff_expert=48, n_experts=4, max_len=256,
                       mtp_depth=1, hybrid_ratio=3)
    model = TianjiModel(cfg, init_params(cfg, seed=0))
    layout = "".join("A" if l == "attn" else "L" for l in model.layers)
    print(f"2. 模型   {cfg.n_layers} 层 布局={layout} (A=MLA, L=GatedDeltaNet)，"
          f"参数 {model.param_count():,}")

    # 3. 前向 + 反向
    tok = ids[:32].reshape(1, -1)
    tgt = ids[1:33].reshape(1, -1)
    out = model.forward(tok, tgt)
    G = model.backward()
    print(f"3. 前反向  loss={out['loss']:.4f} (ppl {np.exp(out['loss']):.1f})，"
          f"梯度 {len(G)} 组，全部有限={all(np.all(np.isfinite(v)) for v in G.values())}")
    assert np.isfinite(out["loss"])

    # 4. 训练
    ld = DataLoader(ids, seq_len=24, batch_size=8, seed=0)
    tr = Trainer(model, HybridOptimizer(), ld, total_steps=30, warmup=5, lr_max=0.05)
    log = tr.run(30, verbose=False)
    print(f"4. 训练   30 步 {log.seconds:.1f}s，"
          f"loss {log.losses[0]:.4f} -> {log.losses[-1]:.4f}")
    assert log.losses[-1] < log.losses[0], "训练未降低损失"

    # 5. 持久化（含校验和）
    ck = os.path.join(os.path.dirname(__file__), "..", "runs", "quickstart", "model.npz")
    digest = save_checkpoint(ck, cfg, model.P, meta={"demo": True})
    P2, info = load_checkpoint(ck)
    same = all(np.array_equal(model.P[k], P2[k]) for k in model.P)
    print(f"5. 持久化 保存+加载 逐位一致={same}，sha256={digest[:16]}...")
    assert same, "checkpoint 往返不一致"

    # 6. 生成
    prompt = "the quick brown"
    pid = np.asarray(bpe.encode(prompt), dtype=np.int64)
    out_ids = generate(model, pid, max_new=24, temperature=0.8, top_p=0.9, seed=0)
    text = bpe.decode(np.asarray(out_ids))
    print(f"6. 生成   提示={prompt!r}")
    print(f"          结果={text!r}")

    # 7. 门禁
    print("7. 门禁   运行中 ...")
    rep = run_all_gates(seed=0, steps=20)
    print(f"          {rep.passed}/{rep.total} 通过")

    ok = rep.all_green
    print("=" * 62)
    print(f"{'✅ 全部通过' if ok else '❌ 存在失败'}   总耗时 {time.time()-t0:.1f}s")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
