#!/usr/bin/env python
"""推理脚本 —— 从 checkpoint 加载模型并生成文本。

用法：
    python scripts/generate.py --ckpt runs/default/model.npz \
        --tokenizer runs/default/tokenizer.txt --prompt "the quick brown"
    python scripts/generate.py --ckpt runs/default/model.npz --interactive

加载时会强制校验 sha256 与参数指纹 —— 损坏的 checkpoint 会直接报错，
不会静默生成一个"看起来像话"的乱码结果。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tianji.l2_tokenizer import BPE                       # noqa: E402
from tianji.l8_infer import generate, sample_top_p        # noqa: E402
from tianji.model import TianjiConfig, TianjiModel        # noqa: E402
from tianji.persist import load_checkpoint, CheckpointError  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="天玑 TIANJI 推理脚本")
    ap.add_argument("--ckpt", type=str, required=True, help="model.npz")
    ap.add_argument("--tokenizer", type=str, default=None, help="tokenizer.txt")
    ap.add_argument("--prompt", type=str, default="the quick brown")
    ap.add_argument("-n", "--new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interactive", action="store_true", help="进入交互模式")
    ap.add_argument("--no-verify", action="store_true", help="跳过校验（不推荐）")
    a = ap.parse_args()

    # ---- 加载
    t0 = time.time()
    try:
        P, info = load_checkpoint(a.ckpt, verify=not a.no_verify)
    except CheckpointError as e:
        print(f"checkpoint 加载失败：{e}", file=sys.stderr)
        return 1
    cfg = TianjiConfig(**info["config"])
    model = TianjiModel(cfg, P)
    print(f"[load] {a.ckpt}")
    print(f"[load] 参数 {info['n_params']:,}  "
          f"训练步数={info['meta'].get('steps')}  "
          f"最终 loss={info['meta'].get('final_loss'):.4f}  "
          f"({time.time()-t0:.2f}s)")
    if not a.no_verify:
        print(f"[load] sha256 校验通过  {info['weights_sha256'][:32]}...")

    tok = a.tokenizer or os.path.join(os.path.dirname(a.ckpt), "tokenizer.txt")
    bpe = None
    if os.path.exists(tok):
        bpe = BPE.load(tok)
        print(f"[load] 词表 {bpe.size}  ({tok})")

    def run(prompt: str, n: int) -> str:
        if bpe is not None:
            pid = np.asarray(bpe.encode(prompt), dtype=np.int64)
            if pid.size == 0:
                pid = np.asarray([0], dtype=np.int64)
        else:
            # 无词表时退化为字符级（用 UTF-8 字节）
            pid = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        t1 = time.time()
        out = generate(model, pid, max_new=n, temperature=a.temperature,
                       top_p=a.top_p, seed=a.seed)
        dt = time.time() - t1
        if bpe is not None:
            text = bpe.decode(np.asarray(out))
        else:
            text = np.asarray(out, dtype=np.uint8).tobytes().decode("utf-8", errors="replace")
        print(f"[{n} token / {dt:.2f}s = {n/max(dt,1e-9):.1f} tok/s]")
        return text

    if a.interactive:
        print("\n交互模式（Ctrl+C 或空行退出）")
        while True:
            try:
                p = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not p:
                break
            print(run(p, a.new_tokens))
            print()
        return 0

    print(f"\n提示: {a.prompt!r}")
    print(f"生成: {run(a.prompt, a.new_tokens)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
