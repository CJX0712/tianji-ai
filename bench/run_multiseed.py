"""B8 · 多 seed 稳定性 —— 把「B1-B3 的差异落在噪声内」从定性变成定量。

背景：主基准（2 阶 vocab=64、6k token、120 步）里三族配置的验证损失差异
（~0.04~0.07 nats）此前被 BENCHMARKS.md 判定为「噪声内、不可作架构论断」，
但那只是**推断**：依据是 Trainer 的 val 只抽 1024 token（4 batch，子样本噪声
±0.05 量级）以及排序曾随同源修复而翻转（G26）。本脚本补上**直接测量**：
同一协议跑 5 个 seed，报告每配置均值 ± 标准差，并用**配对**比较（同 seed 下
两配置之差）算 t 值与胜负局数——每族配置在相同 seed 下共享同一马尔可夫源，
配对才是正确的统计口径。

协议与 B1-B3 完全一致（同语料规模/同步数/同 batch），只改 seed ——
seed 同时改变马尔可夫源、参数初始化与 batch 顺序，即「同一实验换一个随机Draw」。
H 随 seed 变（每个 seed 是不同转移矩阵），故 learned fraction 用该 seed 自己的 H。

B2 布局族（G27 起）用两条**等参数量**布局（均 2 注意力 + 4 GDN，MoE 重叠一致），
只比较空间排布本身，不混杂「注意力层数」这一变量。

运行：
    python bench/run_multiseed.py                 # 三族全跑（约 15 分钟）
    python bench/run_multiseed.py --only layout   # 只跑布局族（约 4 分钟）
产物：bench/results/multiseed.json（由门禁 G24·E3c 读取断言）
     支持增量合并：已跑过的族不会被覆盖（可用 --only 分批跑）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tianji.data import markov_train_val, markov_entropy_rate   # noqa: E402
from run_bench import base_cfg, run_train, arch_configs           # noqa: E402  （复用 B1-B3 同款协议 + B1 等参数配置）

SEEDS = [0, 1, 2, 3, 4]
STEPS = 120
V, ORDER, N_TR, N_VA = 64, 2, 6000, 1500
PROTOCOL = {"order": ORDER, "vocab": V, "n_train": N_TR, "n_val": N_VA,
            "steps": STEPS, "batch": 8, "seq": 32, "opt": "muon2",
            "same_source": True,
            "note": "协议同 B1-B3，只改 seed（源/初始化/batch 顺序同时改变）"}

ARCH_VARIANTS = [
    "全注意力 (MLA)", "Hybrid (2 attn+4 GDN)", "全 GDN (无注意力)",
]
# B1 三架构用 arch_configs(64) 产出的**等参数**配置（宽度匹配，残差 ≤ 0.1%），
# 修正旧版 hybrid_ratio=1e6 把「全 GDN」错配成 1 注意力+5 GDN 且三配置参数量不等的 bug。
# B2 两条等参数量布局：均 2 注意力 + 4 GDN，且注意力层落在 MoE 位置（1/3/5）的
# 个数一致（各 1 个）⇒ 参数量严格相等，只比较空间排布本身（G27 修复的混杂）。
LAYOUT_VARIANTS = [
    ("interleave", dict(layers=["gdn", "attn", "gdn", "gdn", "attn", "gdn"])),
    ("edge", dict(layers=["attn", "gdn", "gdn", "gdn", "gdn", "attn"])),
]
OPT_VARIANTS = [("adamw", 3e-3), ("muon", 0.03), ("muon2", 0.03)]


def _assert_layout_equal_params(rows):
    ps = {r["name"]: r["params"] for r in rows}
    assert len(set(ps.values())) == 1, (
        f"B2 两布局参数量不等（layout 对比必须等参数）：{ps}")


def _assert_arch_equal_params(rows):
    ps = {r["name"]: r["params"] for r in rows}
    mx, mn = max(ps.values()), min(ps.values())
    assert mx - mn <= 0.001 * mx, (
        f"B1 三架构参数量不等（差 {mx - mn}，超 0.1%）：{ps}（架构对比必须等参数）")


def _run_variant(name, cfg, opt_kind, lr, seeds, steps):
    """跑一个配置的多个 seed，返回 {name, params, per_seed, mean/std}。"""
    per_seed = []
    logv = float(np.log(V))
    for s in seeds:
        # 每个 seed 是独立马尔可夫源 ⇒ H 随 seed 重算（与训练语料同源）
        H = float(markov_entropy_rate(vocab=V, seed=s, order=ORDER)["H"])
        tok, val = markov_train_val(N_TR, N_VA, V, s, order=ORDER)
        r = run_train(cfg, tok, val, steps, s, opt_kind=opt_kind, lr=lr)
        lf = (logv - r["val_loss"]) / (logv - H)
        per_seed.append({"seed": s, "final_loss": r["final_loss"],
                         "val_loss": r["val_loss"], "H": H,
                        "lf": float(lf), "seconds": r["seconds"]})
        print(f"  seed={s} {name:<22} loss={r['final_loss']:.4f} "
              f"val={r['val_loss']:.4f} lf={lf:+.3f} "
              f"{r['seconds']:.0f}s", flush=True)
    vals = np.array([p["val_loss"] for p in per_seed])
    lfs = np.array([p["lf"] for p in per_seed])
    print(f"  => {name:<22} mean_val={vals.mean():.4f} "
          f"std={vals.std(ddof=1):.4f}  mean_lf={lfs.mean():+.3f}", flush=True)
    return {"name": name, "params": int(r["params"]),
            "mean_val": float(vals.mean()), "std_val": float(vals.std(ddof=1)),
            "mean_lf": float(lfs.mean()), "std_lf": float(lfs.std(ddof=1)),
            "per_seed": per_seed}


def family_architecture(seeds, steps):
    cfgs = arch_configs(64)
    rows = [_run_variant(n, cfgs[n], "muon2", 0.03, seeds, steps) for n in ARCH_VARIANTS]
    _assert_arch_equal_params(rows)
    return rows


def family_layout(seeds, steps):
    rows = [_run_variant(f"layout={n}", base_cfg(**k), "muon2", 0.03, seeds, steps)
            for n, k in LAYOUT_VARIANTS]
    _assert_layout_equal_params(rows)
    return rows


def family_optimizer(seeds, steps):
    return [_run_variant(n, base_cfg(), n, lr, seeds, steps)
            for n, lr in OPT_VARIANTS]


def stats(rows, seeds):
    """配对比较：同 seed 下两配置之差的均值 / 标准差 / t 值 / 胜负局数。"""
    pairwise = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            da = {p["seed"]: p["val_loss"] for p in a["per_seed"]}
            db = {p["seed"]: p["val_loss"] for p in b["per_seed"]}
            ds = np.array([da[s] - db[s] for s in seeds])
            sd = float(ds.std(ddof=1))
            t = float(ds.mean() / (sd / np.sqrt(len(seeds)))) if sd > 0 else float("inf")
            pairwise.append({"a": a["name"], "b": b["name"],
                             "delta_mean": float(ds.mean()), "diff_std": sd,
                             "t": t,
                             "wins_a": int((ds < 0).sum()),
                             "wins_b": int((ds > 0).sum())})
    best = min(rows, key=lambda r: r["mean_val"])
    # 排序稳定性：最优配置在**每一个** seed 上都是最优
    consistent = True
    for k in range(len(seeds)):
        name_k = min((r["per_seed"][k]["val_loss"], r["name"]) for r in rows)[1]
        if name_k != best["name"]:
            consistent = False
    return pairwise, best["name"], bool(consistent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "arch", "layout", "opt"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    seeds, steps = list(a.seeds), a.steps

    path = os.path.join(os.path.dirname(__file__), "results", "multiseed.json")
    if os.path.exists(path):
        out = json.load(open(path, encoding="utf-8"))
    else:
        out = {"protocol": PROTOCOL, "seeds": seeds, "steps": steps,
               "families": {}}
    if "configs" in out:                      # 迁移旧结构（单族版本）
        out.setdefault("families", {})["architecture"] = out.pop("configs")
    out["protocol"], out["seeds"], out["steps"] = PROTOCOL, seeds, steps

    todo = []
    if a.only in ("all", "arch"):
        todo.append(("architecture", family_architecture))
    if a.only in ("all", "layout"):
        todo.append(("layout", family_layout))
    if a.only in ("all", "opt"):
        todo.append(("optimizer", family_optimizer))

    print("=" * 68)
    print(f"B8 多 seed 稳定性（{len(seeds)} seed × 各族配置，协议同 B1-B3）")
    print("=" * 68, flush=True)
    t0 = time.time()
    for key, fn in todo:
        print(f"\n-- {key} --", flush=True)
        rows = fn(seeds, steps)
        pw, best, consistent = stats(rows, seeds)
        out["families"][key] = {"configs": rows, "pairwise": pw,
                                "best": best, "consistent": consistent}
    out["seconds_this_run"] = time.time() - t0
    out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    for key, fam in out["families"].items():
        if not isinstance(fam, dict) or "best" not in fam:
            continue      # 该族本轮未跑（历史遗留结构）
        print(f"{key}: best={fam['best']} consistent={fam['consistent']}")
        for p in fam["pairwise"]:
            print(f"   {p['a']} vs {p['b']}: Δmean={p['delta_mean']:+.4f} "
                  f"diff_std={p['diff_std']:.4f} t={p['t']:+.2f} "
                  f"({p['wins_a']}:{p['wins_b']})")
    print(f"\n结果已写入 {path}（本次 {out['seconds_this_run']:.0f}s）")


if __name__ == "__main__":
    main()
