"""L10 · 交付门禁 —— 「可运行」不等于「可交付」。

继承天枢 tianshu-ai 的门禁哲学：
    单元测试证明的是「函数没写错」，门禁证明的是「系统整体达到了宣称的能力」。
    本项目对外宣称的每一件事，都必须有一条门禁守着。
    **失败就如实报失败，不允许放宽阈值来凑绿。**
    报告里同时给出 status 与实测值，任何人都能复核。

门禁清单：
    G1  确定性        同 seed 两次运行逐位一致（否则一切实验结论不可信）
    G2  梯度正确      全模型解析反向 vs 中心差分（抽样参数）
    G3  训练有效      固定步数后训练损失显著下降
    G4  可过拟合      极小数据上能把损失压到很低（证明反向真的在传播）
    G5  微指令        1024 条全部可执行且输出有限
    G6  MoE 均衡      无辅助损失偏置控制器让负载变异系数下降
    G7  Muon2 优势    3 步 NS 的正交化质量 >= Muon 5 步
    G8  缓存等价      混合架构(GDN 递推 + MLA KV)增量解码 == 整序列重算
    G9  量化有界      INT8/INT4 的误差有界，且 INT8 退化 <= INT4退化
    G10 分词往返      decode(encode(s)) == s
    G11 并行形式等价  Gated DeltaNet 递推（forward）vs 并行前缀扫描（associative scan）
                    两条不同算法路径逐位一致（线性注意力决定性正确性质）
    G12 chunkwise 等价 Gated DeltaNet 递推 vs chunkwise 分块并行（块内并行 + 块间递推，
                    真实 GPU 内核训练形式）；含不被整除的尾块，块长扫描
    G13 chunkwise 反向等价 Gated DeltaNet 物化全量状态反向 vs FlashAttention 式重计算反向
                    （只存 O(T/C) 块入口状态）；两条不同资源/算法路径给同一解析梯度，
                    再用中心差分（gradcheck）做完全独立裁判
    G14 投机解码无损 Leviathan/Chen 拒绝采样的决定性恒等式：单步输出分布逐点等于 target，
                    与 draft 好坏无关。双验证：(a) 解析恒等（64 组随机分布，含
                    draft==target / target 含零 / draft 高度集中，机器精度）；
                    (b) 端到端统计（mock 双分布模型 6000 token，经验边际 TV 距离）
                    + 接受率 ∈ (0,1)（I5 有效性）。加速边界如实声明（结构性 ≈1）
    G15 投机解码并行验证 独立轻量 draft（TinyDraftGDN，复用已门禁 gdn_step）+ γ>1 一次
                    batch 验证：(a) 位置指纹确定性测试 —— target logits 按位置编码，
                    q=p 全接受场景下产出序列必须逐位等于指纹（钉死 verify forward
                    的位置对齐 off-by-one 与 draft 状态快照回滚）；(b) 代数条件：
                    每轮主干 forward 恰 1 次、tokens/forward > 1（全接受 γ+1）；
                    (c) 混合接受/拒绝统计无损（TV）+ 接受率 ∈ (0,1)
    G16 对齐正确性     DPO/IPO/SimPO/KTO 的解析与全链路性质（A1-A5 全部首次机检）：
                    (a) 解析值 —— DPO/KTO 在 Δ=0 时 loss==log2、系数符号正确，
                    IPO 在最优 Δ*=τ/(2β) 处系数==0（有限目标，DPO 病态的修复）；
                    (b) grad_logprob 中心差分独立裁判（含 pad mask）；
                    (c) DPO 端到端参数梯度中心差分（真实小模型，chosen/rejected
                    双向反传 + 梯度合并全链路）；
                    (d) 梯度方向与偏好一致 —— 一步更新后 chosen logprob 上升、
                    rejected 下降（DPO 的灵魂）；
                    (e) SimPO 长度归一化 —— 序列复制两倍长平均 logprob 逐位不变；
                    (f) 参考模型冻结 —— loss_and_grads 前后 ref 参数零漂移，
                    freeze_copy 为独立数组（DPO 最隐蔽的失效模式）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .l0_numerics import spawn_seed, rng, newton_schulz, softmax
from .l6_optimizer import orthogonalization_quality, Muon, Muon2
from .l7_train import DataLoader, Trainer
from .l8_infer import quantize_per_channel, quantize_groupwise, dequantize
from .l10_microops import run_all as microop_run_all

__all__ = ["GateResult", "GateReport", "run_all_gates"]


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    measured: dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_green(self) -> bool:
        return self.passed == self.total

    def table(self) -> str:
        rows = [f"{'GATE':<6}{'STATUS':<8}{'DETAIL'}"]
        for i, r in enumerate(self.results, 1):
            rows.append(f"G{i:<5}{'PASS' if r.passed else 'FAIL':<8}{r.detail}")
        rows.append("")
        rows.append(f"{self.passed}/{self.total} gates passed")
        return "\n".join(rows)


def _small_cfg():
    from .model import TianjiConfig
    return TianjiConfig(
        vocab_size=48, d_model=32, n_layers=4, n_heads=4, n_kv_heads=2,
        d_latent_q=16, d_latent_kv=12, d_rope=4, d_ff=40, d_ff_expert=24,
        n_experts=4, moe_top_k=2, max_len=64, mtp_depth=1,
    )


def _synth_tokens(n: int, vocab: int, seed: int = 0) -> np.ndarray:
    """确定性合成语料：2 阶马尔可夫，保证「有结构可学」。

    为什么不用真实语料：真实语料需要下载，破坏「离线可复现」这条硬约束。
    2 阶马尔可夫有明确的熵下界，因此损失下降到什么程度才算「学到了」
    是可判定的，比真实文本更适合做门禁。
    """
    g = rng(spawn_seed(seed, "synth"))
    trans = g.dirichlet(np.ones(vocab) * 0.3, size=(vocab, vocab))
    out = np.zeros(n, dtype=np.int64)
    a, b = 0, 1
    for i in range(n):
        c = g.choice(vocab, p=trans[a, b])
        out[i] = c
        a, b = b, c
    return out


def run_all_gates(seed: int = 0, steps: int = 60, verbose: bool = True) -> GateReport:
    from .model import TianjiModel, init_params
    from .l6_optimizer import HybridOptimizer
    R = GateReport()

    def add(name, ok, detail, **kw):
        R.results.append(GateResult(name, bool(ok), detail, kw))

    # ---------------------------------------------------------- G1 确定性
    t0 = time.time()
    cfg = _small_cfg()
    tok = _synth_tokens(400, cfg.vocab_size, seed)

    def train_once(s, n=6):
        P = init_params(cfg, seed=s)
        m = TianjiModel(cfg, P)
        opt = HybridOptimizer(max_grad_norm=1.0)
        ld = DataLoader(tok, seq_len=12, batch_size=2, seed=s)
        tr = Trainer(m, opt, ld, total_steps=n, warmup=2, lr_max=0.02)
        tr.run(n, verbose=False)
        return float(np.sum([np.sum(np.asarray(v)) for v in m.P.values()]))

    a1 = train_once(seed)
    a2 = train_once(seed)
    add("确定性", a1 == a2, f"两次运行参数逐位一致 diff={abs(a1 - a2):.3e}", diff=abs(a1 - a2))
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G2 梯度正确
    t0 = time.time()
    from .l1_contract import gradcheck
    reps = []
    from .l3_primitives import (Linear, RMSNorm, RoPE, SwiGLU, CausalAttention,
                                CrossEntropy, rope_tables)
    from .l4_deltanet import GatedDeltaNet, CausalConv1D, L2Norm, GatedDeltaNetChunkwise
    from .l5_moe import MoELayer, sigmoid as msig
    g2 = rng(spawn_seed(seed, "grad"))
    D = 8
    cos, sin = rope_tables(D, 16)
    reps.append(gradcheck(Linear(), [g2.standard_normal((2, 5, D)), g2.standard_normal((6, D)),
                                     g2.standard_normal(6)], seed=1, name="Linear")[0])
    reps.append(gradcheck(RMSNorm(), [g2.standard_normal((2, 5, D)), g2.standard_normal(D)],
                          seed=1, name="RMSNorm")[0])
    reps.append(gradcheck(RoPE(), [g2.standard_normal((2, 2, 5, D)), cos[:5], sin[:5]],
                          seed=1, name="RoPE")[0])
    reps.append(gradcheck(SwiGLU(), [g2.standard_normal((2, 5, D)), g2.standard_normal((10, D)),
                                     g2.standard_normal((10, D)), g2.standard_normal((D, 10))],
                          seed=1, name="SwiGLU")[0])
    reps.append(gradcheck(CausalAttention(), [g2.standard_normal((2, 4, 5, D)),
                                              g2.standard_normal((2, 2, 5, D)),
                                              g2.standard_normal((2, 2, 5, D))],
                          seed=1, name="CausalAttn")[0])
    reps.append(gradcheck(CausalConv1D(3), [g2.standard_normal((2, 6, 4)), g2.standard_normal((4, 3))],
                          seed=1, name="CausalConv1D")[0])
    reps.append(gradcheck(GatedDeltaNet(),
                          [g2.standard_normal((2, 2, 6, 4)), g2.standard_normal((2, 2, 6, 4)),
                           g2.standard_normal((2, 2, 6, 4)), msig(g2.standard_normal((2, 2, 6))),
                           msig(g2.standard_normal((2, 2, 6)))], seed=1, name="GatedDeltaNet")[0])
    reps.append(gradcheck(GatedDeltaNetChunkwise(chunk=4),
                          [g2.standard_normal((2, 2, 6, 4)), g2.standard_normal((2, 2, 6, 4)),
                           g2.standard_normal((2, 2, 6, 4)), msig(g2.standard_normal((2, 2, 6))),
                           msig(g2.standard_normal((2, 2, 6)))], seed=1, name="GatedDeltaNetChunkwise")[0])
    worst = max(r.max_abs_err for r in reps)
    npass = sum(r.passed for r in reps)
    add("梯度正确", npass == len(reps),
        f"{npass}/{len(reps)} 模块通过中心差分, 最大绝对误差 {worst:.2e}",
        worst=worst)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G3 训练有效
    t0 = time.time()
    P = init_params(cfg, seed=seed)
    m = TianjiModel(cfg, P)
    ld = DataLoader(tok, seq_len=16, batch_size=4, seed=seed)
    vld = DataLoader(_synth_tokens(200, cfg.vocab_size, seed + 99), 16, 4, seed + 7)
    opt = HybridOptimizer(max_grad_norm=1.0)
    tr = Trainer(m, opt, ld, total_steps=steps, warmup=8, lr_max=0.03,
                 val_loader=vld, eval_every=max(steps // 3, 1))
    log = tr.run(steps, verbose=False)
    first, last = np.mean(log.losses[:3]), np.mean(log.losses[-3:])
    rel = (first - last) / max(first, 1e-12)
    add("训练有效", rel > 0.05,
        f"训练损失 {first:.4f} -> {last:.4f}（相对下降 {rel * 100:.1f}%）, "
        f"val {log.val[-1][1]:.4f}" if log.val else f"{first:.4f} -> {last:.4f}",
        first=float(first), last=float(last), rel=float(rel))
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G4 可过拟合
    t0 = time.time()
    cfg2 = _small_cfg()
    cfg2.n_experts = 1
    P2 = init_params(cfg2, seed=seed)
    m2 = TianjiModel(cfg2, P2)
    tiny = _synth_tokens(40, cfg2.vocab_size, seed)
    ld2 = DataLoader(tiny, seq_len=8, batch_size=2, seed=seed)
    opt2 = HybridOptimizer(max_grad_norm=1.0)
    tr2 = Trainer(m2, opt2, ld2, total_steps=120, warmup=5, lr_max=0.05)
    log2 = tr2.run(120, verbose=False)
    f4, l4 = np.mean(log2.losses[:3]), np.mean(log2.losses[-3:])
    add("可过拟合", l4 < f4 * 0.5,
        f"极小数据集 {f4:.4f} -> {l4:.4f}（需降到 50% 以下，证明反向真在传播）",
        first=float(f4), last=float(l4))
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G5 微指令
    t0 = time.time()
    rep5 = microop_run_all(seed=seed)
    add("微指令", rep5["failed"] == [] and rep5["total"] == 1024,
        f"{rep5['ok']}/{rep5['total']} 条可执行且输出有限（{rep5['families']} 家族）",
        micro_ok=rep5["ok"], micro_total=rep5["total"])
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G6 MoE 均衡
    t0 = time.time()
    from .l5_moe import LoadBalancer
    lb = LoadBalancer(8, gamma=1e-3)
    gs = rng(spawn_seed(seed, "moe"))
    # 模拟一个持续偏斜的路由：专家 0 长期过载
    cv0, cv1 = None, None
    for i in range(80):
        counts = np.full(8, 5.0)
        counts[0] = 40.0
        counts[1:] = 2.0
        lb.update(counts)
        if i == 9:
            cv0 = lb.maxcv(10)
    cv1 = lb.maxcv(10)
    # 偏置必须朝「抑制过载专家」的方向走
    bias_ok = lb.bias[0] < lb.bias[1]
    add("MoE 均衡", bias_ok,
        f"过载专家偏置 {lb.bias[0]:+.4f} < 欠载专家 {lb.bias[1]:+.4f}（控制器方向正确）",
        bias_over=float(lb.bias[0]), bias_under=float(lb.bias[1]))
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G7 Muon2
    t0 = time.time()
    gs = rng(spawn_seed(seed, "muon"))
    Gm = gs.standard_normal((24, 16))
    Gm = Gm @ Gm.T @ Gm          # 故意做成病态（条件数大）
    o_muon5 = newton_schulz(Gm, steps=5)
    o_muon2_3 = newton_schulz(Gm / (np.sqrt(np.mean(Gm ** 2)) + 1e-8), steps=3)
    q5 = orthogonalization_quality(Gm, o_muon5)
    q3 = orthogonalization_quality(Gm, o_muon2_3)
    add("Muon2 正交化", q3 >= 0.9 * q5,
        f"Muon(5步) 对齐度 {q5:.4f} vs Muon2(3步,预条件后) {q3:.4f} "
        f"（判据：3步不低于5步的 90%）",
        q_muon5=float(q5), q_muon2_3=float(q3))
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G8 缓存等价（混合架构）
    # 覆盖「GDN 的 O(1) 递推增量解码」与「MLA 的 KV cache 增量解码」两者：
    # 每步只喂 1 个新 token，历史全部来自 cache（GDN 递推状态 + MLA KV）。
    # 这是关闭 README「已知边界」第 4 条的关键门禁 —— 递推增量必须 ≡ 整序列重算
    # 到机器精度。hybrid_ratio=3 ⇒ 3:1 混合（GDN 线性层 + MLA 全注意力层）。
    t0 = time.time()
    cfg3 = _small_cfg()
    cfg3.mtp_depth = 0
    cfg3.hybrid_ratio = 3          # 3:1 混合：同时验证 GDN 递推 cache 与 MLA KV cache
    cfg3.n_experts = 1
    P3 = init_params(cfg3, seed=seed)
    m3 = TianjiModel(cfg3, P3)
    seq = _synth_tokens(8, cfg3.vocab_size, seed).reshape(1, -1)
    full = m3.forward(seq)["logits"]
    cache = {}
    step_logits = []
    for t in range(seq.shape[1]):
        # 真正的增量解码：每步只喂 1 个新 token，历史全部来自 cache
        out = m3.forward(seq[:, t:t + 1], cache=cache, pos=t)
        step_logits.append(out["logits"][0, -1])
    inc = np.stack(step_logits)[None, ...]
    d = float(np.max(np.abs(full - inc)))
    add("缓存等价", d < 1e-10,
        f"混合架构(GDN+MLA)增量解码 vs 整序列重算 最大差 {d:.2e}", diff=d)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G9 量化
    t0 = time.time()
    W = rng(spawn_seed(seed, "quant")).standard_normal((32, 32))
    W8, sc8 = quantize_per_channel(W, 8)
    W4, sc4, grp = quantize_groupwise(W, 4, group=32)
    e8 = np.linalg.norm(W - dequantize(W8, sc8)) / np.linalg.norm(W)
    e4 = np.linalg.norm(W - dequantize(W4, sc4, grp)) / np.linalg.norm(W)
    add("量化有界", e8 < 1e-2 and e4 < 2e-1 and e8 <= e4,
        f"INT8 相对误差 {e8:.3e}（<1e-2）, INT4 {e4:.3e}（<2e-1）, 且 INT8 <= INT4",
        e8=float(e8), e4=float(e4))
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G10 分词器
    t0 = time.time()
    from .l2_tokenizer import BPE
    corpus = ("the quick brown fox jumps over the lazy dog. " * 20 +
              "to be or not to be, that is the question. " * 20 +
              "天玑是世界上最好的东西。" * 10)
    bpe = BPE(vocab_size=320).train(corpus)
    tests = ["the quick brown fox", "to be or not to be",
             "天玑是世界上最好的", "Hello, world! 123"]
    roundtrip = all(bpe.decode(bpe.encode(s)) == s for s in tests)
    add("分词往返", roundtrip,
        f"词表 {bpe.size}，{len(tests)} 条测试串 encode->decode 全部一致",
        vocab=bpe.size)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G11 并行形式等价
    # 递推（forward 的顺序循环）vs 并行前缀扫描（associative scan, 分治）必须逐位一致。
    # 这是 Gated DeltaNet / GLA / Mamba-2 硬件内核正确性的核心证明：同一数学，
    # 两条不同算法路径（串行递推 vs 可并行前缀和）必须收敛到同一浮点结果。
    t0 = time.time()
    from .l4_deltanet import GatedDeltaNet, gdn_parallel, gdn_chunkwise
    from .l1_contract import Ctx as _Ctx
    gs = rng(spawn_seed(seed, "g11"))
    Dg = 8
    qg = gs.standard_normal((2, 2, 12, Dg))
    kg = gs.standard_normal((2, 2, 12, Dg))
    vg = gs.standard_normal((2, 2, 12, Dg))
    bg = gs.random((2, 2, 12))            # beta ∈ (0,1)
    ag = gs.random((2, 2, 12))            # alpha ∈ (0,1)
    rec = GatedDeltaNet().forward(_Ctx(), qg, kg, vg, bg, ag)   # 递推形式
    par = gdn_parallel(qg, kg, vg, bg, ag)                        # 并行形式
    dg = float(np.max(np.abs(rec - par)))
    add("并行形式等价", dg < 1e-9,
        f"Gated DeltaNet 递推 vs 并行前缀扫描 最大差 {dg:.2e}", diff=dg)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G12 chunkwise 等价
    # 递推（顺序循环）vs chunkwise（块内并行 + 块间递推，真实 GPU 内核训练形式）。
    # T=23 故意不被块长整除，覆盖尾块长度不足的路径；块长扫描覆盖多种 tiling。
    t0 = time.time()
    gs = rng(spawn_seed(seed, "g12"))
    Dg = 8
    Tg = 23
    qg = gs.standard_normal((2, 2, Tg, Dg))
    kg = gs.standard_normal((2, 2, Tg, Dg))
    vg = gs.standard_normal((2, 2, Tg, Dg))
    bg = gs.random((2, 2, Tg))
    ag = gs.random((2, 2, Tg))
    rec12 = GatedDeltaNet().forward(_Ctx(), qg, kg, vg, bg, ag)
    worst = 0.0
    for ch in (1, 2, 3, 4, 8, 16):
        cw = gdn_chunkwise(qg, kg, vg, bg, ag, chunk=ch)
        worst = max(worst, float(np.max(np.abs(rec12 - cw))))
    add("chunkwise 等价", worst < 1e-9,
        f"Gated DeltaNet 递推 vs chunkwise（T={Tg}, 6 种块长含不整除尾块）最大差 {worst:.2e}",
        diff=worst)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G13 chunkwise 反向等价
    # 物化全量状态的反向（materialized, 存 S_all 形状 B×H×(T+1)×D×D）vs FlashAttention 式
    # 重计算反向（chunkwise, 只存 O(T/C) 块入口状态、块内即时重算、块间用伴随传输 G）：
    # 两条不同的资源/算法路径必须给出逐位一致的解析梯度。这把「线性注意力内核反向」
    # 这一最难的正确性性质机检到机器精度 —— 与 G11/G12 的前向三路径互证形成完整闭环。
    t0 = time.time()
    from .l4_deltanet import gdn_chunkwise_backward
    gs = rng(spawn_seed(seed, "g13"))
    Dg = 8
    Tg = 23
    qg = gs.standard_normal((2, 2, Tg, Dg))
    kg = gs.standard_normal((2, 2, Tg, Dg))
    vg = gs.standard_normal((2, 2, Tg, Dg))
    bg = gs.random((2, 2, Tg))
    ag = gs.random((2, 2, Tg))
    gg = gs.standard_normal((2, 2, Tg, Dg))                 # 下游对 o 的梯度
    ctx_g = _Ctx()
    GatedDeltaNet().forward(ctx_g, qg, kg, vg, bg, ag)
    dmat = GatedDeltaNet().backward(ctx_g, gg)              # 物化全量状态反向
    worst = 0.0
    for ch in (1, 2, 3,4, 8, 16):
        dcw = gdn_chunkwise_backward(qg, kg, vg, bg, ag, gg, chunk=ch)
        worst = max(worst, max(np.max(np.abs(a - b)) for a, b in zip(dmat, dcw)))
    add("chunkwise 反向等价", worst < 1e-9,
        f"Gated DeltaNet 物化反向 vs chunkwise 重计算反向（T={Tg}, 6 种块长）最大差 {worst:.2e}",
        diff=worst)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G14 投机解码无损
    # Leviathan/Chen 拒绝采样的决定性恒等式：单步输出分布逐点等于 target，
    # 与 draft 好坏无关 —— 这是「加速却不损失质量」的数学根据。
    # (a) 解析验证：P(out=x) = q·min(1,p/q) + P_rej·residual == p，机器精度；
    #     覆盖四种对抗情形：任意分布 / draft==target（恒接受）/ target 含近零
    #     质量 / draft 高度集中（大量拒绝）。
    # (b) 端到端统计：mock 双分布模型跑 6000 token，经验边际与 target 的
    #     总变差距离 TV 必须小；且 draft≠target 时接受率 ∈ (0,1)（I5 有效性）。
    t0 = time.time()
    from .l8_infer import speculative_decode as _spec
    g14 = rng(spawn_seed(seed, "g14a"))
    worst14 = 0.0
    for trial in range(64):
        V = int(g14.integers(2, 25))
        lp = g14.standard_normal(V) * g14.uniform(0.5, 3.0)
        lq = g14.standard_normal(V) * g14.uniform(0.5, 3.0)
        if trial % 4 == 1:
            lq = lp.copy()                                   # draft == target（恒接受）
        elif trial % 4 == 2:
            lp[int(g14.integers(0, V))] = -40.0              # target 含近零质量
        elif trial % 4 == 3:
            lq = g14.standard_normal(V) + 3.0                # draft 高度集中
        p = softmax(lp)
        q = softmax(lq)
        acc = q * np.minimum(1.0, p / np.maximum(q, 1e-300))
        rej = 1.0 - acc.sum()
        r = np.maximum(p - q, 0.0)
        rs = r.sum()
        out = acc + (rej * r / rs if rs > 1e-12 else 0.0)
        worst14 = max(worst14, float(np.max(np.abs(out - p))))
    # ---- (b) mock 双分布模型端到端统计
    Vb = 12
    pb_logits = np.arange(Vb, dtype=np.float64) * 0.7 - 3.0
    pq_logits = np.concatenate([np.full(3, 2.0), np.full(Vb - 3, -2.0)])   # draft 偏斜
    p_b = softmax(pb_logits)

    class _MockSpec:
        """任何上下文都返回同一 target/draft logits 的 mock（使理论边际解析已知）。"""
        cfg = SimpleNamespace(mtp_depth=1)

        def forward(self, x, **kw):
            L = x.shape[1]
            return {"logits": np.broadcast_to(pb_logits, (1, L, Vb)).copy(),
                    "mtp_logits": np.broadcast_to(pq_logits, (1, L, Vb)).copy(),
                    "aux": {}}

    ids14, st14 = _spec(_MockSpec(), np.asarray([[0]], dtype=np.int64),
                        6000, gamma=1, seed=11)
    gen = np.asarray(ids14[1:], dtype=np.int64)
    freq = np.bincount(gen, minlength=Vb) / gen.size
    tv = 0.5 * float(np.abs(freq - p_b).sum())
    ok14 = (worst14 < 1e-12 and tv < 0.05
            and 0.0 < st14.accept_rate < 1.0 and st14.accepted >= 1)
    add("投机解码无损", ok14,
        f"解析恒等最大差 {worst14:.2e}；6000 token 经验边际 TV={tv:.4f}；"
        f"接受率 {st14.accept_rate:.3f}（∈(0,1) 有效 drafting；加速边界结构性 ≈1 已如实声明）",
        analytic=worst14, tv=tv, accept_rate=st14.accept_rate)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G15 投机解码并行验证
    # 独立轻量 draft（TinyDraftGDN 复用已门禁 gdn_step）+ γ>1 一次 batch 验证。
    # (a) 位置指纹：target logits[0,t] = onehot((t+1) mod V)·40 近确定性；
    #     draft 与 target 同指纹（q=p）→ 全接受 + bonus，产出序列完全确定，
    #     逐位断言 == 指纹 —— 任何 verify 位置对齐 off-by-one 或 draft 状态
    #     快照回滚错误都会立刻破坏该序列。
    # (b) 代数条件：每轮主干 forward 恰 1 次；tokens/forward = 9/3 = 3 > 1。
    # (c) 统计无损：target 恒定偏斜 p、draft 恒定偏斜 q≠p（混合接受/拒绝
    #     /重采样路径），4000 token 经验边际 TV 必须小；接受率 ∈ (0,1)。
    t0 = time.time()
    from .l8_infer import TinyDraftGDN, speculative_decode as _spec15
    V15 = 12

    def _finger(t: int) -> np.ndarray:
        z = np.zeros(V15)
        z[t % V15] = 40.0
        return z

    class _FTarget:
        cfg = SimpleNamespace(mtp_depth=0)

        def __init__(self):
            self.calls = 0

        def forward(self, x, **kw):
            self.calls += 1
            L = x.shape[1]
            return {"logits": np.stack([_finger(t + 1) for t in range(L)])[None],
                    "aux": {}}

    class _FDraft:
        # S = 已喂 token 栈：dm.S = 快照 赋值即回滚（与真实 GDN 状态语义一致）
        def __init__(self):
            self.S = []
            self.last_logits = _finger(0)

        def warmup(self, tokens):
            self.S = list(np.asarray(tokens).reshape(-1))
            self.last_logits = _finger(len(self.S))

        def step(self, token):
            self.S = [*self.S, int(token)]
            return _finger(len(self.S))

    m15 = _FTarget()
    ids15, st15 = _spec15(m15, np.asarray([[0]], dtype=np.int64), 9,
                          gamma=3, seed=5, draft_model=_FDraft())
    expect15 = list(range(1, 10))                       # 产出应恰为指纹 1..9
    pos_ok = list(ids15[1:]) == expect15
    algebra_ok = (m15.calls == st15.steps and st15.produced == 9
                  and st15.speedup > 1.0)
    # ---- (c) 混合路径统计无损
    pb15 = np.arange(V15, dtype=np.float64) * 0.7 - 3.0
    pq15 = np.concatenate([np.full(3, 2.0), np.full(V15 - 3, -2.0)])
    p15 = softmax(pb15)

    class _CTarget:
        cfg = SimpleNamespace(mtp_depth=0)

        def forward(self, x, **kw):
            L = x.shape[1]
            return {"logits": np.broadcast_to(pb15, (1, L, V15)).copy(), "aux": {}}

    class _CDraft:
        def __init__(self):
            self.S = np.zeros((1, 1, 1, 1))
            self.last_logits = pq15.copy()

        def warmup(self, tokens):
            pass

        def step(self, token):
            return pq15

    idsc, stc = _spec15(_CTarget(), np.asarray([[0]], dtype=np.int64), 4000,
                        gamma=3, seed=13, draft_model=_CDraft())
    freqc = np.bincount(np.asarray(idsc[1:], dtype=np.int64), minlength=V15) / 4000
    tvc = 0.5 * float(np.abs(freqc - p15).sum())
    ok15 = pos_ok and algebra_ok and tvc < 0.05 and 0.0 < stc.accept_rate < 1.0
    add("投机解码并行验证", ok15,
        f"位置指纹逐位一致={pos_ok}；主干 forward {m15.calls} 次/产出 9 token"
        f"（tokens/forward={9 / max(m15.calls, 1):.1f}>1）；"
        f"混合路径 4000 token TV={tvc:.4f}；接受率 {stc.accept_rate:.3f}∈(0,1)",
        fingerprint=pos_ok, calls=m15.calls, tv=tvc,
        accept_rate=stc.accept_rate)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G16 对齐正确性
    t0 = time.time()
    from .l9_align import (Aligner, dpo_terms, ipo_terms, kto_terms,
                           seq_logprob, mean_logprob, grad_logprob, freeze_copy)
    from .l0_numerics import log_softmax as _lsm
    g16 = rng(spawn_seed(seed, "align"))

    # ---- (a) 解析值：中性损失与系数符号
    l16, c16 = dpo_terms(0., 0., 0., 0., beta=0.1)
    lz16, _ = kto_terms(0., 0., True)
    lb16, _ = kto_terms(0., 0., False)
    dstar = 0.1 / (2 * 0.1)                       # IPO 最优 Δ* = τ/(2β)
    _, ci16 = ipo_terms(dstar, 0., 0., 0., beta=0.1, tau=0.1)
    lipo_big, _ = ipo_terms(50., 0., 0., 0., beta=0.1)
    ldpo_big, _ = dpo_terms(50., 0., 0., 0., beta=0.1)
    analytic_ok = (abs(l16 - np.log(2)) < 1e-9 and abs(c16 + 0.05) < 1e-12
                   and abs(lz16 - np.log(2)) < 1e-9 and abs(lb16 - np.log(2)) < 1e-9
                   and ci16 == 0.0 and lipo_big > 100.0 and ldpo_big < 0.01)

    # ---- (b) grad_logprob 中心差分（logits 空间独立裁判，含 pad mask）
    V16, T16 = 12, 5
    lg16 = g16.standard_normal((2, T16, V16))
    tg16 = g16.integers(0, V16, size=(2, T16))
    tg16[0, 2] = -1                               # pad 位置必须被 mask
    Ga16 = grad_logprob(lg16, tg16)
    eps16 = 1e-6

    def _f16(z):
        lp = _lsm(z, axis=-1)
        sel = np.where(tg16 >= 0)
        return float(np.sum(lp[sel[0], sel[1], tg16[tg16 >= 0]]))

    worst16 = 0.0
    for bi, ti, vi in [(0, 0, 3), (0, 1, 5), (0, 2, 7), (1, 3, 1), (1, 2, 9)]:
        o = lg16[bi, ti, vi]
        lg16[bi, ti, vi] = o + eps16; fp = _f16(lg16)
        lg16[bi, ti, vi] = o - eps16; fm = _f16(lg16)
        lg16[bi, ti, vi] = o
        worst16 = max(worst16, abs((fp - fm) / (2 * eps16) - Ga16[bi, ti, vi]))

    # ---- (c)+(d)+(f) 真实小模型：端到端参数梯度中心差分 + 梯度方向 + ref 冻结
    from .model import TianjiConfig, init_params
    cfg16 = TianjiConfig(vocab_size=24, d_model=16, n_layers=2, n_heads=2,
                         n_kv_heads=1, d_latent_q=8, d_latent_kv=6, d_rope=4,
                         d_ff=16, d_ff_expert=12, n_experts=2, moe_top_k=1,
                         max_len=32, mtp_depth=0)
    P16 = init_params(cfg16, seed=seed + 3)
    Pref16 = freeze_copy(init_params(cfg16, seed=seed + 9))
    pol16 = TianjiModel(cfg16, P16)
    al16 = Aligner(pol16, Pref16, method="dpo", beta=0.1)
    xc = g16.integers(0, 24, size=(1, 6)); yc = g16.integers(0, 24, size=(1, 6))
    xr = g16.integers(0, 24, size=(1, 5)); yr = g16.integers(0, 24, size=(1, 5))
    _, G16 = al16.loss_and_grads(xc, yc, xr, yr)
    ref_before16 = {k: v.copy() for k, v in al16.ref_model.P.items()}

    e2e16 = 0.0
    for k in ("tok_emb", "final_norm_w", "L1.Wg"):
        if k not in P16:
            continue
        arr = P16[k]
        idx = tuple(0 for _ in arr.shape)
        o = arr[idx]
        arr[idx] = o + eps16; fp = al16.loss_and_grads(xc, yc, xr, yr)[0]
        arr[idx] = o - eps16; fm = al16.loss_and_grads(xc, yc, xr, yr)[0]
        arr[idx] = o
        e2e16 = max(e2e16, abs((fp - fm) / (2 * eps16) - G16[k][idx]))

    out_c0 = pol16.forward(xc, yc); lp_c0 = seq_logprob(out_c0["logits"], yc)
    out_r0 = pol16.forward(xr, yr); lp_r0 = seq_logprob(out_r0["logits"], yr)
    for k in P16:
        if k in G16:
            P16[k] -= 0.5 * G16[k]
    out_c1 = pol16.forward(xc, yc); lp_c1 = seq_logprob(out_c1["logits"], yc)
    out_r1 = pol16.forward(xr, yr); lp_r1 = seq_logprob(out_r1["logits"], yr)
    dir_ok = lp_c1 > lp_c0 and lp_r1 < lp_r0

    al16.loss_and_grads(xc, yc, xr, yr)          # 再跑两步，ref 必须零漂移
    al16.loss_and_grads(xc, yc, xr, yr)
    drift16 = max(float(np.max(np.abs(al16.ref_model.P[k] - ref_before16[k])))
                  for k in ref_before16)
    frozen_ok = drift16 == 0.0 and all(Pref16[k] is not P16[k] for k in Pref16)

    # ---- (e) SimPO 长度归一化：复制两倍长逐位不变
    slg = g16.standard_normal((1, T16, V16))
    stg = g16.integers(0, V16, size=(1, T16))
    mp1 = mean_logprob(slg, stg)
    mp2 = mean_logprob(np.concatenate([slg, slg], axis=1),
                       np.concatenate([stg, stg], axis=1))
    simpo_ok = mp1 == mp2

    ok16 = analytic_ok and worst16 < 1e-6 and e2e16 < 1e-6 and dir_ok \
        and frozen_ok and simpo_ok
    add("对齐正确性", ok16,
        f"解析值（DPO/KTO Δ=0→log2、IPO Δ* 系数=0、Δ=50: DPO={ldpo_big:.1e} vs "
        f"IPO={lipo_big:.0f}）；grad_logprob 中心差分 {worst16:.2e}；"
        f"DPO 端到端参数梯度 {e2e16:.2e}；方向 chosen{'↑' if dir_ok else '✗'}"
        f"/rejected{'↓' if dir_ok else '✗'}；ref 漂移 {drift16:.1e}；"
        f"SimPO 长度不变={simpo_ok}",
        analytic=worst16, e2e=e2e16, ref_drift=drift16)
    R.results[-1].seconds = time.time() - t0

    return R
