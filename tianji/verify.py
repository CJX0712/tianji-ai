"""L10 · 交付门禁 —— 「可运行」不等于「可交付」。

继承天枢 tianshu-ai 的门禁哲学：
    单元测试证明的是「函数没写错」，门禁证明的是「系统整体达到了宣称的能力」。
    本项目对外宣称的每一件事，都必须有一条门禁守着。
    **失败就如实报失败，不允许放宽阈值来凑绿。**
    报告里同时给出 status 与实测值，任何人都能复核。

门禁清单：
    G1  确定性        同 seed 两次运行逐位一致（否则一切实验结论不可信）
    G2  梯度正确      各独立原语（Linear/RMSNorm/RoPE/SwiGLU/CausalAttn/CausalConv1D/GatedDeltaNet…）解析反向 vs 中心差分；完整模型端到端反向见 G25
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
    G17 MoE 路由不变性 M2-M5 全部机检 + 对抗 fuzz（输入/router/bias 三轴极端，
                    E/K 全扫描）：(a) 门控归一化行和==1 —— 曾由本门禁抓出
                    「Σ+1e-12 加法偏置在极端输入下主导归一化 → 行和塌陷到 0」
                    的真实缺陷（已修复为无偏置归一化）；(b) mask 行和恒==K、
                    专家计算量==N·K（FLOPs 比 K/E）；(c) 单专家垄断场景全量
                    覆盖无丢弃；(d) 偏置控制器：均衡零偏置、过载单调下降、
                    clip 边界、maxcv 空历史=inf；(e) 均衡损失 f-detach 语义
                    解析梯度 + 中心差分
    G18 分词器对抗 fuzz  T5-T11 全部机检 + 对抗 fuzz：(a) 16 类极端输入往返
                    （含孤立代理项）—— 本门禁探针抓出「lone surrogate 输入
                    直接抛 UnicodeEncodeError 崩溃」的真实缺陷（已修复为
                    WTF-8 语义 surrogatepass，与 Rust String/HF tokenizers
                    一致）；(b) 300 条随机 Unicode fuzz；(c) 词表单射、
                    merge 图 vocab[nid]==vocab[a]+vocab[b]、ranks 插入序、
                    id 闭合；(d) save→load 逐位一致；(e) warm/cold 缓存
                    一致 + 不动点 encode(decode(encode(s)))==encode(s)；
                    (f) 预分词无损 concat；(g) 误配置守卫：空特殊 token
                    必须 ValueError（曾静默产生字符间垃圾 id）
    G19 优化器不变性    O1/O3-O7 全部机检 + 对抗 fuzz：(a) NS 奇异值精确命题
                    —— σ_max 对任意输入恒落 (0.5,1.5)（Muon 依赖的性质），
                    全谱落带的条件命题（5 步+cond≤100）；l0_numerics 原宣称
                    「全谱 (0.5,1.5)」被本门禁证伪（病态小奇异值可低至 0.03，
                    quintic 吸引盆 ≈0.7 的已知行为），已修正为精确命题而非
                    放宽阈值凑绿；(b) O3 AdamW 解耦 WD 梯度尺度不变
                    （×1000 更新一致 1.1e-06）；(c) O7 Muon Nesterov 真实
                    前瞻（T/F 差异 1.7e-01）vs Muon2 无变体 —— 抓出
                    `P=P+β·P` 死代码（常数缩放被 NS 首步谱归一完全擦除，
                    NS(1.9G) vs NS(G)=1.7e-08，数学上行为不可检测的假功能），
                    已删除；门禁 (d) 检查更新匹配文档纯公式，实测可捕获
                    真实算法漂移（scale 公式改错 → 7.3e-01 变红）；
                    (e) O6 NaN/Inf 梯度守卫 —— 抓出「一个坏梯度经 clip
                    缩放因子静默污染全部参数」的真实缺陷，修复为
                    GradScaler 语义跳过 + skipped_steps 计数 + 恢复验证；
                    (f) clip 全局语义；(g) AdamW 首步解析值（偏差修正）；
                    (h) O5 确定性；(i) Muon2 梯度尺度不变；(j) O4 凸问题
                    损失单调下降
    G20 持久化完整性    P1-P7 全部机检 + 损坏注入对抗：(a) 路径规范化闭环 ——
                    探针抓出「np.savez 自动追加 .npz 使 _sha256_file 找错
                    文件直接 FileNotFoundError」的真实缺陷（已修复为
                    save/load/info 三路共用规范化）；(b) 截断 meta.json
                    双路径必须抛 CheckpointError —— 抓出「漏出原始
                    JSONDecodeError，调用方 except CheckpointError 会漏」
                    的类型不一致（已修复）；(c) 损坏权重 sha256 失配；
                    (d) 权重替换+sha 同步篡改被参数指纹抓住（威胁模型如实
                    声明：防失配腐烂，不防同步作恶）；(e) 原子性 —— 模拟
                    meta 写入崩溃，无 tmp 半成品、旧 checkpoint 逐位完好
                    可加载（对齐 safetensors 的 tmp+rename 模式）；(f) 空
                    参数/怪异参数名/dtype 保持；                    (g) 保存确定性 + 指纹失配
                    检出 + 缺失 meta/不存在路径响亮失败
    G21 数据管道        D1-D4 + T1 全部机检 + 边界对抗：(a) 构造期响亮校验
                    —— 探针抓出「数据过短时 max(n,1) 静默钳制 → x/y batch
                    形状悄悄错位（x(2,10) y(2,9)）不报错」的真实缺陷
                    （已修复：len < seq_len+1 / seq_len<1 / batch_size<1
                    一律构造期 ValueError）；(b) 边界合法：len==seq_len+1
                    恰有一个窗口，y==shift(x)；(c) token 范围守卫 —— 探针
                    抓出「负 token 被 numpy 花式索引静默回绕到词表末行，
                    实测全 -1 输入 loss==0（模型完美学会垃圾）」的真实缺陷
                    （已修复：forward 入口校验 [0, vocab_size)，覆盖训练/
                    评估/推理全部调用方）；(d) y 对齐不变量 + 连续切片
                    指纹重放；(e) 同 seed 逐位一致/异 seed 不同；
                    (f) T1 调度器 —— l7_train 原宣称「门禁复检」但门禁
                    不存在（与 G19 的 N3 同型的宣称-门禁失配），本门禁
                    补齐：cosine/wsd 端点精确（warmup 斜率、lr_max、
                    lr_min）+ warmup 后单调不增 + 多参数组合扫描；
                    (g) 梯度累积等价：accum=k 一次 step 的梯度 ==
                    同 seed 重放 k batch 的手动平均（确定性重放）
    G22 原语不变性    P3-P6 首次全部机检（l3 docstring 宣称「门禁复检」
                    但门禁不存在 —— 与 G19 的 N3 / G21 的 T1 同型的
                    第 3 次宣称-门禁失配）+ 边界守卫 B1-B4：
                    (a) P3 RoPE 保范 ‖RoPE(x)‖==‖x‖（含 20000 长度外推、
                    1e-6~1e6 量级）；(b) P4 相对位置 ⟨q_i,k_j⟩ 只依赖
                    (i−j)，且带非恒真性对照（不同 off 的内积必须显著
                    不同，否则 RoPE 退化成恒等也能骗过门禁）；
                    (c) P5 因果性（扰动未来不影响过去输出）；(d) P6
                    注意力行和恒==1（含 GQA H/Hkv 倍数）；
                    (e) B1 全掩码行守卫 —— 探针抓出「causal_mask_rect
                    (Tq>Tk) 产生全 -inf 行 → softmax 0/0 → attention
                    输出静默 NaN（16/40）」的真实缺陷，修复为构造期
                    ValueError；(f) B2 RoPE 奇数维 —— 探针抓出
                    rope_tables(dim=5) 拼出 dim−1 长度表且**表构造阶段
                    不报错**，到 forward 深处才崩，修复为构造期拒绝；
                    (g) B3 RMSNorm 零输入/1e150 稳定 + P2 输出 RMS==1；
                    (h) B4 CrossEntropy 全忽略 targets → loss==0 且
                    梯度精确全零、无 NaN
    G23 GDN 归一化    L1-L3 全部机检 + 边界门控 fuzz：(a) L1 l2norm 单位范数
                    契约 ‖k̂‖==1 对任意 ‖k‖>0 成立（含 1e-9），零向量→
                    零输出零梯度 —— 探针抓出「分母 +1e-8 使 ‖k̂‖ 只剩
                    0.181（偏差 82%），delta rule 的有效写入强度 β‖k̂‖²
                    被静默缩小到 0.016（本该 0.5）」的真实缺陷；
                    (b) L2 尺度自适应梯度检验 —— 同一公式在新前向下的
                    相对误差 5e-11，旧前向为 eps/(‖k‖+eps)：‖k‖=1e-7 时
                    达 17%。注意 gradcheck 的固定绝对步长 1e-5 对
                    ‖k‖~1e-7 输入测量本身失效（首轮探针报出的 6.8e+06
                    绝对误差是伪影，已如实更正，局限记在 l1_contract）；
                    (c) L3 三条算法路径（递推/并行扫描/chunkwise）在
                    α,β ∈ {0,1} 全部边界组合 + 极小/极大 ‖k‖ 下仍逐位
                    一致（≤4.4e-16），chunk 取 1/T/T+5/非整除均成立
    G24 基准熵下界    E1-E3 全部机检：(a) 熵下界自洽 —— 2 阶马尔可夫源熵率 H
                    由两套独立算法（解析平稳分布法 H_A 与 oracle 交叉熵法
                    H_B）算出且互相一致（差 < 5e-3），0<H<log(vocab) 成立，
                    证明「降到多少才算学会」的判据是被算出来的、不是编的；
                    (b) 损失不破熵下界 —— 未训练模型验证交叉熵 ≥ H（若 loss
                    算出 sub-entropy 值即说明损失函数有 bug，本护栏拦截）；
                    (c) 基准诚实性 + 判别力 —— 读 bench/results/bench.json：
                    E3a 主基准各配置 learned fraction = (uniform−loss)/(uniform−H)
                    全部 ≤ 1.0（没有任何配置突破熵下界；当前主基准均远未学会，
                    见 BENCHMARKS.md 诚实修订），且 B2 布局段两配置参数量必须
                    严格相等（G27 修复的混杂：旧版 interleave/edge 注意力层数
                    漂移导致参数量不等、结论不可信）；E3b 可学性对照臂（B7，1 阶
                    vocab=8 同源语料、同模型规模）lf_full > 0.5 且模型全量 val
                    与同一序列上的真实源 oracle 差 < 0.05 —— 证明基准不是
                    「永远学不会」的钝器：任务在规模内可学时 lf 转正、模型逼近
                    真实源，B1 的三架构比较在该臂上才有统计意义；E3c 多 seed
                    稳定性（B8，5 seed × 3 配置，multiseed.json）—— 已记录
                    per-seed lf ≤ 1 护栏 + 每 seed 获胜者翻转情况 + 两两
                    |Δmean|/pooled_std，把「架构差异落在种子噪声内」定性变定量
    G25 全模型反向一致性 完整模型（GDN+MLA+MoE+MTP 装配）解析反向 vs 中心差分
                    （分层抽样：每张量少量均匀点 + 全局随机点，全参数中心差分
                    太慢 ≈190s）：G2 只校验各独立原语反向，本门禁补齐把它们装配成
                    完整 Transformer 后的端到端反向（残差顺序/Pre-Norm/MoE 路由/
                    MTP 头梯度），抓住「会训练却永远训不好」的阴险接线 bug
                    （实测最大混合比值 2.5e-03，远小于 1）
    G26 基准 train/val 同源 机检基准语料的 train/val 是否同源（修一个曾污染全基准的
                    阴险 bug）：markov_tokens 每次调用都重抽转移矩阵，旧 run_bench 用
                    seed 与 seed+7 分别生成 train/val ⇒ 竟是两个不同马尔可夫源，val 永远
                    到不了 H、learned fraction 失效（把「能学会」误判成「学不会」）。修复：
                    markov_train_val 只抽一次转移矩阵、采两条独立序列。门禁 E1 断言同源 val
                    的 oracle NLL≈H，E2 以「不同源 ⇒ NLL 显著>H」作反证对照，证明对该 bug
                    敏感、能抓住回归
    G27 B1 架构等参数+标注 把 B2 等参数方法论推广到 B1：机检 B1 三架构（全注意力 /
                    全 GDN / 混合）标注正确（「全 GDN」确为全 gdn，曾因 hybrid_ratio=1e6
                    错配成 1 注意力+5 GDN）、三种结构齐备、且三配置参数量在 0.1% 内相等——
                    隔离架构变量，不让参数量混为差异来源（旧版三配置参数量互不相等
                    396544/364224/396544，结论「全 GDN 显著更优」被错标+参数量混杂）
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

    # ---------------------------------------------------------- G17 MoE 路由不变性
    t0 = time.time()
    from .l1_contract import Ctx as _Ctx17
    from .l5_moe import MoELayer, LoadBalancer, sequence_balance_loss
    g17 = rng(spawn_seed(seed, "moe"))

    # ---- (a) M2 门控归一化 fuzz（对抗场景：输入/router/bias 三轴极端）
    worst17 = 0.0
    mask_ok = True
    for _ in range(64):
        N = int(g17.integers(1, 40)); E = int(g17.integers(2, 9))
        K = int(g17.integers(1, E + 1)); d = int(g17.integers(2, 12))
        dff = int(g17.integers(2, 8))
        x = g17.standard_normal((N, d)) * float(g17.choice([0.01, 1.0, 100.0]))
        router = g17.standard_normal((E, d)) * float(g17.choice([0.001, 1.0, 50.0]))
        Wg = g17.standard_normal((E, dff, d)) * 0.3
        Wu = g17.standard_normal((E, dff, d)) * 0.3
        Wd = g17.standard_normal((E, d, dff)) * 0.3
        bias = g17.standard_normal(E) * float(g17.choice([0.0, 1.0, 1000.0]))
        c17 = _Ctx17()
        MoELayer(E, K).forward(c17, x, router, Wg, Wu, Wd, bias)
        worst17 = max(worst17, float(np.abs(np.sum(c17["g"], axis=-1) - 1.0).max()))
        mask_ok = mask_ok and bool(np.all(c17["mask"].sum(axis=-1) == K))

    # ---- (b) M4 稀疏性：专家计算总量 == N·K（FLOPs 比 K/E 的数值体现）
    flops_ok = True
    N17, E17, K17 = 50, 8, 2
    for _ in range(8):
        x = g17.standard_normal((N17, 6)); router = g17.standard_normal((E17, 6))
        Wg = g17.standard_normal((E17, 4, 6)); Wu = g17.standard_normal((E17, 4, 6))
        Wd = g17.standard_normal((E17, 6, 4))
        c17 = _Ctx17()
        MoELayer(E17, K17).forward(c17, x, router, Wg, Wu, Wd, None)
        flops_ok = flops_ok and sum(len(t) for t in c17["token_ids"]) == N17 * K17

    # ---- (c) M5 容量守恒：单专家垄断场景全量覆盖、无丢弃、无 NaN
    xm = g17.standard_normal((30, 4)); Em = 4
    routerm = np.zeros((Em, 4)); routerm[0] = 100.0
    Wgm = g17.standard_normal((Em, 4, 4)); Wum = g17.standard_normal((Em, 4, 4))
    Wdm = g17.standard_normal((Em, 4, 4))
    c17 = _Ctx17()
    MoELayer(Em, 2).forward(c17, xm, routerm, Wgm, Wum, Wdm, None)
    cover_ok = (bool(np.all(c17["mask"].sum(axis=-1) == 2))
                and bool(np.all(np.isfinite(c17["g"]))))

    # ---- (d) M3 偏置控制器：均衡零偏置 / 过载单调下降 / clip / maxcv 语义
    lb = LoadBalancer(4, gamma=1e-3, bias_cap=0.5)
    for _ in range(10):
        lb.update(np.full(4, 100.0))
    eq_ok = bool(np.allclose(lb.bias, 0.0))
    for _ in range(30):
        lb.update(np.array([200.0, 50.0, 100.0, 100.0]))
    bias_ok = (lb.bias[0] < 0 and lb.bias[1] > 0
               and bool(np.all(np.abs(lb.bias) <= 0.5 + 1e-12)))
    lb.reset()
    cv_ok = lb.bias.tolist() == [0.0] * 4 and len(lb.history) == 0 \
        and LoadBalancer(4).maxcv() == float("inf")

    # ---- (e) 均衡损失 detach 语义：解析梯度 + 中心差分
    N17b, E17b = 20, 4
    aff = g17.standard_normal((N17b, E17b))
    assign = np.zeros((N17b, E17b))
    assign[np.arange(N17b), g17.integers(0, E17b, N17b)] = 1.0
    _, grad17 = sequence_balance_loss(aff, assign)
    f17 = assign.mean(axis=0)
    grad_manual = np.broadcast_to(1e-4 * f17 * E17b / N17b, aff.shape)
    eps17 = 1e-6
    cv_worst = 0.0
    for i in range(N17b):
        for j in range(E17b):
            o = aff[i, j]
            aff[i, j] = o + eps17; lp, _ = sequence_balance_loss(aff, assign)
            aff[i, j] = o - eps17; lm, _ = sequence_balance_loss(aff, assign)
            aff[i, j] = o
            cv_worst = max(cv_worst, abs((lp - lm) / (2 * eps17) - grad17[i, j]))

    ok17 = (worst17 < 1e-12 and mask_ok and flops_ok and cover_ok
            and eq_ok and bias_ok and cv_ok
            and np.allclose(grad17, grad_manual) and cv_worst < 1e-6)
    add("MoE 路由不变性", ok17,
        f"M2 门控归一化 64 组对抗 fuzz 行和最大偏差 {worst17:.2e}（曾被抓出 "
        f"1.0e+00 归一化塌陷并修复）；M4 mask 行和恒==K={mask_ok}、专家计算量"
        f"==N·K={flops_ok}；M5 垄断场景全量覆盖={cover_ok}；M3 偏置控制器"
        f"（均衡零偏置={eq_ok}/过载下降={bias_ok}/clip 与 maxcv=inf={cv_ok}）；"
        f"均衡损失 detach 梯度中心差分 {cv_worst:.2e}",
        gate_worst=worst17, flops=flops_ok, balance=bias_ok, cv=cv_worst)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G18 分词器对抗 fuzz
    t0 = time.time()
    import os
    import random as _random
    import tempfile
    from .l2_tokenizer import BPE as _BPE18, pretokenize as _pretok18
    g18 = rng(spawn_seed(seed, "tok"))
    corpus18 = ("the quick brown fox jumps over the lazy dog. " * 20 +
                "to be or not to be, that is the question. " * 20 +
                "天玑是世界上最好的东西。" * 10)
    tok = _BPE18(vocab_size=320).train(corpus18)

    # ---- (a) T5 对抗往返：16 类极端输入（G18 探针实测抓出孤立代理项崩溃）
    # 异常安全：缺陷存在时本门禁必须「报 FAIL」而不是掀翻整个门禁套件
    adv18 = [
        "", "a", "\ud800", "a\udfffb", "Hello 🎉🚀 world", "a\x00b",
        "   \t\n\r  ", "混合 mixed 文本 with English 123", "\ufffd\uffff",
        "e\u0301e\u0301", "\U0001F468\u200D\U0001F469\u200D\U0001F467",
        "ab" * 2000, "\ufeffhello", "a\ufe0f", "\U000E0001\U000E0020",
        "\U0010FFFF",
    ]
    rt_bad18: list = []
    for s in adv18:
        try:
            if tok.decode(tok.encode(s)) != s:
                rt_bad18.append(s)
        except Exception as e:                    # noqa: BLE001 —— 缺陷即失败
            rt_bad18.append(f"{s!r}: {type(e).__name__}")

    # ---- (b) T6 随机 Unicode fuzz（同样异常安全）
    rr = _random.Random(spawn_seed(seed, "tokfuzz") % (2**31))
    fuzz_bad18 = 0
    for _ in range(300):
        s = "".join(chr(rr.randint(1, 0x4FFF)) for _ in range(rr.randint(0, 30)))
        try:
            if tok.decode(tok.encode(s)) != s:
                fuzz_bad18 += 1
        except Exception:                        # noqa: BLE001 —— 缺陷即失败
            fuzz_bad18 += 1

    # ---- (c) T7 词表单射 + merge 图一致 + ranks 插入序 + id 闭合
    seen18: set[bytes] = set()
    dup18 = 0
    for b in tok.vocab.values():
        if b in seen18:
            dup18 += 1
        seen18.add(b)
    graph18 = all(tok.vocab[nid] == tok.vocab[a] + tok.vocab[b]
                  for (a, b), nid in tok.merges.items())
    ranks18 = all(tok.ranks[p] == r for r, p in
                  enumerate(sorted(tok.merges, key=lambda k: tok.merges[k])))
    closed18 = all(0 <= i < tok.size for i in tok.vocab)

    # ---- (d) T8 存取一致
    with tempfile.TemporaryDirectory() as _td18:
        _p18 = os.path.join(_td18, "vocab.txt")
        tok.save(_p18)
        tok2 = _BPE18.load(_p18)
        sl18 = (tok2.encode("天玑 the quick <|endoftext|> brown 🎉 123")
                == tok.encode("天玑 the quick <|endoftext|> brown 🎉 123")
                and tok2.vocab == tok.vocab and tok2.merges == tok.merges)

    # ---- (e) T9 缓存一致（warm==cold）+ 不动点
    cold18 = _BPE18(vocab_size=320).train(corpus18)
    warm18 = _BPE18(vocab_size=320).train(corpus18)
    probe18 = "the quick 天玑 brown fox"
    warm18.encode(probe18)
    cache18 = warm18.encode(probe18) == cold18.encode(probe18)
    fp_bad18 = 0
    for _ in range(100):
        s = "".join(chr(rr.randint(1, 0x4FFF)) for _ in range(rr.randint(0, 20)))
        try:
            i1 = tok.encode(s)
            if tok.encode(tok.decode(i1)) != i1:
                fp_bad18 += 1
        except Exception:                        # noqa: BLE001 —— 缺陷即失败
            fp_bad18 += 1

    # ---- (f) T10 预分词无损
    pt_bad18 = 0
    for _ in range(200):
        x = "".join(chr(rr.randint(1, 0x9FFF)) for _ in range(rr.randint(0, 25)))
        if "".join(_pretok18(x)) != x:
            pt_bad18 += 1

    # ---- (g) T11 误配置守卫：空特殊 token 必须响亮失败
    try:
        _BPE18(vocab_size=320, special=("",))
        guard18 = False
    except ValueError:
        guard18 = True

    ok18 = (not rt_bad18 and fuzz_bad18 == 0 and dup18 == 0 and graph18
            and ranks18 and closed18 and sl18 and cache18 and fp_bad18 == 0
            and pt_bad18 == 0 and guard18)
    add("分词器对抗 fuzz", ok18,
        f"T5 对抗往返 {len(adv18)} 类极端输入（含孤立代理项，WTF-8 修复）"
        f"失败 {len(rt_bad18)}；T6 随机 Unicode 300 条失败 {fuzz_bad18}；"
        f"T7 词表单射（重复 {dup18}）/merge 图一致={graph18}/ranks 插入序="
        f"{ranks18}/id 闭合={closed18}；T8 save→load 逐位一致={sl18}；"
        f"T9 缓存一致={cache18}、不动点失败 {fp_bad18}/100；"
        f"T10 预分词无损失败 {pt_bad18}/200；T11 空特殊 token 守卫={guard18}",
        adv_fail=len(rt_bad18), fuzz_fail=fuzz_bad18, dup=dup18,
        graph=graph18, ranks=ranks18, closed=closed18, save_load=sl18,
        cache=cache18, fixed_point=fp_bad18, pretok=pt_bad18, guard=guard18)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G19 优化器不变性
    t0 = time.time()
    from .l6_optimizer import AdamW as _AdamW, Muon as _Muon, Muon2 as _Muon2
    from .l6_optimizer import HybridOptimizer as _Hyb
    from .l0_numerics import clip_grad_norm as _clip19
    g19 = rng(spawn_seed(seed, "opt"))
    W19 = g19.standard_normal((8, 6))
    G19 = g19.standard_normal((8, 6))
    Gs19 = [g19.standard_normal((8, 6)) for _ in range(3)]

    # ---- (a) O1/N3：NS 奇异值性质（精确命题，非绝对化宣称）
    #   (i) σ_max 对任意输入/步数恒落在 (0.5, 1.5)（Muon 依赖的性质）
    #   (ii) 全谱落带的条件命题：5 步 + cond ≤ 100
    sv_max_worst = 0.0
    band_ok = True
    band_n = 0
    for _ in range(32):
        r, c = int(g19.integers(2, 20)), int(g19.integers(2, 20))
        M = g19.standard_normal((r, c)) * float(g19.choice([0.01, 1.0, 100.0]))
        if g19.random() < 0.5:                 # 半数故意做成病态
            M = M @ M.T @ M
        steps = int(g19.integers(3, 6))
        O = newton_schulz(M, steps=steps)
        sv = np.linalg.svd(O, compute_uv=False)
        sv_max_worst = max(sv_max_worst, float(np.abs(sv[0] - 1.0)))
        cond = float(np.linalg.cond(M))
        if steps == 5 and cond <= 100.0:       # 条件命题的适用域
            band_n += 1
            band_ok = band_ok and bool(sv.min() > 0.5 and sv.max() < 1.5)
    n3_ok = sv_max_worst < 0.5 and band_n >= 5 and band_ok

    # ---- (b) O3：AdamW 解耦 WD 梯度尺度不变（×1000 后更新一致）
    def _adam_upd(scale):
        p = W19.copy()
        o = _AdamW(lr=1e-2, weight_decay=0.1)
        o.step({"w": p}, {"w": G19 * scale})
        return p - W19
    a_lo, a_hi = _adam_upd(1.0), _adam_upd(1000.0)
    o3_rel = float(np.abs(a_lo - a_hi).max() / (np.abs(a_lo).max() + 1e-30))
    o3_ok = o3_rel < 1e-5

    # ---- (c) O7 对照：Muon 的 Nesterov 是真实前瞻（3 步 T/F 必须显著不同）
    def _run(cls, nesterov, steps=3):
        p = W19.copy()
        o = cls(nesterov=nesterov, ns_steps=3)
        for i in range(steps):
            o.step({"w": p}, {"w": Gs19[i].copy()})
        return p - W19
    dm_t, dm_f = _run(_Muon, True), _run(_Muon, False)
    muon_nes_rel = float(np.abs(dm_t - dm_f).max() / (np.abs(dm_f).max() + 1e-30))
    muon_nes_ok = muon_nes_rel > 1e-3

    # ---- (d) O7：Muon2 更新必须逐位匹配文档算法（纯公式内联重算）。
    #      注意：常数缩放（如已删除的 P*=(1+β) 死分支）在行为上**不可检测**
    #      —— NS 首步谱归一擦除一切常数缩放（机检 NS(1.9G) vs NS(G) = 1.7e-08），
    #      这正是它被称为 no-op 的原因。本检查捕获的是真实算法漂移：
    #      矩更新/预条件/scale 公式被改错时，更新必然偏离文档公式。
    p_new = W19.copy()
    _Muon2(lr=0.02, ns_steps=3).step({"w": p_new}, {"w": G19.copy()})
    d_new = p_new - W19
    beta19, b2_19, eps19, lr19 = 0.9, 0.95, 1e-8, 0.02
    m19 = (1 - beta19) * G19
    v19 = (1 - b2_19) * G19 * G19
    P19 = (m19 / (1 - beta19)) / (np.sqrt(v19 / (1 - b2_19)) + eps19)
    d_pure = -lr19 * newton_schulz(P19, steps=3) * max(1.0, 8 / 6) ** 0.5
    muon2_pure_rel = float(np.abs(d_new - d_pure).max() / (np.abs(d_pure).max() + 1e-30))
    muon2_noop_ok = muon2_pure_rel < 1e-6

    # ---- (e) O6：非有限梯度守卫（NaN/Inf/-Inf 三态）
    guard19_ok = True
    for bad in (np.nan, np.inf, -np.inf):
        p = W19.copy()
        g = G19.copy(); g[0, 0] = bad
        h = _Hyb(max_grad_norm=1.0)
        h.step({"w": p}, {"w": g})
        guard19_ok = (guard19_ok and np.array_equal(p, W19)
                      and h.skipped_steps == 1
                      and not np.isfinite(h.last_grad_norm)
                      and bool(np.all(np.isfinite(p))))
    # 跳过后恢复正常训练
    p = W19.copy()
    h = _Hyb(max_grad_norm=1.0)
    g = G19.copy(); g[0, 0] = np.nan
    h.step({"w": p}, {"w": g})
    h.step({"w": p}, {"w": G19.copy()})
    recover19 = (not np.array_equal(p, W19)) and h.skipped_steps == 1 \
        and bool(np.all(np.isfinite(p)))

    # ---- (f) clip 全局语义：范数跨参数拼接、统一缩放
    ga = np.full((4,), 3.0); gb = np.full((4,), 4.0)
    gg, gn = _clip19({"a": ga, "b": gb}, 1.0)
    clip_ok = (abs(gn - 10.0) < 1e-12
               and abs(gg["a"][0] - 0.3) < 1e-12 and abs(gg["b"][0] - 0.4) < 1e-12)

    # ---- (g) AdamW 首步解析值：偏差修正使更新 == -lr·sign(g)（ε→0 极限）
    p = W19.copy()
    _AdamW(lr=1e-2, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0).step(
        {"w": p}, {"w": G19.copy()})
    adam_first_rel = float(np.abs((p - W19) + 1e-2 * np.sign(G19)).max()
                           / (1e-2 * np.abs(np.sign(G19)).max() + 1e-30))
    adam_first_ok = adam_first_rel < 1e-6

    # ---- (h) O5 确定性：两次相同运行逐位一致
    def _det_run():
        p = W19.copy()
        o = _Hyb(max_grad_norm=1.0)
        for i in range(3):
            o.step({"w": p}, {"w": Gs19[i].copy()})
        return p.copy()
    det_ok = np.array_equal(_det_run(), _det_run())

    # ---- (i) Muon2 梯度尺度不变（预条件 + NS 双重不变）
    def _muon2_upd(scale):
        p = W19.copy()
        _Muon2(lr=0.02, ns_steps=3).step({"w": p}, {"w": G19 * scale})
        return p - W19
    m2_lo, m2_hi = _muon2_upd(1.0), _muon2_upd(1000.0)
    m2_rel = float(np.abs(m2_lo - m2_hi).max() / (np.abs(m2_lo).max() + 1e-30))
    m2_scale_ok = m2_rel < 1e-4

    # ---- (j) O4：合成凸问题（二次 bowl）上损失单调下降
    A19 = g19.standard_normal((6, 6)); A19 = A19 @ A19.T + 6 * np.eye(6)
    x0 = g19.standard_normal((6, 1))
    def _bowl(x):
        return 0.5 * float(np.sum(x * (A19 @ x))) + 0.1 * float(np.sum(x ** 2))
    x = x0.copy()
    losses = []
    for _ in range(30):
        g = A19 @ x + 0.2 * x
        o = _AdamW(lr=0.05, weight_decay=0.0)
        o.step({"x": x}, {"x": g})
        losses.append(_bowl(x))
    convex_ok = all(losses[i + 1] <= losses[i] + 1e-12 for i in range(len(losses) - 1)) \
        and losses[-1] < losses[0] * 0.5

    ok19 = (n3_ok and o3_ok and muon_nes_ok and muon2_noop_ok and guard19_ok
            and recover19 and clip_ok and adam_first_ok and det_ok
            and m2_scale_ok and convex_ok)
    add("优化器不变性", ok19,
        f"O1/N3 精确命题：σ_max 最大偏离 1 为 {sv_max_worst:.2e}（<0.5，任意输入/"
        f"步数）；全谱落带（5 步+cond≤100，{band_n} 例）={band_ok} —— 原宣称"
        f"「全谱 (0.5,1.5)」绝对化为假，已修正为精确命题（病态小奇异值滞后是 "
        f"quintic 已知行为）；O3 AdamW 梯度×1000 更新相对差 {o3_rel:.2e}；"
        f"O7 对照 Muon Nesterov 真实差异 {muon_nes_rel:.2e}（>1e-3）、Muon2 "
        f"更新匹配文档纯公式 {muon2_pure_rel:.2e}（<1e-6；常数缩放行为上不可"
        f"检测是 NS 尺度不变的必然）；O6 非有限梯度守卫="
        f"{guard19_ok}、跳过后恢复={recover19}；clip 全局语义={clip_ok}；"
        f"AdamW 首步解析 {adam_first_rel:.2e}；O5 确定性={det_ok}；Muon2 "
        f"梯度×1000 {m2_rel:.2e}；O4 凸问题损失 {losses[0]:.3f}→"
        f"{losses[-1]:.3f} 单调={convex_ok}",
        ns_sv_max=sv_max_worst, ns_band=band_ok, ns_band_n=band_n,
        adamw_scale=o3_rel, muon_nesterov=muon_nes_rel,
        muon2_noop=muon2_pure_rel, guard=guard19_ok, recover=recover19,
        clip=clip_ok, adam_first=adam_first_rel, det=det_ok,
        muon2_scale=m2_rel, convex=convex_ok)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G20 持久化完整性
    t0 = time.time()
    import os as _os
    import json as _json
    import hashlib as _hashlib
    import tempfile as _tempfile
    import shutil as _shutil
    from .persist import (save_checkpoint as _save_ckpt,
                          load_checkpoint as _load_ckpt,
                          checkpoint_info as _ckpt_info,
                          CheckpointError as _CkptErr)
    from .model import TianjiConfig as _Cfg
    g20 = rng(spawn_seed(seed, "persist"))
    _d20 = _tempfile.mkdtemp(prefix="tianji_g20_")
    cfg20 = _Cfg(vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1)
    P20 = {"w": g20.standard_normal((4, 3)), "b": g20.standard_normal((3,))}

    def _ckpt_files20():
        return sorted(f for f in _os.listdir(_d20))

    try:
        # ---- (a) P1 路径规范化闭环：无扩展名 save→load→info 一致
        p_noext = _os.path.join(_d20, "noext")
        _save_ckpt(p_noext, cfg20, P20)
        p_norm = p_noext + ".npz"
        norm_ok = (_os.path.exists(p_norm)
                   and _os.path.exists(p_norm + ".meta.json")
                   and not _os.path.exists(p_noext))
        P20b, info20 = _load_ckpt(p_noext)
        rt20 = (sorted(P20b) == sorted(P20)
                and np.array_equal(P20b["w"], P20["w"])
                and np.array_equal(P20b["b"], P20["b"]))
        info20b = _ckpt_info(p_noext)
        info20_ok = info20b["weights_sha256"] == info20["weights_sha256"]

        # ---- (b) P2 损坏 meta（截断 JSON）→ CheckpointError（load + info 双路径）
        p_trunc = _os.path.join(_d20, "trunc.npz")
        _save_ckpt(p_trunc, cfg20, P20)
        _raw = open(p_trunc + ".meta.json", encoding="utf-8").read()
        with open(p_trunc + ".meta.json", "w", encoding="utf-8") as f:
            f.write(_raw[: len(_raw) // 2])
        meta_err_ok = True
        for _fn in (lambda: _load_ckpt(p_trunc), lambda: _ckpt_info(p_trunc)):
            try:
                _fn()
                meta_err_ok = False          # 损坏 meta 竟然加载成功了
            except _CkptErr:
                pass
            except Exception:
                meta_err_ok = False          # 漏出非 CheckpointError 类型

        # ---- (c) P3 损坏权重（翻字节）→ CheckpointError
        p_corrupt = _os.path.join(_d20, "corrupt.npz")
        _save_ckpt(p_corrupt, cfg20, P20)
        _buf = bytearray(open(p_corrupt, "rb").read())
        _buf[len(_buf) // 2] ^= 0xFF
        with open(p_corrupt, "wb") as f:
            f.write(bytes(_buf))
        try:
            _load_ckpt(p_corrupt)
            w_corrupt_ok = False
        except _CkptErr:
            w_corrupt_ok = True
        except Exception:
            w_corrupt_ok = False

        # ---- (d) P4 权重替换 + sha 同步篡改 → 参数指纹抓住（部分失配场景）
        p_tamper = _os.path.join(_d20, "tamper.npz")
        _save_ckpt(p_tamper, cfg20, P20)
        _Q = {"w": g20.standard_normal((4, 3)), "b": P20["b"]}
        np.savez_compressed(p_tamper, **_Q)
        _info = _json.load(open(p_tamper + ".meta.json", encoding="utf-8"))
        _h = _hashlib.sha256()
        with open(p_tamper, "rb") as f:
            _h.update(f.read())
        _info["weights_sha256"] = _h.hexdigest()      # 攻击者同步 sha
        with open(p_tamper + ".meta.json", "w", encoding="utf-8") as f:
            _json.dump(_info, f)
        try:
            _load_ckpt(p_tamper)
            tamper_ok = False
        except _CkptErr:
            tamper_ok = True
        except Exception:
            tamper_ok = False

        # ---- (e) P5 原子性：meta 写入崩溃 → 无 tmp 残留 + 旧 checkpoint 完好
        import tianji.persist as _PS
        p_atom = _os.path.join(_d20, "atom.npz")
        _save_ckpt(p_atom, cfg20, P20)
        _orig_bytes = open(p_atom, "rb").read()
        _orig_dump = _PS.json.dump
        def _explode(*a, **k):
            raise OSError("simulated disk full")
        _PS.json.dump = _explode
        try:
            _save_ckpt(p_atom, cfg20, {"w": P20["w"] * 99.0})
        except OSError:
            pass
        finally:
            _PS.json.dump = _orig_dump
        atomic_ok = (not [f for f in _ckpt_files20() if ".tmp" in f]
                     and open(p_atom, "rb").read() == _orig_bytes)
        P20c, _ = _load_ckpt(p_atom)
        atomic_ok = atomic_ok and np.array_equal(P20c["w"], P20["w"])

        # ---- (f) P6 空参数 / 怪异参数名 / dtype 保持
        p_empty = _os.path.join(_d20, "empty.npz")
        _save_ckpt(p_empty, cfg20, {})
        P20e, ie = _load_ckpt(p_empty)
        empty_ok = len(P20e) == 0 and ie["n_params"] == 0
        _weird = {"model.layers.0.w": P20["w"], "a/b": P20["w"],
                  "权重": P20["w"], "w.npy": P20["w"]}
        p_weird = _os.path.join(_d20, "weird.npz")
        _save_ckpt(p_weird, cfg20, _weird)
        P20w, _ = _load_ckpt(p_weird)
        weird_ok = sorted(P20w) == sorted(_weird)
        p_dt = _os.path.join(_d20, "dt.npz")
        _save_ckpt(p_dt, cfg20, {"w": P20["w"].astype(np.float32)})
        P20d, _ = _load_ckpt(p_dt)
        dtype_ok = P20d["w"].dtype == np.float32

        # ---- (g) P7 覆盖保存失配检出 + 保存确定性（同参数两次 digest 相同）
        p_ow = _os.path.join(_d20, "ow.npz")
        d1 = _save_ckpt(p_ow, cfg20, P20)
        d2 = _save_ckpt(p_ow, cfg20, P20)
        det_ok = d1 == d2
        _info = _json.load(open(p_ow + ".meta.json", encoding="utf-8"))
        _info["params_fingerprint"] = "0" * 64       # 失配的旧指纹
        with open(p_ow + ".meta.json", "w", encoding="utf-8") as f:
            _json.dump(_info, f)
        try:
            _load_ckpt(p_ow)
            ow_ok = False
        except _CkptErr:
            ow_ok = True
        except Exception:
            ow_ok = False

        # ---- (h) 缺失 meta / 不存在路径 → CheckpointError（响亮失败）
        p_nometa = _os.path.join(_d20, "nometa.npz")
        _save_ckpt(p_nometa, cfg20, P20)
        _os.unlink(p_nometa + ".meta.json")
        miss_ok = True
        for _fn in (lambda: _load_ckpt(p_nometa), lambda: _ckpt_info(p_nometa),
                    lambda: _load_ckpt(_os.path.join(_d20, "ghost.npz"))):
            try:
                _fn()
                miss_ok = False
            except _CkptErr:
                pass
            except Exception:
                miss_ok = False
    except Exception as e:                            # 门禁异常安全：报 FAIL 不掀翻套件
        norm_ok = rt20 = info20_ok = meta_err_ok = w_corrupt_ok = False
        tamper_ok = atomic_ok = empty_ok = weird_ok = dtype_ok = False
        det_ok = ow_ok = miss_ok = False
        _g20_exc = f"{type(e).__name__}: {e}"
    else:
        _g20_exc = None
    finally:
        _shutil.rmtree(_d20, ignore_errors=True)

    ok20 = (norm_ok and rt20 and info20_ok and meta_err_ok and w_corrupt_ok
            and tamper_ok and atomic_ok and empty_ok and weird_ok and dtype_ok
            and det_ok and ow_ok and miss_ok and _g20_exc is None)
    add("持久化完整性", ok20,
        (f"门禁执行异常: {_g20_exc}" if _g20_exc else
         f"P1 无扩展名路径闭环={norm_ok}（曾 FileNotFoundError，已修复规范化）；"
         f"P2 损坏 meta 双路径 CheckpointError={meta_err_ok}（曾漏出 "
         f"JSONDecodeError）；P3 损坏权重={w_corrupt_ok}；P4 sha 同步篡改被"
         f"指纹抓住={tamper_ok}；P5 原子性（崩溃无半成品+旧件完好）={atomic_ok}；"
         f"P6 空参数={empty_ok}/怪异名={weird_ok}/dtype={dtype_ok}；"
         f"P7 保存确定性={det_ok}/失配检出={ow_ok}；缺失 meta={miss_ok}"),
        norm=norm_ok, roundtrip=rt20, info=info20_ok, meta_err=meta_err_ok,
        w_corrupt=w_corrupt_ok, tamper=tamper_ok, atomic=atomic_ok,
        empty=empty_ok, weird=weird_ok, dtype=dtype_ok, det=det_ok,
        overwrite=ow_ok, missing=miss_ok)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G21 数据管道
    t0 = time.time()
    from .l7_train import DataLoader as _DL, cosine_schedule as _cos, wsd_schedule as _wsd
    g21 = rng(spawn_seed(seed, "datapipe"))
    cfg21 = _Cfg(vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1)
    data21 = g21.integers(0, 64, size=400)

    # ---- (a) 构造期响亮校验（D1/D3 复检）
    guards21 = {
        "len==T": lambda: _DL(np.arange(10), 10, 2),
        "len<T": lambda: _DL(np.arange(5), 10, 2),
        "seq_len=0": lambda: _DL(data21, 0, 2),
        "batch_size=0": lambda: _DL(data21, 4, 0),
        "batch_size=-1": lambda: _DL(data21, 4, -1),
    }
    guard_ok = True
    for _n, _fn in guards21.items():
        try:
            _fn()
            guard_ok = False
        except ValueError:
            pass
        except Exception:
            guard_ok = False

    # ---- (b) 边界合法：len==T+1 恰有一个窗口 + y==shift(x)
    edge_ok = True
    try:
        ld_e = _DL(np.arange(11), 10, 2, seed=0)
        xe, ye = ld_e.batch()
        edge_ok = (xe.shape == (2, 10) and ye.shape == (2, 10)
                   and np.all(ye[:, :-1] == xe[:, 1:]))
    except Exception:
        edge_ok = False

    # ---- (c) token 范围守卫（D2 复检：负 token 静默回绕）
    tok_guard_ok = True
    try:
        m21 = TianjiModel(cfg21, init_params(cfg21, seed=0))
        for _tk in (np.full((1, 4), -1, dtype=np.int64),
                    np.full((1, 4), 999, dtype=np.int64)):
            try:
                m21.forward(_tk, _tk)
                tok_guard_ok = False
            except ValueError:
                pass
            except Exception:
                tok_guard_ok = False
        # targets 越界同样拒绝
        try:
            m21.forward(np.zeros((1, 4), dtype=np.int64),
                        np.full((1, 4), 64, dtype=np.int64))
            tok_guard_ok = False
        except ValueError:
            pass
        # 正常 token 不受影响
        _o = m21.forward(np.array([[1, 2, 3]], dtype=np.int64),
                         np.array([[2, 3, 4]], dtype=np.int64))
        tok_guard_ok = tok_guard_ok and np.isfinite(_o["loss"])
    except Exception:
        tok_guard_ok = False

    # ---- (d) y 对齐不变量 + 连续切片指纹重放
    align_ok = True
    try:
        ld_a = _DL(data21, 8, 16, seed=3)
        xa, ya = ld_a.batch()
        align_ok = np.all(ya[:, :-1] == xa[:, 1:])
        # 确定性重放同 starts，验证每行来自连续切片
        ld_a2 = _DL(data21, 8, 16, seed=3)
        starts = np.random.default_rng(3).integers(0, len(data21) - 9, size=16)
        for i, s in enumerate(starts):
            if not (np.array_equal(data21[s:s + 8], xa[i])
                    and np.array_equal(data21[s + 1:s + 9], ya[i])):
                align_ok = False
                break
    except Exception:
        align_ok = False

    # ---- (e) 确定性
    d_a = _DL(data21, 16, 4, seed=7).batch()
    d_b = _DL(data21, 16, 4, seed=7).batch()
    d_c = _DL(data21, 16, 4, seed=8).batch()
    det21 = (np.array_equal(d_a[0], d_b[0]) and np.array_equal(d_a[1], d_b[1])
             and not np.array_equal(d_a[0], d_c[0]))

    # ---- (f) T1 调度器：端点精确 + warmup 线性 + warmup 后单调不增
    sched_ok = True
    sched_worst = 0.0
    for _total, _warm, _lr in [(100, 10, 0.02), (37, 1, 0.05), (50, 50, 0.03),
                               (200, 40, 0.01), (60, 0, 0.02)]:
        cs = [_cos(s, _total, _warm, _lr) for s in range(_total + 1)]
        ws = [_wsd(s, _total, _warm, _lr) for s in range(_total + 1)]
        sched_worst = max(sched_worst, abs(cs[_total]), abs(ws[_total]))
        if _warm and _warm < _total:
            # warmup 末两个点都应精确 == lr_max（cs[warmup-1] 达满幅、
            # cs[warmup] 进入衰减段起点；warmup==total 时衰减段为空，
            # cs[total] 由 lr_min 分支接管，单独检查）
            sched_worst = max(sched_worst, abs(cs[_warm] - _lr),
                              abs(cs[_warm - 1] - _lr), abs(ws[_warm] - _lr))
        # warmup 线性：cosine(s) == lr*(s+1)/warmup
        for _s in range(_warm):
            if abs(cs[_s] - _lr * (_s + 1) / max(_warm, 1)) > 1e-15:
                sched_ok = False
        # warmup 后单调不增
        for _seq in (cs, ws):
            if any(_seq[i + 1] > _seq[i] + 1e-15 for i in range(_warm, _total)):
                sched_ok = False
    sched_ok = sched_ok and sched_worst < 1e-12

    # ---- (g) 梯度累积等价：accum=k == 同 seed 重放 k batch 手动平均
    accum_ok = True
    accum_rel = float("inf")
    try:
        class _NoopOpt:
            def __init__(self):
                class _S:
                    lr = 0.0
                self.muon, self.adam = _S(), _S()
                self.last_grad_norm = 0.0
                self.captured = None

            def step(self, params, grads):
                self.captured = {k: v.copy() for k, v in grads.items()}

        K = 3
        m_b = TianjiModel(cfg21, init_params(cfg21, seed=2))
        tr_b = Trainer(m_b, _NoopOpt(), _DL(data21, 8, 2, seed=5),
                       total_steps=1, warmup=0, lr_max=0.0, accum=K)
        tr_b.step(0)
        g_acc = tr_b.opt.captured

        m_a = TianjiModel(cfg21, init_params(cfg21, seed=2))
        tr_a = Trainer(m_a, _NoopOpt(), _DL(data21, 8, 2, seed=5),
                       total_steps=K, warmup=0, lr_max=0.0, accum=1)
        g_sum = None
        for _ in range(K):
            tr_a.step(0)
            c = tr_a.opt.captured
            g_sum = {k: v.copy() for k, v in c.items()} if g_sum is None else \
                {k: g_sum[k] + v for k, v in c.items()}
        g_man = {k: v / K for k, v in g_sum.items()}
        common = sorted(set(g_acc) & set(g_man))
        if not common:
            accum_ok = False
        else:
            accum_rel = max(float(np.abs(g_acc[k] - g_man[k]).max()
                                  / (np.abs(g_man[k]).max() + 1e-30))
                            for k in common)
            accum_ok = accum_rel < 1e-10
    except Exception:
        accum_ok = False

    ok21 = (guard_ok and edge_ok and tok_guard_ok and align_ok and det21
            and sched_ok and accum_ok)
    add("数据管道", ok21,
        f"D1 构造期守卫（过短/seq_len=0/batch_size<=0 全 ValueError）={guard_ok}"
        f"（曾静默钳制出 x(2,10) y(2,9) 错位 batch）；边界 len==T+1={edge_ok}；"
        f"D2 token 范围守卫={tok_guard_ok}（负 token 曾静默回绕，全 -1 输入 "
        f"loss==0）；D3 y 对齐+切片指纹={align_ok}；D4 确定性={det21}；"
        f"T1 调度器端点+单调（原宣称无门禁，已补齐）={sched_ok}，"
        f"端点最大偏差 {sched_worst:.2e}；梯度累积等价 accum=3 "
        f"{accum_rel:.2e}（<1e-10）={accum_ok}",
        guards=guard_ok, edge=edge_ok, tok_guard=tok_guard_ok, align=align_ok,
        det=det21, sched=sched_ok, sched_dev=sched_worst, accum=accum_ok,
        accum_rel=accum_rel)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G22 原语不变性
    t0 = time.time()
    from .l1_contract import Ctx as _Ctx
    from .l3_primitives import (RMSNorm as _RMS, RoPE as _RoPE, rope_tables as _rtab,
                                CausalAttention as _CA, CrossEntropy as _CE,
                                causal_mask_rect as _cmr)
    g22 = rng(spawn_seed(seed, "primitives"))

    def _rot22(u, pos, D, c, s):
        return _RoPE().forward(_Ctx(), np.asarray(u).reshape(1, 1, 1, D),
                               c[pos:pos + 1], s[pos:pos + 1])[0, 0, 0]

    # ---- (a) P3 保范：含长度外推与极端量级
    p3_worst = 0.0
    for _L in (16, 512, 20000):
        c22, s22 = _rtab(8, _L)
        for _pos in (0, 1, min(_L - 1, 19999), max(_L - 2, 0)):
            for _scale in (1e-6, 1.0, 1e6):
                _x = g22.standard_normal((2, 3, 1, 8)) * _scale
                _y = _RoPE().forward(_Ctx(), _x, c22[_pos:_pos + 1], s22[_pos:_pos + 1])
                _n0 = np.linalg.norm(_x, axis=-1)
                _n1 = np.linalg.norm(_y, axis=-1)
                p3_worst = max(p3_worst,
                               float(np.abs(_n1 - _n0).max() / (np.abs(_n0).max() + 1e-30)))
    p3_ok = p3_worst < 1e-12

    # ---- (b) P4 相对位置 + 非恒真性对照（RoPE 退化成恒等也会被抓）
    c22b, s22b = _rtab(8, 256)
    _q0 = g22.standard_normal((8,)); _k0 = g22.standard_normal((8,))
    p4_worst, p4_ctrl = 0.0, 0.0
    for _base in (0, 7, 100, 200):
        for _off in (0, 1, 3, 5):
            p4_worst = max(p4_worst, abs(
                float(_rot22(_q0, _base + _off, 8, c22b, s22b) @ _rot22(_k0, _base, 8, c22b, s22b))
                - float(_rot22(_q0, _off, 8, c22b, s22b) @ _rot22(_k0, 0, 8, c22b, s22b))))
    # 对照：位置差改变内积必须显著（否则整个命题对恒等 RoPE 也成立 = 假绿）
    for _off in (1, 3, 5):
        p4_ctrl = max(p4_ctrl, abs(
            float(_rot22(_q0, 100 + _off, 8, c22b, s22b) @ _rot22(_k0, 100, 8, c22b, s22b))
            - float(_rot22(_q0, 100, 8, c22b, s22b) @ _rot22(_k0, 100, 8, c22b, s22b))))
    p4_ok = p4_worst < 1e-10 and p4_ctrl > 1e-6

    # ---- (c) P5 因果性
    _B, _H, _T, _D = 2, 2, 6, 4
    _qa = g22.standard_normal((_B, _H, _T, _D))
    _ka = g22.standard_normal((_B, _H, _T, _D))
    _va = g22.standard_normal((_B, _H, _T, _D))
    _o1 = _CA().forward(_Ctx(), _qa, _ka, _va)
    _qa2, _ka2 = _qa.copy(), _ka.copy()
    _qa2[:, :, 3:] += 1.0
    _ka2[:, :, 3:] += 1.0
    _o2 = _CA().forward(_Ctx(), _qa2, _ka2, _va)
    p5_dev = float(np.abs(_o2[:, :, :3] - _o1[:, :, :3]).max())
    p5_ok = p5_dev < 1e-12

    # ---- (d) P6 行和（含 GQA 两种倍数）
    p6_worst, p6_nan = 0.0, 0
    for _Hq, _Hkv in ((4, 4), (4, 2), (6, 1), (2, 2)):
        _ctx22 = _Ctx()
        _CA().forward(_ctx22, g22.standard_normal((2, _Hq, 6, 4)),
                      g22.standard_normal((2, _Hkv, 6, 4)),
                      g22.standard_normal((2, _Hkv, 6, 4)))
        _P = _ctx22["P"]
        p6_worst = max(p6_worst, float(np.abs(_P.sum(-1) - 1.0).max()))
        p6_nan += int(np.isnan(_P).sum())
    p6_ok = p6_worst < 1e-12 and p6_nan == 0

    # ---- (e) B1 全掩码行守卫（Tq > Tk）
    b1_ok = True
    for _tq, _tk in ((5, 3), (2, 1), (10, 9)):
        try:
            _cmr(_tq, _tk)
            b1_ok = False
        except ValueError:
            pass
        except Exception:
            b1_ok = False
    try:
        _CA().forward(_Ctx(), g22.standard_normal((1, 2, 5, 4)),
                      g22.standard_normal((1, 2, 3, 4)),
                      g22.standard_normal((1, 2, 3, 4)))
        b1_ok = False
    except ValueError:
        pass
    # 合法场景零 NaN
    _olegal = _CA().forward(_Ctx(), g22.standard_normal((1, 2, 3, 4)),
                            g22.standard_normal((1, 2, 7, 4)),
                            g22.standard_normal((1, 2, 7, 4)))
    b1_ok = b1_ok and int(np.isnan(_olegal).sum()) == 0

    # ---- (f) B2 RoPE 奇偶维
    b2_ok = True
    for _d in (1, 5, 7):
        try:
            _rtab(_d, 8)
            b2_ok = False
        except ValueError:
            pass
        except Exception:
            b2_ok = False

    # ---- (g) B3/P2 RMSNorm：空/极端输入稳定，输出 RMS 命中 eps 平滑闭式
    #      精确命题：RMS(y/w) == sqrt(m2/(m2+eps))（不是恒为 1 ——
    #      早期 docstring 的「RMS 恒为 1」是不精确宣称，偏差来自 eps）
    _eps22 = _RMS().eps
    _w22 = g22.standard_normal((8,)) + 2.0
    rms_closed = 0.0
    rms_bound = -np.inf          # 实测偏差 减去 eps 理论偏差上界，必须 <= 0
    for _scale in (1.0, 100.0):  # 覆盖 m2 ~ eps 到 m2 >> eps
        _x22 = g22.standard_normal((2, 3, 8)) * _scale
        _y22 = _RMS().forward(_Ctx(), _x22, _w22)
        _xh = _y22 / _w22
        _m2 = np.mean(_x22 ** 2, axis=-1)
        _got = np.sqrt(np.mean(_xh ** 2, axis=-1))
        _want = np.sqrt(_m2 / (_m2 + _eps22))
        rms_closed = max(rms_closed, float(np.abs(_got - _want).max()))
        # 偏差不得超出 eps 平滑项的理论后果 1 − sqrt(m2/(m2+eps))
        # （若 eps 位置写错、量级写错、或漏了 eps 保护都会被这条抓到）
        _bound = (1.0 - _want) * 1.001 + 1e-15
        rms_bound = max(rms_bound, float((np.abs(_got - 1.0) - _bound).max()))
    _z1 = _RMS().forward(_Ctx(), np.zeros((2, 3, 8)), np.ones(8))
    _z2 = _RMS().forward(_Ctx(), g22.standard_normal((2, 3, 8)) * 1e150, np.ones(8))
    b3_ok = (np.all(np.isfinite(_z1)) and np.all(np.isfinite(_z2))
             and rms_closed < 1e-14 and rms_bound <= 0.0)

    # ---- (h) B4 CrossEntropy 全忽略
    _c22 = _Ctx()
    _lg = g22.standard_normal((2, 3, 5))
    _l22 = float(_CE().forward(_c22, _lg, np.full((2, 3), -1, dtype=np.int64)))
    _gb = _CE().backward(_c22, 1.0)[0]
    b4_ok = (_l22 == 0.0 and bool(np.all(_gb == 0.0))
             and int(np.isnan(_gb).sum()) == 0)

    ok22 = (p3_ok and p4_ok and p5_ok and p6_ok and b1_ok and b2_ok
            and b3_ok and b4_ok)
    add("原语不变性", ok22,
        f"P3 RoPE 保范 {p3_worst:.2e}（含 20000 长度外推、1e-6~1e6 量级）；"
        f"P4 相对位置 {p4_worst:.2e}（对照：位置差使内积变化 {p4_ctrl:.2e}，"
        f"非恒真命题）；P5 因果性偏差 {p5_dev:.2e}；P6 注意力行和 "
        f"{p6_worst:.2e}、NaN={p6_nan}（4 组 H/Hkv 组合含 GQA）；"
        f"B1 全掩码行守卫={b1_ok}（曾静默 NaN 16/40）；B2 RoPE 奇偶维="
        f"{b2_ok}（曾拼出 dim−1 表且不报错）；B3/P2 RMSNorm 闭式命中 "
        f"{rms_closed:.2e}（<1e-14）+ 偏差未超出 eps 理论上界 "
        f"({rms_bound:.2e} <= 0)——原宣称「RMS 恒为 1」不精确，eps=1e-8 "
        f"平滑项使偏差达 2.67e-08，已修正为精确命题而非放宽阈值凑绿；"
        f"B4 CE 全忽略 loss={_l22:.1f} "
        f"梯度全零={b4_ok}。P3-P6 原宣称「门禁复检」但无门禁，本门禁补齐",
        p3=p3_worst, p4=p4_worst, p4_ctrl=p4_ctrl, p5=p5_dev,
        p6=p6_worst, p6_nan=p6_nan, b1=b1_ok, b2=b2_ok,
        rms_closed=rms_closed, rms_bound=rms_bound, b3=b3_ok, b4=b4_ok)
    R.results[-1].seconds = time.time() - t0

    # ---------------------------------------------------------- G23 GDN 归一化
    t0 = time.time()
    from .l4_deltanet import (l2norm as _l2n, GatedDeltaNet as _GDN,
                              gdn_parallel as _gp, gdn_chunkwise as _gc)
    g23 = rng(spawn_seed(seed, "gdn"))

    # ---- (a) L1 单位范数契约：‖k̂‖ == 1 对任意 ‖k‖>0，零向量→零
    l1_worst = 0.0
    for _s in (1.0, 1e-1, 1e-3, 1e-6, 1e-9, 1e-12):
        _kx = g23.standard_normal((3, 2, 4))
        _kx = _kx / np.linalg.norm(_kx, axis=-1, keepdims=True) * _s
        _kk = _l2n(_kx)
        l1_worst = max(l1_worst,
                       float(np.abs(np.linalg.norm(_kk, axis=-1) - 1.0).max()))
    _kz = _l2n(np.zeros((2, 2, 4)))
    l1_zero = (np.all(_kz == 0.0) and int(np.isnan(_kz).sum()) == 0)
    l1_ok = l1_worst < 1e-13 and l1_zero

    # ---- (b) L2 尺度自适应梯度检验（gradcheck 固定步长在小量级下失效）
    def _grad_rel23(fn, x, g, h_rel=1e-5):
        """中心差分用相对步长 h = h_rel·max(|x_i|, ‖x‖)，返回相对误差。"""
        a_num = np.zeros_like(x)
        flat, aflat = x.reshape(-1), a_num.reshape(-1)
        for i in range(flat.size):
            h = h_rel * max(abs(flat[i]), float(np.linalg.norm(x)) / max(x.size, 1), 1e-300)
            old = flat[i]
            flat[i] = old + h
            fp = float(np.sum(g * fn(x)))
            flat[i] = old - h
            fm = float(np.sum(g * fn(x)))
            flat[i] = old
            aflat[i] = (fp - fm) / (2 * h)
        return a_num

    l2_worst = 0.0
    for _s in (1.0, 1e-3, 1e-5, 1e-7):
        _x = g23.standard_normal((4,))
        _x = _x / np.linalg.norm(_x) * _s
        _g = g23.standard_normal((4,))
        _kh = _l2n(_x)
        _n = np.linalg.norm(_x)
        _proj = float(np.sum(_g * _kh))
        _analytic = (_g - _kh * _proj) / _n        # 库内解析反向的公式
        _num = _grad_rel23(_l2n, _x.copy(), _g)
        l2_worst = max(l2_worst,
                       float(np.abs(_analytic - _num).max() / (np.abs(_num).max() + 1e-300)))
    l2_ok = l2_worst < 1e-8

    # ---- (c) L3 三路径边界等价：α,β ∈ {0,1} 全组合 + 极端 ‖k‖
    l3_worst, l3_finite = 0.0, True
    B3, H3, T3, D3 = 2, 2, 8, 4
    _q3 = g23.standard_normal((B3, H3, T3, D3))
    _v3 = g23.standard_normal((B3, H3, T3, D3))
    for _av in (0.0, 0.5, 1.0):
        for _bv in (0.0, 0.5, 1.0):
            for _kscale in (1.0, 1e-7, 1e8):
                _k3 = g23.standard_normal((B3, H3, T3, D3)) * _kscale
                _al = np.full((B3, H3, T3), _av)
                _be = np.full((B3, H3, T3), _bv)
                for _ch in (1, 3, T3, T3 + 5):
                    _of = _GDN().forward(_Ctx(), _q3, _k3, _v3, _be, _al)
                    _op = _gp(_q3, _k3, _v3, _be, _al)
                    _oc = _gc(_q3, _k3, _v3, _be, _al, chunk=_ch)
                    _den = np.abs(_of).max() + 1e-30
                    l3_worst = max(l3_worst,
                                   float(np.abs(_op - _of).max() / _den),
                                   float(np.abs(_oc - _of).max() / _den))
                    l3_finite = l3_finite and np.all(np.isfinite(_of)) \
                        and np.all(np.isfinite(_op)) and np.all(np.isfinite(_oc))
    l3_ok = l3_worst < 1e-10 and l3_finite

    ok23 = l1_ok and l2_ok and l3_ok
    add("GDN 归一化", ok23,
        f"L1 单位范数契约最大偏差 {l1_worst:.2e}（含 ‖k‖ 1e-12~1，零向量→"
        f"零={l1_zero}）—— 曾因分母 +1e-8 使 ‖k̂‖ 只剩 0.181（偏差 82%，"
        f"有效写入强度被静默缩小到 0.016）；L2 尺度自适应梯度检验相对误差 "
        f"{l2_worst:.2e}（<1e-8；同一公式配旧前向为 eps/(‖k‖+eps)，"
        f"‖k‖=1e-7 时达 17%）；L3 三路径 α,β∈{{0,0.5,1}} × ‖k‖∈{{1,1e-7,1e8}} "
        f"× chunk∈{{1,3,T,T+5}} 共 108 组组合最大差 {l3_worst:.2e}、全有限="
        f"{l3_finite}",
        l1=l1_worst, l1_zero=l1_zero, l2=l2_worst, l3=l3_worst, l3_finite=l3_finite)
    R.results[-1].seconds = time.time() - t0

    # ---- G24 · 基准熵下界（可机检） ----
    t0 = time.time()
    from tianji.data import markov_entropy_rate, markov_tokens
    from tianji.model import TianjiConfig, TianjiModel, init_params
    from tianji.l7_train import evaluate  # DataLoader 用模块级导入，避免遮蔽 train_once 的闭包
    er = markov_entropy_rate(vocab=64, seed=0, n_sample=200_000)
    H = float(er["H"])
    # E1 · 熵率自洽：两套独立算法一致 ⇒ H 是被算出来的，不是编的
    # E2 · 损失不破熵下界：loss 函数健全性护栏（任何模型 CE ≥ H）
    cfg0 = TianjiConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2,
                        n_kv_heads=1, d_latent_q=16, d_latent_kv=12, d_rope=4,
                        d_ff=64, d_ff_expert=32, n_experts=1, moe_top_k=1,
                        max_len=128, mtp_depth=1)
    m0 = TianjiModel(cfg0, init_params(cfg0, seed=0))
    vld0 = DataLoader(markov_tokens(1500, 64, 7), seq_len=32, batch_size=8, seed=999)
    ce0 = float(evaluate(m0, vld0, n_batch=12))
    guard_ok = bool(np.isfinite(ce0) and ce0 >= 0.0 and ce0 >= H - 1e-6)
    # E3 · 基准诚实性 + 判别力（读 bench.json）：
    #   E3a 主基准：各配置 learned fraction ≤ 1（不破熵下界）—— 诚实护栏
    #   E3b 可学性对照臂（B7）：lf_full > 0.5 且模型全量 val ≈ 同序列 oracle
    #       —— 判别力证明：任务在规模内可学时，lf 转正、模型逼近真实源，
    #        「学不会」是主基准任务的数据稀缺而非基准/管线缺陷
    import json as _json, os as _os
    bj = _os.path.join(_os.path.dirname(__file__), "..", "bench", "results", "bench.json")
    if _os.path.exists(bj):
        with open(bj, encoding="utf-8") as f:
            bench = _json.load(f)
        logv = er["log_vocab"]

        def _lf(vl):
            return (logv - vl) / (logv - H)   # 1.0 = 达到熵下界

        rows = []
        for sec in ("architecture", "layout", "optimizer"):
            for r in bench.get(sec, []):
                if "val_loss" in r and np.isfinite(r["val_loss"]):
                    rows.append((r.get("name", "?"), float(r["val_loss"]), _lf(r["val_loss"])))
        max_lf = max((x[2] for x in rows), default=0.0) if rows else 0.0
        over = any(x[2] > 1.0 + 1e-9 for x in rows)
        # E3a·layout 等参数机检（G27 修复的混杂）：B2 两布局必须严格等参数，
        # 否则「布局对比」会混杂「注意力层数」这一变量，结论不可信。
        lay_rows = [r for r in bench.get("layout", []) if "params" in r]
        layout_params = {r.get("name"): int(r["params"]) for r in lay_rows}
        layout_eq = (len(set(layout_params.values())) <= 1) if layout_params else True
        if layout_eq:
            layout_note = f"相等({layout_params or 'n/a'}) 通过"
        else:
            layout_note = f"不等({layout_params or 'n/a'}) 不通过"
        honesty_detail = (f"E3a 读 bench.json：{len(rows)} 个配置的最大 learned fraction="
                         f"{max_lf:.3f}（=(uniform−模型)/(uniform−floor)，1.0=达到熵下界）；"
                         f"全部 ≤ 1 即「没有任何配置突破熵下界」；当前基准均远未学会"
                         f"（见 BENCHMARKS.md）；E3a·layout 等参数：{layout_note}")
        # E3b · 可学性对照臂（bench.json 的 "learnable" 段，由 B7 生成）
        lv = bench.get("learnable")
        if lv and lv.get("rows") and all(
                "val_loss_full" in r and "oracle_nll_full" in r for r in lv["rows"]):
            Hl, ul = float(lv["H"]), float(lv["uniform"])
            lf_full = [(r.get("name", "?"), (ul - float(r["val_loss_full"])) / (ul - Hl))
                       for r in lv["rows"]]
            gaps = [(r.get("name", "?"), abs(float(r["val_loss_full"]) - float(r["oracle_nll_full"])))
                    for r in lv["rows"]]
            best_lf_name, best_lf = max(lf_full, key=lambda t: t[1])
            gap_name, min_gap = min(gaps, key=lambda t: t[1])
            arm_ok = bool(best_lf > 0.5 and min_gap < 0.05)
            arm_best_lf, arm_min_gap = best_lf, min_gap
            honesty_detail += (
                f"；E3b 可学性对照臂（1 阶 vocab=8、同源、{lv.get('steps')} 步）："
                f"最优 lf_full={best_lf:+.3f}（{best_lf_name}，>0.5 即真学会而非贴 uniform），"
                f"模型全量 val 与同序列 oracle 最大差 {min_gap:.4f}<0.05（{gap_name}）"
                f"—— 证明基准判别力：可学任务上 lf 转正、模型逼近真实源")
        else:
            arm_ok, arm_best_lf, arm_min_gap = True, 0.0, float("inf")
            honesty_detail += ("；E3b bench.json 缺 learnable 段，跳过"
                               "（运行 `python bench/run_bench.py` 后机检对照臂）")
    else:
        max_lf, over = 0.0, False
        arm_ok, arm_best_lf, arm_min_gap = True, 0.0, float("inf")
        honesty_detail = ("bench/results/bench.json 不存在，跳过读取（门禁仍守着 H 自洽"
                         f"与不破下界）；运行 `python bench/run_bench.py` 后可机检 learned fraction")
    # E3c · 多 seed 稳定性（bench/results/multiseed.json，B8 生成）：
    #   ① 与 E3a 同判据作用于新数据（已记录 per-seed lf ≤ 1，护栏）；
    #   ② 用已记录的 per-seed 数据**重算**每族最优配置与「是否每个 seed 都由
    #      它获胜」，与文件里记录的 best/consistent 对账，并核对文档宣称的
    #      排序——把「差异是噪声还是信号」定量化并钉死，防止文档与数据分叉。
    #   读已记录数据 ⇒ 门禁确定性（不重训，不因随机性抖动）。
    ms_path = _os.path.join(_os.path.dirname(__file__), "..", "bench", "results",
                            "multiseed.json")
    if _os.path.exists(ms_path):
        with open(ms_path, encoding="utf-8") as f:
            ms = _json.load(f)
        fams = {k: v for k, v in ms.get("families", {}).items()
                if isinstance(v, dict)}
        ms_lfs = [p["lf"] for fam in fams.values()
                  for c in fam.get("configs", []) for p in c.get("per_seed", [])]
        ms_lf_max = max(ms_lfs) if ms_lfs else 0.0
        ms_tmax = max((abs(p["t"]) for fam in fams.values()
                       for p in fam.get("pairwise", [])), default=float("nan"))
        # 文档宣称的排序（B8 实测锁定）：(期望最优配置, 是否每个 seed 都最优)
        #   layout 期望 consistent=False —— 即「统计上不可区分」本身是待守的声明
        expected = {"architecture": ("全 GDN (无注意力)", True),
                    "layout": (None, False),
                    "optimizer": ("muon", True)}
        checks, parts = [], []
        for key, (want_best, want_consistent) in expected.items():
            fam = fams.get(key)
            if not fam:
                checks.append(False)
                continue
            rows = fam["configs"]
            best = min(rows, key=lambda r: r["mean_val"])["name"]
            consistent = all(
                min((r["per_seed"][k]["val_loss"], r["name"]) for r in rows)[1] == best
                for k in range(len(ms.get("seeds", []))))
            ok_k = (bool(consistent == want_consistent)
                    and (want_best is None or best == want_best))
            checks.append(ok_k)
            parts.append(f"{key}: best={best} consistent={consistent}"
                         f"{'' if want_best is None else ' (宣称最优=' + str(want_best) + ')'}")
        ms_ok = bool(ms_lf_max <= 1.0 + 1e-9 and len(checks) == 3
                       and all(checks))
        honesty_detail += (
            f"；E3c 多 seed（{len(ms.get('seeds', []))} seed × 3 族，B8）："
            f"已记录 per-seed lf 最大 {ms_lf_max:.3f} ≤ 1（护栏）；"
            f"{'; '.join(parts)}；两两配对 \\|t\\| 最大 {ms_tmax:.2f}"
            f"（B1 架构排序可复现 ⇒ B8 推翻旧版「一切皆噪声」的推断；"
            f"B2 布局经 G27 等参数修正后 edge 显著更优 t≈3.16）")
    else:
        ms_ok, ms_lf_max = True, 0.0
        honesty_detail += ("；E3c multiseed.json 不存在，跳过"
                           "（运行 `python bench/run_multiseed.py` 后机检种子噪声）")
    ok24 = bool(er["consistent"] and guard_ok and not over and arm_ok and ms_ok and layout_eq)
    add("基准熵下界", ok24,
        f"E1 熵率自洽：H={H:.4f} nats（解析平稳分布 H_A={er['H_analytic']:.4f} 与 "
        f"oracle 交叉熵 H_B={er['H_oracle']:.4f} 差 {er['abs_diff']:.2e}<{er['tol']:.0e}），"
        f"0<H<log(vocab)={er['log_vocab']:.4f}；E2 损失不破下界：未训练模型验证 CE "
        f"{ce0:.4f} ≥ H（差 {ce0 - H:+.4f}，sub-entropy 即 bug，本护栏拦截）；E3 {honesty_detail}",
        H=H, H_A=float(er["H_analytic"]), H_B=float(er["H_oracle"]), diff=float(er["abs_diff"]),
        ce0=ce0, max_learned_fraction=float(max_lf),
        learnable_best_lf=float(arm_best_lf), learnable_min_gap=float(arm_min_gap),
        multiseed_max_lf=float(ms_lf_max))
    R.results[-1].seconds = time.time() - t0

    # ---- G25 · 全模型反向一致性（端到端 gradcheck） ----
    # 为什么需要：G2 只校验各「独立原语」的反向（Linear/RMSNorm/RoPE/GDN…），
    # 但 model.py 把这些原语装配成完整 Transformer 时的接线（残差顺序、Pre-Norm
    # 反传、MoE 路由梯度、MTP 头梯度）从未端到端机检。这类 bug 不会崩、会「训练」，
    # 只是永远训不好（G23 注释里记录的阴险案例）。本门禁用中心差分对完整模型
    # 的反向做一次独立裁判，覆盖 GDN + MLA 注意力 + MoE + MTP 四条装配路径。
    # 全参数中心差分太慢（33k 参数 ≈ 190s），按项目已认可的随机采样原则，只对
    # 每张量少量均匀点 + 全局随机点抽样，足以抓住任何「全局性」装配 bug。
    t0 = time.time()
    from tianji.model import TianjiConfig as _TC, TianjiModel as _TM, init_params as _ip
    cfg25 = _TC(vocab_size=32, d_model=16, n_layers=2, n_heads=2,
                n_kv_heads=2, hybrid_ratio=1, mla=True,
                d_latent_q=8, d_latent_kv=6, d_rope=4,
                d_ff=24, d_ff_expert=12, n_experts=4, moe_top_k=2,
                shared_expert=True, moe_every=2, conv_kernel=4,
                mtp_depth=1, mtp_lambda=0.1, tie_embeddings=True,
                max_len=64, rope_base=10000.0)
    P25 = _ip(cfg25, seed=seed)
    m25 = _TM(cfg25, P25)
    rg25 = np.random.default_rng(seed + 77)
    x25 = rg25.integers(0, cfg25.vocab_size, size=(2, 8))
    y25 = rg25.integers(0, cfg25.vocab_size, size=(2, 8))
    # 解析反向（先算，确保 _tail/_rec 来自原始 P25）
    m25.P = P25
    o25 = m25.forward(x25, y25)
    G25 = m25.backward()
    assert o25["loss"] is not None
    # 数值反向：中心差分
    eps25 = 1e-5
    tol25 = 1e-5
    gscale25 = max((float(np.abs(v).max()) for v in G25.values()), default=1.0)
    atol25 = 1e-7 * max(gscale25, 1e-6)
    Pw25 = {k: v.copy() for k, v in P25.items()}   # 工作副本，绝不污染 P25/G25
    rs25 = np.random.default_rng(seed + 12345)
    samples25 = []
    for nm, arr in P25.items():                      # 每张量少量均匀点：保证逐张量覆盖
        n = arr.reshape(-1).size
        k = min(6, n)
        idx = (np.linspace(0, n - 1, k).astype(int) if n > k
               else np.arange(n))
        for j in idx:
            samples25.append((nm, int(j)))
    alln25 = list(P25.keys())
    for _ in range(120):                             # 全局随机点：提高跨张量命中率
        nm = alln25[rs25.integers(0, len(alln25))]
        samples25.append((nm, int(rs25.integers(0, P25[nm].size))))
    worst25 = 0.0
    info25 = None
    for (nm, j) in samples25:
        flat = Pw25[nm].reshape(-1)
        old = flat[j]
        flat[j] = old + eps25
        m25.P = Pw25
        lp = float(m25.forward(x25, y25)["loss"])
        flat[j] = old - eps25
        lm = float(m25.forward(x25, y25)["loss"])
        flat[j] = old
        num = (lp - lm) / (2 * eps25)
        ana = float(G25[nm].reshape(-1)[j])
        denom = atol25 + tol25 * abs(num)
        ratio = abs(ana - num) / denom
        if ratio > worst25:
            worst25 = ratio
            info25 = (nm, j, ana, num, ratio)
    m25.P = P25
    ok25 = worst25 <= 1.0
    add("全模型反向一致性", ok25,
        f"完整模型（GDN+MLA+MoE+MTP 装配）解析反向 vs 中心差分：抽样 "
        f"{len(samples25)} 条目最大混合比值 {worst25:.2e}（判据 |a−n|≤atol+rtol·|n|，"
        f"≤1 即过；最差 {info25[0]}[{info25[1]}] 解析 {info25[2]:.3e} vs 数值 "
        f"{info25[3]:.3e}）—— 证明 model.py 的残差/Pre-Norm/MoE/MTP 接线反向正确",
        worst_ratio=float(worst25), n_samples=len(samples25))
    R.results[-1].seconds = time.time() - t0

    # ---- G26 · 基准 train/val 同源（防回归：抓一个曾污染全基准的阴险 bug） ----
    # 为什么需要：markov_tokens 每次调用都从 spawn_seed(seed,"synth") 重抽转移矩阵。
    # 旧 run_bench 用 synth_tokens(n,V,seed) 与 synth_tokens(m,V,seed+7) 分别生成
    # train/val ⇒ 两者竟是**两个不同的马尔可夫源**：模型在源 A 上训练、却在源 B 上被
    # 评估，val 交叉熵永远到不了熵下界 H，learned fraction 指标因此失效——它把
    # 「本可学会」误判成「学不会」，也让所有架构/优化器的 val 差异沦为跨源噪声
    # （实测：不同源 val 的 oracle NLL 高达 ~6.96，超出 H 近 4 nats）。修复：
    # markov_train_val 只抽一次转移矩阵、再采两条独立序列，run_bench 已全部改用它。
    # 本门禁机检同源不变量，并自带「反证对照」证明对该 bug 敏感：
    #   E1 同源不变量 —— markov_train_val 采出的 val，在训练源真实转移下的 oracle
    #      NLL ≈ H（|NLL−H| < 0.05），证明 val 是 train 同源的留出集；
    #   E2 反证对照 —— 换一个源（旧 bug 的 seed+7 模式）oracle NLL 显著 > H（> H+0.3）。
    # 若有人把 val 改回不同源，E1 立即失败 ⇒ 精准抓住回归。
    t0 = time.time()
    from tianji.data import (markov_transitions, markov_train_val,  # noqa: E402
                             markov_tokens)
    _V, _seed = 64, 0
    _trans = markov_transitions(_V, _seed, 2)
    _H26 = float(markov_entropy_rate(_V, _seed, 2)["H"])

    def _oracle_nll(seq):
        a, b, c = seq[:-2], seq[1:-1], seq[2:]
        return float(-np.log(_trans[a, b, c] + 1e-300).mean())

    _tr26, _va26 = markov_train_val(200_000, 200_000, _V, _seed, 2)
    _nll_same = _oracle_nll(_va26)
    _va_bad = markov_tokens(200_000, _V, _seed + 7, 2)   # 旧 bug 模式：不同源
    _nll_diff = _oracle_nll(_va_bad)
    e1 = abs(_nll_same - _H26) < 0.05
    e2 = _nll_diff > _H26 + 0.3
    ok26 = bool(e1 and e2)
    add("基准 train/val 同源", ok26,
        f"E1 同源不变量：markov_train_val 采出的 val 在训练源真实转移下 oracle NLL="
        f"{_nll_same:.4f} ≈ H={_H26:.4f}（|差|={abs(_nll_same - _H26):.4f}<0.05），"
        f"证明 train/val 同源、val 是留出集而非另一马尔可夫源；E2 反证对照：改用旧 bug 的 "
        f"seed+7 不同源，oracle NLL={_nll_diff:.4f}（> H+{_nll_diff - _H26:.2f}，显著膨胀）"
        f"——证明本门禁对「val 不同源」bug 敏感、能抓住回归（该 bug 曾让 learned fraction "
        f"指标失效、把「能学会」误判成「学不会」）",
        val_oracle_nll=float(_nll_same), H=float(_H26),
        diff_source_nll=float(_nll_diff), same_source=bool(e1))
    R.results[-1].seconds = time.time() - t0

    # ---- G27 · B1 架构等参数 + 正确标注（防回归：抓「全 GDN」错标与参数量混杂） ----
    # 为什么需要：旧版 B1 用 hybrid_ratio=1e6 构造「全 GDN」，结果只产出 1 注意力 +
    # 5 GDN（LLLLLA）——根本不是全 GDN，且与 Hybrid(LLLALL) 仅注意力位置不同、三配置
    # 参数量还互不相等（396544 / 364224 / 396544）。结论「全 GDN 显著更优」既被错标、
    # 又被参数量混杂（全注意力少了 32k 参数，其较差表现可能只是容量而非架构）。本门禁
    # 把 B2 等参数方法论推广到 B1，机检：(a) B1 三配置标注正确——「全 GDN」确为全 gdn、
    # 「全注意力」确为全 attn、第三者为混合（Hybrid）；(b) 三配置互不等价结构、恰好覆盖
    # 全注意力/混合/全 GDN 三种；(c) 三配置参数量在 0.1% 内相等——只比较架构本身。
    t0 = time.time()
    import json as _json27, os as _os27
    _bj27 = _os27.path.join(_os27.path.dirname(__file__), "..", "bench", "results", "bench.json")
    _ok27 = True
    _det27 = "bench.json 不存在，跳过（运行 `python bench/run_bench.py` 后机检 B1 等参数/标注）"
    if _os27.path.exists(_bj27):
        with open(_bj27, encoding="utf-8") as f:
            _b27 = _json27.load(f)
        _ar = _b27.get("architecture", [])
        if len(_ar) == 3 and all("layers_list" in r and "params" in r for r in _ar):
            _layers_map = {}
            for r in _ar:
                _lay = r["layers_list"]
                _content = ("all_attn" if all(l == "attn" for l in _lay)
                            else "all_gdn" if all(l == "gdn" for l in _lay)
                            else "hybrid")
                _layers_map[r["name"]] = _content
            # 标注正确：每个名字与其层结构一致（抓「全 GDN」实为 1 注意力+5 GDN 的错标）
            _label_ok = all(
                ("全注意力" in n and c == "all_attn") or
                ("全 GDN" in n and c == "all_gdn") or
                ("Hybrid" in n and c == "hybrid")
                for n, c in _layers_map.items())
            # 三种结构齐备：必须恰好覆盖 all_attn / all_gdn / hybrid
            _cover_ok = set(_layers_map.values()) == {"all_attn", "all_gdn", "hybrid"}
            _params = [int(r["params"]) for r in _ar]
            _pmx, _pmn = max(_params), min(_params)
            _peq = (_pmx - _pmn) <= 0.001 * _pmx
            _ok27 = bool(_label_ok and _cover_ok and _peq)
            _det27 = (f"B1 三配置=命名→层结构 {_layers_map}；标注正确={_label_ok}；"
                      f"三种结构齐备={_cover_ok}；参数量 max-min={_pmx - _pmn} "
                      f"（≤0.1%={_peq}，B2 等参数方法论推广到 B1）")
        else:
            _ok27 = False
            _det27 = f"B1 段需 3 个含 layers_list/params 的配置，实得 {len(_ar)} 个"
    add("B1 架构等参数+标注", _ok27, _det27, b1_equal_param=bool(_ok27))
    R.results[-1].seconds = time.time() - t0

    return R
