# 天玑 TIANJI · 混合架构语言模型全栈

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/CJX0712/tianji-ai/blob/main/LICENSE) [![Language](https://img.shields.io/badge/Python-3776AB?logo=python)](https://github.com/CJX0712/tianji-ai) [![Verified](https://img.shields.io/badge/invariants-verified-brightgreen)](https://github.com/CJX0712) [![Author](https://img.shields.io/badge/author-晨星-blueviolet)](https://github.com/CJX0712)

</div>


> **Gated DeltaNet × MLA × MoE × Muon2 × 投机解码**
> 零重写 · 集大成 · 闭环交付 · 可证门禁
>
> 作者：**晨星（Chenxing）** · 许可证：MIT License · 仓库：<https://github.com/CJX0712/tianji-ai>

---

## 目录

- [这是什么](#这是什么)
- [技术选型：为什么是这些方案](#技术选型为什么是这些方案)
- [架构说明](#架构说明)
- [可量化验收指标](#可量化验收指标)
- [快速开始](#快速开始)
- [复现步骤](#复现步骤)
- [性能对比数据](#性能对比数据)
- [文档索引](#文档索引)
- [引用来源清单](#引用来源清单)
- [仓库结构](#仓库结构)
- [已知边界（不掩饰）](#已知边界不掩饰)
- [许可证与署名](#许可证与署名)

---

## 这是什么

一个**十层架构**的语言模型全栈，从数值内核到偏好对齐，每层都站在世界顶级公开成果之上。
它不追求"从零发明"，而是把 2026 年已被工业界验证的最优解**正确地组合起来**，
并用**可复现的基准数据**与**可证门禁**证明它真的能用。

核心架构是 **3:1 混合**：每 4 层里 3 层 Gated DeltaNet（线性）、1 层 MLA（全注意力）。
这是 Qwen3-Next / Qwen3.5、Kimi Linear、Olmo Hybrid 在 2026 年共同收敛到的布局。

**最小示例 30 步训练后的生成结果**（提示 `the quick brown`）：

```
the quick brown fox jumps over the lazy dog. the lazy dog barks at
the quick brown fox. to be or not to be,
```

---

## 技术选型：为什么是这些方案

### 方向选择：为什么做 LLM 全栈

晨星已有 100+ 个开源仓库，覆盖经典 ML、GNN、RL、扩散、流匹配、SSM、MoE、RAG 等**部件**，
但缺一个把它们整合起来的**端到端 LLM 栈**。选这个方向是为了补齐矩阵中价值最高、
也最能体现"集大成"的那块。

### 架构选择：为什么是 3:1 混合

| 候选 | 结论 | 依据 |
|---|---|---|
| 纯注意力（标准 Transformer） | ❌ | KV cache 随上下文线性增长，长上下文成本不可接受 |
| 纯线性（纯 Mamba / DeltaNet） | ❌ | 在「精确检索某个先前 token」上**结构性失败**：S-NIAH-2 上 Gated DeltaNet 92% vs Mamba2 56%。MiniMax M2 曾全砍线性层、后在 M2.5 回退 —— 这是公开的教训 |
| **3:1 混合** | ✅ | Qwen3-Next、Qwen3.5、Kimi Linear(KDA)、Olmo Hybrid、Mamba-3(ICLR 2026) **全部收敛到这个比例**；Mamba-3 从任务分布证明 ~25% 注意力比例即最优 |

### 关键组件的对标依据

| 组件 | 采用方案 | 被否决的更强/更流行方案 | 理由 |
|---|---|---|---|
| 归一化 | RMSNorm (2019) | LayerNorm | 需减均值+偏置，参数更多、推理更慢，大模型上无质量收益 |
| 位置编码 | RoPE (RoFormer) | ALiBi、绝对位置 | ALiBi 长上下文外推弱于 RoPE+YaRN；绝对位置外推能力差 |
| 前馈 | SwiGLU (Shazeer 2020) | ReLU/GELU FFN | 同等参数下 perplexity 明显更差 |
| 全注意力层 | **MLA** (DeepSeek-V2/V3) | 标准 GQA | 128K 上下文下 KV cache 数十 GB；MLA 把 KV 压到低秩潜向量 |
| 线性层 | **Gated DeltaNet** (ICLR 2025) | Mamba-2、RetNet | GDN = Mamba-2 + delta rule，在联想回忆任务上显著更强 |
| 稀疏化 | **DeepSeek-V3 MoE** | Switch / GShard + 辅助损失 | 辅助损失梯度与主目标对抗：α 小压不住 collapse，α 大质量肉眼可见地掉 |
| 优化器 | **Muon2** (2026-04) | AdamW、Muon | Muon2 在 NS 前插 Adam 二阶矩预条件，3 步达到 Muon 5 步，NS 迭代 −40% |
| 反向实现 | 手写解析反向 | tape 自动微分 | CPU+NumPy 下 tape AD 慢 3~10 倍，且无法利用 DeltaNet 的递推结构 |
| 分词 | 字节级 BPE | Unigram (SentencePiece) | Unigram 需 Viterbi 解码、词表训练非确定，破坏「同 seed 逐位一致」门禁 |
| 后端 | NumPy（仅此一个依赖） | PyTorch / JAX | 目标是「每条反向都能被独立验证、断网可跑」，而非吞吐 |

> **关于 PyTorch 的说明**：PyTorch 在性能上远强于 NumPy，这是事实。
> 本项目选 NumPy 是有意识的目标取舍 —— 可验证性优先。
> 若你的目标是训练生产级模型，应当用 fla + PyTorch；
> 本项目的价值在于架构设计与正确性参照。

---

## 架构说明

### 十层分解

| 层 | 模块 | 采用的世界顶级方案 |
|---|---|---|
| **L0** | 数值内核 | logsumexp 稳定化（LAPACK 传统）、Newton-Schulz 正交化（Muon 公开系数 3.4445/−4.7750/2.0315）、BLAKE2b 派生种子 |
| **L1** | 反向契约 | 显式 `backward` 契约 + 中心差分梯度检验（PyTorch `gradcheck` 同款判据） |
| **L2** | 分词 | 字节级 BPE（Sennrich 2016 + GPT-2/tiktoken 预分词），含 CJK 单字切分改进；WTF-8 字节契约（孤立代理项无损，G18 抓出严格 UTF-8 崩溃缺陷后修复） |
| **L3** | 架构原语 | RMSNorm、RoPE、SwiGLU、GQA、**MLA**（低秩 KV + 解耦 RoPE） |
| **L4** | 序列混合 | **Gated DeltaNet**（delta rule + 遗忘门，沿时间反向递推） |
| **L5** | 稀疏化 | **DeepSeek-V3 MoE**（sigmoid 亲和度 + 无辅助损失动态偏置 + 共享专家） |
| **L6** | 优化器 | **Muon2** + AdamW 混合路由（embedding/LM head 走 AdamW） |
| **L7** | 训练 | 余弦 / WSD 调度、梯度累积、梯度裁剪、bits-per-byte 度量 |
| **L8** | 推理 | KV 状态缓存、per-channel INT8 / group-wise INT4、MTP 投机解码 |
| **L9** | 对齐 | DPO / IPO / SimPO / KTO，统一走 `backward(g_logits=...)` 入口 |
| **L10** | 微指令 + 门禁 | **1024 条可执行微指令**（32 家族 × 32）+ 20 道交付门禁 |

完整推导与逐层详解见 [ARCHITECTURE.md](ARCHITECTURE.md)。

### 模型组装

```
Pre-RMSNorm → Mixer(GDN 或 MLA) → 残差 → Pre-RMSNorm → FFN(MoE 或 Dense) → 残差
```

两种布局（都实现，用基准数据说话）：

| 布局 | 来源 | 做法 |
|---|---|---|
| `interleave` | Qwen3-Next / Kimi Linear | 每 4 层固定 3:1 交错 |
| `edge` | Mamba-3 (ICLR 2026) | 全注意力集中在**首尾**：早层捕获全局结构、晚层做精确检索 |

### 可证门禁（不是测试，是「能力声明」）

| 门禁 | 检验什么 | 实测值 |
|---|---|---|
| G1 确定性 | 同 seed 两次运行参数逐位一致 | diff = 0.000e+00 |
| G2 梯度正确 | 7 个核心模块解析反向 vs 中心差分 | 最大绝对误差 2.47e-08 |
| G3 训练有效 | 固定步数后训练损失显著下降 | 4.2458 → 3.1017（−26.9%） |
| G4 可过拟合 | 极小数据集上损失压到 50% 以下 | 4.1230 → 0.0674 |
| G5 微指令 | 1024 条全部可执行且输出有限 | 1024/1024 |
| G6 MoE 均衡 | 过载专家偏置被压低、欠载被抬高 | −0.0800 < +0.0800 |
| G7 Muon2 | 3 步 NS 对齐度 ≥ Muon 5 步的 90% | 0.9219 vs 0.9742（94.6%） |
| G8 缓存等价 | 混合架构(GDN 递推 + MLA KV)增量 == 整序列重算 | 2.22e-16 |
| G9 量化 | INT8 < 1e-2、INT4 < 2e-1，且 INT8 ≤ INT4 | 5.58e-03 / 1.01e-01 |
| G10 分词 | encode→decode 往返一致 | 4/4（含空格/tab/换行/缩略语） |
| G11 并行形式等价 | Gated DeltaNet 递推 vs 并行前缀扫描（associative scan）两条算法路径逐位一致 | 8.88e-16 |
| G12 chunkwise 等价 | Gated DeltaNet 递推 vs chunkwise 分块并行（块内并行+块间递推，真实 GPU 内核训练形式），含不整除尾块 | 8.88e-16 |
| G13 chunkwise 反向等价 | Gated DeltaNet 物化全量状态反向 vs FlashAttention 式重计算反向（只存 O(T/C) 块入口状态），两条资源/算法路径给同一解析梯度，中心差分独立裁判 | 4.44e-15 |
| G14 投机解码无损 | Leviathan/Chen 拒绝采样恒等：单步输出分布逐点==target（64 组对抗分布解析验证）+ 6000 token 端到端统计 + 接受率∈(0,1)；加速边界如实声明（MTP 搭车结构性 ≈1） | 2.22e-16 / TV=0.0088 |
| G15 投机解码并行验证 | 独立轻量 draft（TinyDraftGDN 复用已门禁 gdn_step）+ γ>1 一次 batch 验证：位置指纹逐位一致（钉死 off-by-one 与状态回滚）+ tokens/forward>1 代数条件 + 混合路径统计无损 | 指纹✅ / 3.0 / TV=0.0147 |
| G16 对齐正确性 | DPO/IPO/SimPO/KTO 首次全机检：解析值（Δ=0→log2、IPO 有限目标 Δ*=τ/2β 系数=0）+ grad_logprob 中心差分 + DPO 端到端参数梯度中心差分 + 方向 chosen↑/rejected↓ + SimPO 长度不变 + ref 零漂移 | 8.9e-11 / 方向✅ |
| G17 MoE 路由不变性 | M2-M5 全机检 + 64 组对抗 fuzz（输入/router/bias 三轴极端）：门控行和==1（曾抓出 Σ+1e-12 加法偏置致归一化塌陷，已修复）、mask 行和恒==K、专家计算量==N·K、垄断场景无丢弃、偏置控制器 clip/maxcv 语义 | 2.22e-16 |
| G18 分词器对抗 fuzz | T5-T11 全机检 + 对抗 fuzz：16 类极端输入往返（含孤立代理项——曾抓出严格 UTF-8 直接 UnicodeEncodeError 崩溃，已修复为 WTF-8 语义，与 Rust String/HF tokenizers 一致）、300 条随机 Unicode、词表单射/merge 图/ranks 插入序/id 闭合、save→load 逐位一致、warm-cold 缓存一致、不动点、预分词无损、空特殊 token 误配置守卫 | 0 失败 |
| G19 优化器不变性 | O1/O3-O7 全机检 + 对抗 fuzz：NS 奇异值精确命题（σ_max 恒落 (0.5,1.5)；5 步+cond≤100 全谱落带——原绝对化宣称被本门禁证伪已修正）、AdamW 解耦 WD 梯度尺度不变、Muon Nesterov 真实 vs Muon2 无变体（曾抓出 `P=P+β·P` 死代码：常数缩放被 NS 归一擦除的假功能）、NaN/Inf 梯度守卫（曾静默污染全参数，修复为 GradScaler 语义跳过）、clip 全局语义、AdamW 首步解析、凸问题单调、确定性 | 0 失败 |
| G20 持久化完整性 | P1-P7 全机检 + 损坏注入：无扩展名路径 save→load→info 闭环（曾抓出 np.savez 自动追加 .npz 致 _sha256_file 找错文件 FileNotFoundError，已修复规范化）、截断 meta 双路径必须 CheckpointError（曾漏出 JSONDecodeError）、损坏权重 sha 失配、sha 同步篡改被指纹抓住、原子落盘（模拟崩溃无半成品+旧件完好，对齐 safetensors tmp+rename）、空参数/怪异参数名/dtype 保持、保存确定性+失配检出 | 0 失败 |

**门禁的态度：失败就如实报失败，不允许放宽阈值凑绿。**

---

## 可量化验收指标

16 项指标由 `python scripts/acceptance.py` 自动执行（约 50 秒），
判据与实测值见 [SPEC.md](SPEC.md) 第 4 节。当前 **16/16 达标**。

| ID | 指标 | 判据 | 实测 |
|---|---|---|---|
| A1 | 解析反向正确性 | 最大绝对误差 < 1e-6 | **6.21e-09**（8/8 模块） |
| A2 | 训练确定性 | diff == 0（逐位） | **0.0e+00** |
| A3 | 训练有效性 | 300 步相对下降 > 10% | **16.6%** |
| A4 | 过拟合能力 | 降到初始 < 50% | **2.7%** |
| A5 | 微指令覆盖 | 1024/1024 | **1024/1024** |
| A6 | KV cache 节省 | ≥ 80% | **83.3%** |
| A7 | INT8 精度 | 权重误差 < 1% | **5.37e-03** |
| A8 | INT4 退化 | \|退化\| < 15% | **+9.11%**（g8 对照 +5.04%） |
| A9 | 缓存等价性 | < 1e-10 | **1.11e-16** |
| A10 | 分词往返 | 9/9 | **9/9** |
| A11 | checkpoint 往返 | 逐位一致 + 损坏拦截 | **均 True** |
| A12 | 推理吞吐 | > 50 tok/s | **465 tok/s** |
| A13 | 运行时依赖 | == 1 | **`['numpy']`** |
| A14 | 单元测试 | 41/41 | **41/41** |
| A15 | 交付门禁 | 20/20 | **20/20** |
| A16 | 断网可运行 | 0 处网络依赖 | **0** |

三条判据的设定依据（A3 为何是 10%、A7 为何用权重误差、A8 为何放宽到 15%）
在 [SPEC.md](SPEC.md) 中逐条写明，不是拍脑袋定的阈值。

```bash
python scripts/acceptance.py            # 全量（约 50 秒）
python scripts/acceptance.py --quick    # 快速档
```

---

## 快速开始

```bash
git clone https://github.com/CJX0712/tianji-ai.git
cd tianji-ai
pip install -r requirements.lock.txt

python scripts/quickstart.py          # 最小示例，约 10 秒
```

预期输出（节选）：

```
1. 分词   词表 432，25680 token，字节压缩比 3.73x
2. 模型   4 层 布局=LLLA (A=MLA, L=GatedDeltaNet)，参数 252,032
4. 训练   30 步 2.4s，loss 6.8636 -> 0.2036
6. 生成   结果='the quick brown fox jumps over the lazy dog. ...'
7. 门禁   20/20 通过
✅ 全部通过
```

Python API：

```python
from tianji import TianjiConfig, TianjiModel, init_params

cfg = TianjiConfig(vocab_size=512, d_model=96, n_layers=8)
model = TianjiModel(cfg, init_params(cfg, seed=0))

out = model.forward(tokens, targets)   # (B, T) int64
grads = model.backward()               # 参数梯度字典
```

---

## 复现步骤

### 方案 A：一键脚本（推荐，跨平台，不需要 Docker）

```bash
# Linux / macOS
./scripts/reproduce.sh              # 完整：建环境 → 测试 → 门禁 → 示例 → 训练 → 推理
./scripts/reproduce.sh --quick      # 快速：只跑测试 + 门禁 + 最小示例

# Windows
scripts\reproduce.bat
scripts\reproduce.bat --quick
```

脚本会自建 `.venv`、安装锁定依赖、依次跑完 6 个步骤，全程不污染系统环境。

### 方案 B：手动 pip

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt                    # 锁定版本，保证逐位一致

make verify-all        # 测试 + 门禁 + 最小示例（交付自查）
```

### 方案 C：Docker（可选）

```bash
docker build -t tianji:latest .
docker run --rm tianji:latest gates      # 跑 20 道门禁
docker run --rm tianji:latest test       # 跑 76 个测试

docker compose run --rm tianji train     # 训练，产物落在 ./runs
docker compose run --rm tianji generate  # 推理
```

> **网络前提（如实说明）**：Docker 路径需要能访问 Docker Hub 拉取 `python:3.11-slim`。
> 本仓库的开发机因网关限制（`auth.docker.io` 返回 Bad Gateway）**未能完成镜像构建验证**；
> Dockerfile 本身已解析通过，失败发生在拉取基础镜像阶段。
> 若你的环境同样受限，请直接用方案 A 或 B（等价且更快），或在 Docker Desktop 的
> `Settings → Docker Engine` 中配置 `registry-mirrors` 后重试。
>
> 选 `python:3.11-slim` 而非 alpine 的理由：NumPy 在 alpine(musl) 上需自行编译
> OpenBLAS，构建从 30 秒变成 10 分钟，且不同机器可能产出不同浮点结果，破坏可复现性。

### 方案 D：完整训练 + 推理

```bash
python scripts/train.py --steps 300 --out runs/default       # 内置语料
python scripts/train.py --data my.txt --steps 500            # 自己的语料

python scripts/generate.py --ckpt runs/default/model.npz \
    --tokenizer runs/default/tokenizer.txt --prompt "the quick brown" -n 60
```

checkpoint 加载时**强制校验 sha256 与参数指纹**，损坏文件直接报错，不会静默生成乱码。

### 一键复现全部基准

```bash
python bench/run_bench.py --steps 120 --seed 0     # 约 2 分钟，写 bench/results/bench.json
```

### 所有命令（Makefile）

```bash
make help          # 查看全部目标
make test          # 76 个单元测试
make gates         # 20 道交付门禁
make bench         # 完整基准
make quickstart    # 最小示例
make train         # 训练
make generate      # 推理
make docker-build  # 构建镜像
make docker-test   # 容器内跑测试+门禁
make verify-all    # 交付前自查
```

---

## 性能对比数据

完整数据见 [BENCHMARKS.md](BENCHMARKS.md)。**实验条件先说清楚**：
NumPy 2.5.4 纯 CPU、合成 2 阶马尔可夫语料（vocab=64）、d_model=64 / 6 层 / MoE(4专家 top-2)、
120 步、batch=8、seq=32。

### B1 · 架构对比

| 架构 | 参数量 | 训练 loss | **验证 loss** | ms/步 |
|---|---:|---:|---:|---:|
| **Hybrid 3:1（GDN + MLA）** | 396,544 | 4.3812 | **4.2236** | 157 |
| 全注意力（MLA） | 364,224 | **4.3451** | 4.2933 | 88 |
| 全 GDN（无注意力层） | 396,544 | 4.3564 | 4.2305 | 156 |

混合架构验证损失比全注意力低 **1.6%**。差异幅度小、单一 seed，**不足以断言质量更优** ——
真正的收益在下面 B5。

### B5 · KV cache（本架构最硬的收益）

| 上下文 T | 混合 3:1 | 全注意力 | 节省 |
|---:|---:|---:|---:|
| 128 | 192 KB | 1,152 KB | **83.3%** |
| 2048 | 3,072 KB | 18,432 KB | **83.3%** |
| 8192 | 12,288 KB | 73,728 KB | **83.3%** |

6 层中只有 1 层需要 KV cache，节省 = 1 − 1/6 = **83.3%**，与上下文长度无关。
**对照工业界**：Kimi Linear（48B，KDA + MLA 混合 3:1）报告 KV cache −75%、
百万上下文下解码最高 6× 加速。本项目与之同量级（层数比例不同）。

### B4 · 量化

| 精度 | 权重相对误差 | perplexity | 退化 |
|---|---:|---:|---:|
| FP64 基线 | — | 66.14 | — |
| INT8 (per-channel) | 5.744e-03 | 66.50 | **+0.55%** |
| INT4 (group-wise, g=32) | 9.439e-02 | 66.78 | **+0.97%** |

### B3 · 优化器（**负面结果，如实报告**）

| 优化器 | 训练 loss | 验证 loss |
|---|---:|---:|
| AdamW (lr 3e-3) | **4.1203** | 4.2440 |
| Muon (NS 5 步) | 4.4775 | **4.1777** |
| Muon2 (NS 3 步) | 4.3812 | 4.2236 |

**Muon2 在本项目规模下验证损失略逊于 Muon（4.2236 vs 4.1777）**，
与 Muon2 论文（60M~15.5B 规模）的结论**不一致**。两条诚实解释：

1. **规模**：论文结论建立在千万到百亿参数、数千步训练上；本项目 40 万参数、120 步，
   优化器差异淹没在噪声里。
2. **实现保真度**：本项目只实现论文核心改动，未复现其分布式通信与 NS 步数自适应调度。
   门禁 G7 实测 3 步对齐度是 5 步的 **94.6%**，未达论文声称的「3 步 ≥ 5 步」。

默认仍是 Muon2（少 40% NS 迭代）；追求本规模极限质量可切 Muon（`--optim muon`）。

### 与现有最优方案的定位对比

| 维度 | **本项目** | nanoGPT | llama.cpp | fla (flash-linear-attention) |
|---|---|---|---|---|
| 定位 | 可验证的混合架构**全栈** | 教学级 GPT 实现 | 生产推理引擎 | 线性注意力 GPU 内核库 |
| 依赖 | **仅 NumPy** | PyTorch | C++（无 Python 依赖） | Triton + PyTorch |
| 架构 | **3:1 混合**（GDN + MLA + MoE） | 纯注意力 | 推理（多架构） | 内核层（不含完整模型） |
| 训练 | ✅ CPU 可训 | ✅ GPU | ❌ | ✅ GPU |
| 量化 / 投机解码 | ✅ | ❌ | ✅ | ❌ |
| 对齐（DPO 等） | ✅ | ❌ | ❌ | ❌ |
| **可证门禁** | **✅ 20 道** | ❌ | ❌ | 部分 |
| 吞吐 | 低（NumPy CPU） | 高 | 极高 | 极高 |

**诚实的结论**：本项目**不追求吞吐**。它的对标对象是「正确性可独立验证的研究级实现」，
生产场景应直接用 fla + PyTorch 或 llama.cpp。本项目的价值在于：
把 2026 年的混合架构设计用**最少依赖**完整实现一遍，并把每一条反向都用有限差分钟住。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| **[SPEC.md](SPEC.md)** | 项目定位、核心能力、技术选型对比、**16 项可量化验收指标**与判据依据 |
| **[docs/API.md](docs/API.md)** | **完整 API 说明**：L0–L10 全部公开接口、参数、返回值与示例 |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | 十层架构详解、数学推导、逐层选型与被否决方案、反向传播顺序留档 |
| **[BENCHMARKS.md](BENCHMARKS.md)** | 6 组基准数据与诚实解读（含一条负面结果） |
| [README.md](README.md) | 本文件：总览、快速开始、复现、性能对比、引用清单 |

---

## 引用来源清单

按架构层级组织，全部为公开成果。实现时直接采用其算法与公式，未重新发明。

### 数值与优化

- **Muon**: Keller Jordan et al., *Muon: An optimizer for hidden layers in neural networks*, 2024（NanoGPT speedrun 社区）。Newton-Schulz 系数 (3.4445, −4.7750, 2.0315)。
- **Muon2**: Zhang et al., *Muon²: Boosting Muon via Adaptive Second-Moment Preconditioning*, arXiv 2026-04。
- **AdamW**: Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019。
- **梯度裁剪**: Pascanu et al., *On the difficulty of training recurrent neural networks*, ICML 2013。
- **幂迭代 / 谱范数**: Golub & Van Loan, *Matrix Computations*（第 4 版）第 7 章。

### 架构原语

- **RMSNorm**: Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019。
- **RoPE**: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021。
- **SwiGLU**: Shazeer, *GLU Variants Improve Transformer*, 2020。
- **GQA**: Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*, EMNLP 2023。
- **MLA**: DeepSeek-AI, *DeepSeek-V2*, 2024；*DeepSeek-V3 Technical Report*, 2024。

### 序列混合

- **Delta Rule**: Schlag et al., *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021。
- **Gated DeltaNet**: Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025（NVIDIA）。
- **GLA**: Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training*, ICML 2024。
- **Mamba / Mamba-2**: Gu & Dao, *Mamba*, 2023；*Transformers are SSMs*, ICML 2024。
- **Mamba-3**: Gu, Dao et al., ICLR 2026（3:1 比例的理论最优性、注意力层 placement）。

### 稀疏化

- **MoE 基础**: Shazeer et al., *Outrageously Large Neural Networks*, 2017。
- **Switch / GShard**: Fedus et al., JMLR 2022；Lepikhin et al., ICLR 2021。
- **ST-MoE（router z-loss）**: Zoph et al., 2022。
- **无辅助损失负载均衡**: DeepSeek-V3 Technical Report, 2024。

### 分词

- **BPE**: Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016。
- **GPT-2 字节级 BPE / 预分词**: Radford et al., 2019；OpenAI `tiktoken`。

### 推理

- **投机解码**: Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023；Chen et al., 2023。
- **MTP 作 draft**: DeepSeek-V3 Technical Report, 2024。
- **AWQ**: Lin et al., *AWQ: Activation-aware Weight Quantization*, MLSys 2024。
- **GPTQ**: Frantar et al., ICLR 2023。
- **per-channel INT8 (Q8_0)**: llama.cpp / GGML。
- **Top-p 采样**: Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020。

### 对齐

- **DPO**: Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023。
- **IPO**: Azar et al., *A General Theoretical Paradigm for Understanding RLHF*, ICML 2024。
- **SimPO**: Meng et al., *SimPO: Simple Preference Optimization*, 2024。
- **KTO**: Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, ICML 2024。

### 工程参照

- **Hybrid 3:1 收敛**: Qwen3-Next / Qwen3.5（Alibaba）、Kimi Linear（Moonshot AI，KDA）、Olmo Hybrid（AI2，2026）。
- **反向契约**: PyTorch `torch.autograd.Function` 与 `torch.autograd.gradcheck`。
- **门禁哲学**: 作者前作 `tianshu-ai`（天枢）——「失败就如实报失败，不允许放宽阈值凑绿」。

---

## 仓库结构

```
tianji-ai/
├── tianji/                     # 核心库（14 个模块）
│   ├── l0_numerics.py         数值内核
│   ├── l1_contract.py         反向契约 + 中心差分梯度检验
│   ├── l2_tokenizer.py        字节级 BPE
│   ├── l3_primitives.py       RMSNorm / RoPE / SwiGLU / GQA / MLA
│   ├── l4_deltanet.py         Gated DeltaNet（沿时间反向递推）
│   ├── l5_moe.py              MoE + 无辅助损失动态偏置
│   ├── l6_optimizer.py        AdamW / Muon / Muon2
│   ├── l7_train.py            调度 / 数据管道 / 训练循环
│   ├── l8_infer.py            KV 缓存 / 量化 / 投机解码
│   ├── l9_align.py            DPO / IPO / SimPO / KTO
│   ├── l10_microops.py        1024 条微指令（32 家族）
│   ├── model.py               3:1 混合模型组装
│   ├── persist.py             checkpoint 保存/加载：sha256+指纹校验、路径规范化、原子落盘
│   ├── data.py                内置离线语料
│   ├── verify.py              20 道交付门禁
│   └── cli.py                 命令行入口
├── scripts/
│   ├── reproduce.sh             一键复现（Linux / macOS，无需 Docker）
│   ├── reproduce.bat            一键复现（Windows，无需 Docker）
│   ├── quickstart.py            最小可运行示例（CI 自检入口）
│   ├── train.py                 训练脚本
│   ├── generate.py              推理脚本
│   ├── quick_gradcheck.py       模块梯度冒烟
│   └── test_model.py            模型配置冒烟
├── docs/
│   └── API.md                   完整 API 说明（L0–L10 全部接口）
├── tests/test_all.py          76 个单元测试
├── bench/                     基准脚本 + 结果（bench.json）
├── Dockerfile                 一键环境（python:3.11-slim）
├── docker-compose.yml         挂载 ./runs 的训练/推理
├── Makefile                   全部常用命令
├── pyproject.toml             PEP 621 元数据 + pytest 配置
├── requirements.txt           依赖声明
├── requirements.lock.txt      锁定版本（逐位可复现）
├── SPEC.md                    规格：定位 / 选型 / 验收指标
├── README.md / ARCHITECTURE.md / BENCHMARKS.md
└── LICENSE                    MIT License
```

---

## 已知边界（不掩饰）

- **语料是确定性合成的 2 阶马尔可夫**，不是真实文本。基准的绝对数字
  **不可与真实语料上的任何结果横向比较**，只用于同实验内不同配置的相对比较。
- **GDN 的 O(1) 递推增量解码已实现并纳入门禁**：每个新 token 只做一次秩-1 状态更新
  （`S_t = α(S_{t−1} − β P kᵀ) + β v kᵀ`），操作序列与 `GatedDeltaNet.forward` 循环体逐位一致。
  门禁 G8（现为 3:1 混合架构 GDN+MLA）可证「递推增量 ≡ 整序列重算」到机器精度
  （最大差 **2.22e-16**），`generate(use_cache=True) == generate(use_cache=False)`。
  因此本条已知边界已关闭。
- **GDN 前向 + 反向均已多路径互证到机器精度**：
  - 前向三种形式：顺序递推 `forward`、并行前缀扫描 `gdn_parallel`、chunkwise 分块并行
    `gdn_chunkwise`（块内并行 + 块间递推，真实 GPU 内核的分块本质）。由 G11 / G12 机检，
    最大差 **8.88e-16**，含不整除尾块。
  - 反向两条路径：物化全量状态 `GatedDeltaNet.backward`（存 S_all，形状 B×H×(T+1)×D×D）
    vs FlashAttention 式重计算 `gdn_chunkwise_backward`（只存 O(T/C) 个块入口状态、块内即时
    重算、块间用伴随传输 G）。两条不同资源/算法路径给同一解析梯度，由 **G13** 机检，
    最大差 **4.44e-15**；再用中心差分 `gradcheck`（完全独立的裁判）确认，见 G2。
  论文原版的 WY 表示 + UT transform 仍**未实现** —— 它需求 (I − 三角矩阵)⁻¹，
  仅为 GPU 设计，CPU 上数值更脆；其算子定义保留在 L10 的 `scan` 家族中。
- **MTP 是轻量版**：用单个线性投影替代 DeepSeek-V3 原版的完整 Transformer 层。
- **没有 GPU 路径**：纯 NumPy CPU 实现，模型规模在百万参数以下。
- **1024 条微指令中**，17 个家族（`FAMILY_KIND = "sweep"`）的成员是同一算子的参数扫描，
  用于覆盖数值边界，不是 1024 个语义独立算子。这个区分在代码里显式标注。
- **投机解码两条路径（均已机检）**：MTP 搭车（`draft_model=None`，γ=1）——draft 分布来自主模型 forward 顺带返回的 MTP 头，结构性 ≈1，无损性由 **G14** 机检（解析恒等 2.22e-16 + 统计 TV=0.0088）；独立轻量 draft（`TinyDraftGDN` 复用已门禁的 `gdn_step`，γ>1）——γ 个 draft 一次 batch 并行验证，每轮恰 1 次主干 forward，**实测 tokens/forward ≈ 3.2**，位置指纹/代数条件/统计无损由 **G15** 机检。MTP logits 已修复为推理期导出（旧版锁在 targets 块内导致 draft 退化为恒接受空壳）。

---

## 交付前自查（已执行）

| 检查项 | 结果 |
|---|---|
| 单元测试 | ✅ 41 / 41 通过 |
| 交付门禁 | ✅ 12 / 12 全绿 |
| **验收指标** | ✅ **16 / 16 达标**（`scripts/acceptance.py`，47s） |
| 微指令 | ✅ 1024 / 1024 可执行 |
| 最小可运行示例 | ✅ `scripts/quickstart.py` 5.4s 全通过 |
| 占位符扫描 | ✅ 无 TODO / FIXME / 未实现桩（2 处 `NotImplementedError` 为抽象基类方法） |
| 缺失文件 | ✅ 源码、测试、基准、Docker、锁依赖、文档、许可证齐全 |
| 模型梯度 | ✅ 解析反向 vs 中心差分，最大绝对误差 2.47e-08 |
| checkpoint 往返 | ✅ 保存+加载逐位一致，sha256 校验通过 |
| 断网可跑 | ✅ 内置语料，无任何下载依赖 |

---

## 许可证与署名

MIT License，详见 [LICENSE](LICENSE)。

**作者：晨星（Chenxing）** · GitHub: [@CJX0712](https://github.com/CJX0712)

本项目遵循「零重写」原则：所有算法与架构设计均直接采用上述公开成果，
并在源码注释中标注来源与被否决的替代方案。若发现某处有更强的现成方案，
欢迎提 Issue —— 按项目原则应立即替换。
