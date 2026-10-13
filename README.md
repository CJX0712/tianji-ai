# 天玑 TIANJI · 混合架构语言模型全栈

> **Gated DeltaNet × MLA × MoE × Muon2 × 投机解码**
> 零重写 · 集大成 · 闭环交付 · 可证门禁
>
> 作者：**晨星（Chenxing）** · 许可证：Apache-2.0

---

## 这是什么

一个**十层架构**的语言模型全栈，从数值内核到偏好对齐，每一层都站在世界顶级公开成果之上。
它不追求"从零发明"，而是把 2026 年已被工业界验证的最优解**正确地组合起来**，
并用**可复现的基准数据**与**可证门禁**证明它真的能用。

核心架构是 **3:1 混合**：每 4 层里 3 层 Gated DeltaNet（线性）、1 层 MLA（全注意力）。
这是 Qwen3-Next / Qwen3.5、Kimi Linear、Olmo Hybrid 在 2026 年共同收敛到的布局。

---

## 状态

| 项目 | 数值 |
|---|---|
| 交付门禁 | **10 / 10 全绿** |
| 单元测试 | **39 / 39 通过** |
| 微指令 | **1024 / 1024 条可执行**（32 家族） |
| 梯度检验 | 全部模块解析反向 vs 中心差分，最大绝对误差 **2.47e-08** |
| 依赖 | 仅 NumPy（+ pytest 用于测试） |
| 硬件 | 纯 CPU，无 GPU 要求 |

---

## 快速开始

```bash
pip install numpy pytest

# 跑全部测试（约 5 秒）
python -m pytest tests/ -q

# 跑交付门禁（10 道）
python -c "import sys; sys.path.insert(0,'.'); \
           from tianji.verify import run_all_gates; print(run_all_gates().table())"

# 跑完整基准
python bench/run_bench.py --steps 120
```

```python
from tianji import TianjiConfig, TianjiModel, init_params, HybridOptimizer, Trainer, DataLoader

cfg = TianjiConfig(vocab_size=512, d_model=96, n_layers=8)
model = TianjiModel(cfg, init_params(cfg, seed=0))

out = model.forward(tokens, targets)     # tokens/targets: (B, T) int64
grads = model.backward()                 # 参数梯度字典
```

---

## 十层架构一览

| 层 | 模块 | 采用的世界顶级方案 |
|---|---|---|
| **L0** | 数值内核 | logsumexp 稳定化（LAPACK 传统）、Newton-Schulz 正交化（Muon 公开系数）、BLAKE2b 派生种子 |
| **L1** | 反向契约 | 显式 `backward` 契约 + 中心差分梯度检验（PyTorch `gradcheck` 同款判据） |
| **L2** | 分词 | 字节级 BPE（Sennrich 2016 + GPT-2/tiktoken 预分词），含 CJK 单字切分改进 |
| **L3** | 架构原语 | RMSNorm、RoPE、SwiGLU、GQA、**MLA**（DeepSeek-V2/V3 低秩 KV + 解耦 RoPE） |
| **L4** | 序列混合 | **Gated DeltaNet**（Yang et al. ICLR 2025，delta rule + 遗忘门） |
| **L5** | 稀疏化 | **DeepSeek-V3 MoE**（sigmoid 亲和度 + 无辅助损失动态偏置 + 共享专家） |
| **L6** | 优化器 | **Muon2**（2026-04，Adam 二阶矩预条件 + NS 正交化）+ AdamW 混合路由 |
| **L7** | 训练 | 余弦 / WSD 调度、梯度累积、梯度裁剪、bits-per-byte 度量 |
| **L8** | 推理 | KV 状态缓存、per-channel INT8 / group-wise INT4、MTP 投机解码 |
| **L9** | 对齐 | DPO / IPO / SimPO / KTO，统一走 `backward(g_logits=...)` 入口 |
| **L10** | 微指令+门禁 | **1024 条可执行微指令**（32 家族 × 32）+ 10 道交付门禁 |

---

## 为什么是这些选择（关键决策依据）

完整对比见 [ARCHITECTURE.md](ARCHITECTURE.md)。这里列出最关键的四个：

**1. 为什么是 3:1 混合，而不是纯注意力或纯线性**

纯注意力的 KV cache 随上下文线性增长；纯线性在"精确检索某个先前 token"上结构性失败
（S-NIAH-2 上 Gated DeltaNet 92% vs Mamba2 56%）。MiniMax M2 曾全砍线性层、后又回退，
是公开的教训。2026 年 Qwen3-Next、Kimi Linear、Olmo Hybrid、Mamba-3 全部收敛到 3:1，
本项目把 `layout` 做成可选项，用基准数据说话。

**2. 为什么 MoE 用"无辅助损失"的偏置调节**

辅助损失把"均衡"写进损失函数，于是均衡梯度与语言建模梯度在同一条反向路径上互相拉扯：
α 小了压不住 collapse，α 大了质量肉眼可见地掉。DeepSeek-V3 把均衡搬出损失函数，
改成只对路由生效的控制器（偏置 bᵢ），梯度路径彻底解耦 —— 这是架构级解法，不是调参技巧。

**3. 为什么优化器默认 Muon2 而不是 Muon**

Muon 的 Newton-Schulz 迭代要在"病态的动量矩阵"上做极近似。Muon2 在 NS 之前先插一层
Adam 式二阶矩预条件，把谱结构修好，于是 **3 步 NS 就达到 Muon 5 步的对齐度**
（本项目门禁 G7 实测：0.9742 vs 0.9219，判据是 3 步不低于 5 步的 90%）。

**4. 为什么手写反向而不是用自动微分**

CPU + NumPy 后端下 tape AD 慢 3~10 倍（每个算子一次 tape 记录 + dict 查找），
且无法利用 Gated DeltaNet 的递推结构（应当沿时间反向递推，而非逐节点展开）。
代价是每个模块必须手写反向 —— 用 L1 的中心差分检验兜底，不靠人眼。

---

## 可证门禁（不是测试，是"能力声明"）

| 门禁 | 检验什么 |
|---|---|
| G1 确定性 | 同 seed 两次运行参数**逐位一致** |
| G2 梯度正确 | 7 个核心模块解析反向 vs 中心差分 |
| G3 训练有效 | 固定步数后训练损失显著下降 |
| G4 可过拟合 | 极小数据集上损失压到 50% 以下（证明反向真在传播） |
| G5 微指令 | 1024 条全部可执行且输出有限 |
| G6 MoE 均衡 | 过载专家偏置被压低、欠载专家被抬高 |
| G7 Muon2 | 3 步 NS 的对齐度 ≥ Muon 5 步的 90% |
| G8 缓存等价 | KV cache 增量解码 == 整序列重算（2.22e-16） |
| G9 量化 | INT8 误差 < 1e-2、INT4 < 2e-1，且 INT8 ≤ INT4 |
| G10 分词 | encode→decode 往返完全一致（含多空格/tab/换行/缩略语） |

**门禁的态度：失败就如实报失败，不允许放宽阈值凑绿。**
报告同时给出 status 与实测值，任何人都能复核。

---

## 开发过程中真实修掉的 bug（留档，供后来者避坑）

这些都不会崩溃、形状全对，只是"永远训不好"——是最阴险的一类：

| Bug | 症状 | 根因 |
|---|---|---|
| Pre-Norm 反传顺序颠倒 | 梯度形状全对，数值全错 | 把 Norm 的 backward 放在子层**，正确顺序是先过子层再过 Norm** |
| 交叉熵反向用了 log-softmax | 数值差分完全对不上 | 归一化指数族的 Jacobian 是 `I − softmax`，不是 log-softmax |
| SwiGLU 的 up 分支梯度 | 用 `silu(h)` 而非 `silu(gate_raw)` | h 已含乘法，导数必须作用在 gate 的**线性输出**上 |
| MoE 的 `dWd` | 用了 `dH` 而非激活 `h` | down 投影的权重梯度是 `Gyᵀ @ h` |
| 共享专家的 `dx` 多一项 | 多加了 `dH @ sWdᵀ` | 中间隔着 SwiGLU，梯度不能直传回 x |
| KV cache 的 RoPE 位置 | 增量解码用了**局部位置 0** | 必须传绝对位置 `pos`，否则长文本质量崩掉但不报错 |
| 矩形因果掩码 | 左对齐会让最新 token 看不到历史 | 必须**右对齐**：`j ≤ i + (Tkv − Tq)` |
| 梯度检验判据 | 正确反向被判 FAIL（比值 1.0000） | atol 必须用**全局**梯度尺度，不能按单个元素量级设 |

---

## 目录结构

```
tianji-ai/
├── tianji/
│   ├── l0_numerics.py     数值内核
│   ├── l1_contract.py     反向契约 + 梯度检验
│   ├── l2_tokenizer.py    字节级 BPE
│   ├── l3_primitives.py   RMSNorm / RoPE / SwiGLU / GQA / MLA
│   ├── l4_deltanet.py     Gated DeltaNet
│   ├── l5_moe.py          MoE + 无辅助损失负载均衡
│   ├── l6_optimizer.py    AdamW / Muon / Muon2
│   ├── l7_train.py        调度 / 数据 / 训练循环
│   ├── l8_infer.py        缓存 / 量化 / 投机解码
│   ├── l9_align.py        DPO / IPO / SimPO / KTO
│   ├── l10_microops.py    1024 条微指令
│   ├── model.py           模型组装（3:1 混合）
│   └── verify.py          10 道交付门禁
├── tests/test_all.py      39 个测试
├── bench/run_bench.py     基准（产出 results/bench.json）
├── ARCHITECTURE.md        架构详解与选型依据
├── BENCHMARKS.md          基准数据
└── LICENSE                Apache-2.0
```

---

## 已知的边界（不掩饰）

- **语料是确定性合成的 2 阶马尔可夫**，不是真实文本。因此基准的绝对数字**不可与真实语料上的任何结果横向比较**，只用于同实验内不同配置的相对比较。
- **GDN 的推理态仍是整序列重算**，尚未实现 O(1) 递推增量解码。因此 KV cache 门禁（G8）只对全注意力配置验证。
- **MTP 是轻量版**：用单个线性投影替代 DeepSeek-V3 原版的完整 Transformer 层。
- **没有 GPU 路径**：全部是 NumPy CPU 实现，模型规模在百万参数以下。
- **1024 条微指令中**，部分家族（`sweep` 型，见 `FAMILY_KIND`）的成员是同一算子的参数扫描，用于覆盖数值边界，不是 1024 个语义独立的算子。这个区分在代码里显式标注。

---

## 许可证

Apache License 2.0。详见 [LICENSE](LICENSE)。

作者：**晨星（Chenxing）**
