# 天玑 TIANJI · API 说明

> 完整公开 API 参考。所有函数/类均可从顶层 `tianji` 包直接导入。
> 约定：张量均为 `numpy.ndarray`，浮点默认 `float64`；整数 token 为 `int64`。

```python
import tianji
print(tianji.__version__, tianji.__author__)   # 1.0.0  晨星 (Chenxing)
```

---

## 目录

- [L0 数值内核](#l0-数值内核)
- [L1 反向契约](#l1-反向契约)
- [L2 分词](#l2-分词)
- [L3 架构原语](#l3-架构原语)
- [L4 Gated DeltaNet](#l4-gated-deltanet)
- [L5 MoE](#l5-moe)
- [L6 优化器](#l6-优化器)
- [L7 训练](#l7-训练)
- [L8 推理](#l8-推理)
- [L9 对齐](#l9-对齐)
- [L10 微指令](#l10-微指令)
- [模型](#模型)
- [持久化](#持久化)
- [数据](#数据)
- [门禁](#门禁)
- [命令行](#命令行)

---

## L0 数值内核

```python
from tianji import (logsumexp, softmax, log_softmax, newton_schulz,
                    spectral_norm, rms, topk, clip_grad_norm,
                    stable_hash, spawn_seed, rng)
```

### `logsumexp(x, axis=-1, keepdims=False) -> np.ndarray`
数值稳定的 `log Σ exp(xᵢ)`。平移 `m = max(xᵢ)`，避免溢出。

```python
logsumexp(np.array([[1.0, 2.0, 800.0]]))   # -> [800.0]，不会 inf
```

### `softmax(x, axis=-1) -> np.ndarray`
平移不变，行和恒为 1。

### `log_softmax(x, axis=-1) -> np.ndarray`
等价于 `x - logsumexp(x, keepdims=True)`，数值更稳。

### `newton_schulz(G, steps=5, eps=1e-7, coeffs=(3.4445, -4.7750, 2.0315)) -> np.ndarray`
把矩阵推向最近半正交矩阵 `UVᵀ`。`G` 必须二维。系数取自 Muon 公开实现。
输出奇异值落在 (0.5, 1.5)。

### `spectral_norm(A, iters=16, eps=1e-12) -> float`
幂迭代求最大奇异值（比完整 SVD 快，Newton-Schulz 每步都要用）。

### `rms(x, axis=-1, eps=1e-8) -> np.ndarray`
Root-mean-square，RMSNorm 的分母。

### `topk(a, k, axis=-1) -> (values, indices)`
Top-K，按降序返回。`k` 超出维度时自动截断。

### `clip_grad_norm(grads: dict, max_norm: float) -> (grads, norm)`
全局梯度范数裁剪，返回裁剪后的梯度与原范数。

### `stable_hash(obj) -> int` / `spawn_seed(master, *tags) -> int` / `rng(seed)`
确定性工具。`spawn_seed` 用 BLAKE2b 派生互不相关的子流种子
（避免 `seed + i` 的已知相关结构）。`rng` 返回 `np.random.Generator`（PCG64）。

---

## L1 反向契约

```python
from tianji import Module, Ctx, gradcheck, GradReport
```

### `class Module`
所有可微模块的基类。契约：

| 方法 | 签名 | 说明 |
|---|---|---|
| `forward` | `(ctx: Ctx, *args) -> np.ndarray` | 不得修改入参，中间量一律存 `ctx` |
| `backward` | `(ctx: Ctx, grad_out) -> tuple` | 返回与 `forward` 等长同形的梯度元组；不可微位置返回 `None` |

类属性 `differentiable: tuple[int, ...]` 声明哪些位置是可微输入。

### `class Ctx(dict)`
前向传给反向的暂存区（dict 子类）。支持 `ctx.put(**kw)`。

### `gradcheck(module, args, *, seed=0, eps=1e-5, tol=1e-5, name=None) -> (GradReport, grads, ctx)`
中心差分梯度检验。判据（PyTorch 同款）：

```
|a − n| ≤ atol + rtol·|n|，其中 atol = 1e-7 × 全局梯度尺度
```

> **为什么 atol 用全局尺度**：中心差分噪声 ≈ `eps_machine·|loss|/eps`，与元素大小无关。
> 按元素量级设 atol，会让梯度只有 1e-8 的参数被噪声误判为 FAIL。

```python
rep, grads, ctx = gradcheck(RMSNorm(), [x, w], seed=1)
print(rep.passed, rep.max_abs_err)
```

### `class GradReport`
字段：`name, passed, max_abs_err, max_rel_err, tol, detail, per_input`。

---

## L2 分词

```python
from tianji import BPE, pretokenize
```

### `class BPE(vocab_size=512, special=("<|endoftext|>",))`

| 方法 | 说明 |
|---|---|
| `train(text: str \| Iterable[str], verbose=False) -> BPE` | 从语料训练合并表（确定性） |
| `encode(text: str) -> list[int]` | 编码 |
| `decode(ids: Sequence[int]) -> str` | 解码 |
| `save(path) / classmethod load(path)` | 词表落盘 / 读取（纯文本，可 diff） |
| `size` | 实际词表大小 |

```python
bpe = BPE(vocab_size=512).train(corpus)
ids = bpe.encode("the quick brown fox")
assert bpe.decode(ids) == "the quick brown fox"
```

**性质**：`decode(encode(s)) == s` 往返一致（含多空格 / tab / 换行 / 前导尾随空白 / 英文缩略语）。

### `pretokenize(text, cjk_split=True) -> list[str]`
GPT-2 预分词的零依赖等价实现。CJK 单字成块（对 GPT-2 原正则的改进）。

---

## L3 架构原语

```python
from tianji import (Linear, RMSNorm, RoPE, SwiGLU, CausalAttention,
                    CrossEntropy, rope_tables, causal_mask, causal_mask_rect,
                    silu, gelu)
```

| 类 | `forward` 签名 | 说明 |
|---|---|---|
| `Linear` | `(ctx, x, W, b=None)` | `y = x@Wᵀ+b`；`W` 为 `(out, in)`（PyTorch 布局） |
| `RMSNorm` | `(ctx, x, w)` | `y = (x/RMS(x)) ⊙ w` |
| `RoPE` | `(ctx, x, cos, sin)` | 旋转位置编码；`cos/sin` 不可微 |
| `SwiGLU` | `(ctx, x, W_gate, W_up, W_down)` | `y = (SiLU(xW_gᵀ) ⊙ xW_uᵀ) W_dᵀ` |
| `CausalAttention` | `(ctx, q, k, v)` | `q:(B,H,T,Dq)`、`k:(B,Hkv,T,Dq)`、`v:(B,Hkv,T,Dv)` |
| `CrossEntropy` | `(ctx, logits, targets)` | 因果 LM 损失，`ignore_index = -1` |

**CausalAttention 的关键泛化**：允许 `Dq ≠ Dv`（MLA 需要：`Dq = d_head + d_rope`，`Dv = d_head`），
且支持 `Tq ≠ Tkv`（增量解码，配合 `causal_mask_rect`）。

### 辅助函数
- `rope_tables(dim, max_len, base=10000.0) -> (cos, sin)`：形状 `(max_len, dim)`，前后半已拼接
- `causal_mask(T)`：方阵因果掩码
- `causal_mask_rect(Tq, Tk)`：矩形因果掩码，**右对齐** `j ≤ i + (Tk − Tq)`
- `silu(x)` / `gelu(x)`

---

## L4 Gated DeltaNet

```python
from tianji import GatedDeltaNet, CausalConv1D, L2Norm, l2norm
```

### `class GatedDeltaNet`
`forward(ctx, q, k, v, beta, alpha) -> o`

| 参数 | 形状 | 说明 |
|---|---|---|
| `q, k, v` | `(B, H, T, D)` | `k` 内部会做 L2 归一化 |
| `beta` | `(B, H, T)` | 写入强度，值域 (0,1) |
| `alpha` | `(B, H, T)` | 遗忘门，值域 (0,1) |

递推式：`S_t = α_t S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`，`o_t = S_t q_t`。

反向**沿时间递推**（`G_t = g_tᵒ q_tᵀ + α_{t+1}[G_{t+1} − β_{t+1}(G_{t+1}k_{t+1})k_{t+1}ᵀ]`），
不做逐节点展开。梯度检验容差 1e-6 内通过。

### `class CausalConv1D(kernel_size=4)`
`forward(ctx, x, weight)`：`x:(B,T,C)`，`weight:(C,K)`。深度可分离因果卷积
（Mamba-2 / fla 的标配局部混合预处理）。

### `L2Norm` / `l2norm(x, eps=1e-8)`

---

## L5 MoE

```python
from tianji import MoELayer, LoadBalancer, sigmoid, sequence_balance_loss
```

### `class MoELayer(n_experts, top_k=2, shared=True)`
`forward(ctx, x, router, Wg, Wu, Wd, bias, sWg=None, sWu=None, sWd=None)`

- `x:(N,d)`（已展平），`router:(E,d)`，`Wg/Wu:(E,dff,d)`，`Wd:(E,d,dff)`
- `bias:(E,)` 由 `LoadBalancer` 维护，**不可微**
- 共享专家参数 `sWg/sWu/sWd` 为 2D，`shared=False` 时传 `None`

门控语义（DeepSeek-V3）：`s = σ(uᵀe)` 参与门控值；`TopK(s + b)` 中偏置只影响**谁被选中**。
每 token 的门控权重之和恒为 1；无 token 丢弃。

### `class LoadBalancer(n_experts, gamma=1e-3, bias_cap=1.0)`

| 方法 | 说明 |
|---|---|
| `update(counts)` | 过载 `b -= γ`，欠载 `b += γ`（不经过梯度） |
| `maxcv(window=32)` | 近期负载变异系数；历史不足 2 条返回 `inf` |
| `reset()` | 清零 |

### `sequence_balance_loss(affinity, assign, alpha=1e-4) -> (loss, grad)`
极小序列级均衡损失（V3 保留项）。`f` 按 detach 处理。

---

## L6 优化器

```python
from tianji import AdamW, Muon, Muon2, HybridOptimizer, orthogonalization_quality
```

| 类 | 关键参数 | 用途 |
|---|---|---|
| `AdamW` | `lr=1e-3, betas=(0.9,0.95), eps=1e-8, weight_decay=0.01` | 1D 参数、embedding、LM head |
| `Muon` | `lr=0.02, momentum=0.95, nesterov=True, ns_steps=5` | 2D 隐藏权重 |
| `Muon2` | `lr=0.02, betas=(0.9,0.95), ns_steps=3` | **默认**：NS 前插 Adam 二阶矩预条件 |
| `HybridOptimizer` | `muon2=None, adamw=None, max_grad_norm=1.0` | 按角色自动路由 |

三者统一接口：`step(params: dict, grads: dict)`（原地更新）。

`HybridOptimizer.is_muon(name, p)` 决定路由：`ndim ≥ 2` 且不在
`ADAM_ONLY_EXACT = ("tok_emb", "lm_head")` 与 `ADAM_ONLY_SUFFIX = ("norm1_w", "norm2_w", "final_norm_w", "conv_w")` 中。
> embedding / LM head 虽是 2D，但梯度是稀疏查表式的，正交化会破坏语义结构 —— 必须显式排除。

3D 参数（MoE 专家权重）按 `(首维, -1)` 展平后正交化再还原。

`orthogonalization_quality(G, O) -> float`：与真极分解 `UVᵀ` 的余弦相似度。

---

## L7 训练

```python
from tianji import Trainer, DataLoader, cosine_schedule, wsd_schedule, evaluate
```

### `class DataLoader(tokens, seq_len, batch_size, seed=0)`
`batch() -> (x, y)`，随机起点 + 连续切片。

### `class Trainer(model, opt, loader, *, total_steps, warmup=20, lr_max=0.02, lr_min=0.0, accum=1, schedule="cosine", val_loader=None, eval_every=0, use_balancer=True)`

| 成员 | 说明 |
|---|---|
| `step(step) -> float` | 单步（含梯度累积），返回 loss |
| `run(steps, verbose=True, print_every=20) -> TrainLog` | 训练循环 |
| `lr_at(step) -> float` | 当前学习率 |

`TrainLog` 字段：`step, losses, val, grad_norms, seconds, lr_trace`。

### `cosine_schedule(step, total, warmup, lr_max, lr_min=0.0) -> float`
### `wsd_schedule(step, total, warmup, lr_max, lr_min=0.0, decay_frac=0.1) -> float`
WSD 把稳定期与衰减期解耦，可随时从 checkpoint 启动衰减。

### `evaluate(model, loader, n_batch=8) -> float`

---

## L8 推理

```python
from tianji import (quantize_per_channel, quantize_groupwise, dequantize,
                    sample_top_p, generate, speculative_decode, SpecStats)
```

### 量化
- `quantize_per_channel(W, bits=8) -> (Wq_int32, scale)`：逐输出通道对称量化
- `quantize_groupwise(W, bits=4, group=32) -> (Wq, scale, group)`：分组量化（`group` 自动不超过输入维）
- `dequantize(Wq, scale, group=None) -> W`

### `sample_top_p(logits, top_p=0.9, temperature=1.0, seed=None) -> int`
Top-p（nucleus）采样。

### `generate(model, prompt, max_new=32, *, temperature=1.0, top_p=0.9, seed=0, use_cache=True) -> list[int]`
自回归生成。全注意力层启用 KV cache（GDN 层仍整序列重算，见"已知边界"）。

### `speculative_decode(model, prompt, max_new=32, *, gamma=1, temperature=1.0, top_p=0.9, seed=0) -> (ids, SpecStats)`
用 MTP 头做 draft、主模型验证。接受概率 `min(1, p_target/p_draft)`，
拒绝时从残差分布 `∝ max(0, p_target − p_draft)` 重采样 —— **数学上严格无损**。

`SpecStats`：`drafted, accepted, steps, accept_rate, produced, speedup`（speedup = 产出 token 数 / 主干 forward 次数）。两条路径：`draft_model=None` 走 MTP 搭车（γ=1，结构性 ≈1，G14 机检无损）；`draft_model=TinyDraftGDN` 走独立轻量 draft + γ>1 一次 batch 并行验证（实测 tokens/forward 可达 3+，G15 机检位置指纹/代数条件/统计无损）。

---

## L9 对齐

```python
from tianji import (Aligner, freeze_copy, seq_logprob, mean_logprob,
                    grad_logprob, dpo_terms, ipo_terms, simpo_terms, kto_terms)
```

### `class Aligner(policy, ref_params=None, method="dpo", beta=0.1)`
`loss_and_grads(x_c, y_c, x_r, y_r) -> (loss, grads)`

`method`：`"dpo" | "ipo" | "simpo" | "kto"`。四者都是 logits 的标量可微函数，
统一走 `model.backward(g_logits=...)` 复用同一反传骨架。

### `freeze_copy(params) -> dict`
参考模型深拷贝。**必须深拷贝** —— 与 policy 共享数组会让 Δ 的基准漂移，
这是 DPO 实现里最隐蔽的错误（不报错，只是对齐无效）。

### 损失函数项
| 函数 | 返回 | 说明 |
|---|---|---|
| `dpo_terms(lp_p_c, lp_p_r, lp_ref_c, lp_ref_r, beta)` | `(loss, coef)` | `−log σ(βΔ)` |
| `ipo_terms(..., beta, tau=0.1)` | `(loss, coef)` | `(Δ − τ/2β)²`，修掉 DPO 的无限拉大病态 |
| `simpo_terms(mp_c, mp_r, beta=2.0, gamma=0.5)` | `(loss, coef)` | 长度归一化 + 固定边界，无参考模型 |
| `kto_terms(lp_p, lp_ref, desirable, beta)` | `(loss, coef)` | 二元好/坏标注 |

### 辅助
- `seq_logprob(logits, targets) -> float`
- `mean_logprob(logits, targets) -> float`（SimPO 用）
- `grad_logprob(logits, targets) -> np.ndarray`（`onehot − softmax`）

---

## L10 微指令

```python
from tianji import MICROOPS, FAMILIES, FAMILY_KIND, run_microops
```

- `MICROOPS: dict[str, MicroOp]` —— 1024 条，键为 `"family.name"`
- `FAMILIES: dict[str, list[str]]` —— 32 个家族
- `FAMILY_KIND: dict[str, str]` —— `"core"`（成员语义独立，15 个）或 `"sweep"`（参数扫描，17 个）
- `run_microops(seed=0, shape=(4,8)) -> dict` —— 全部执行冒烟，返回 `{total, ok, failed, families}`

`MicroOp` 字段：`family, name, arity, fn, note`。

```python
from tianji import MICROOPS
op = MICROOPS["uact.silu"]
y = op.fn(x)                      # arity=1；arity=2 时为 op.fn(x, y)
```

---

## 模型

```python
from tianji import TianjiConfig, TianjiModel, init_params, build_layer_layout
```

### `class TianjiConfig`（dataclass）
主要字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `vocab_size` | 512 | 词表 |
| `d_model` | 96 | 隐藏维 |
| `n_layers` | 8 | 层数 |
| `n_heads` / `n_kv_heads` | 4 / 2 | 查询头 / KV 头（GQA，仅 `mla=False` 时用） |
| `hybrid_ratio` | 3 | 每 `(ratio+1)` 层中 `ratio` 层为 GDN |
| `layout` | `"interleave"` | `"interleave"`（Qwen）或 `"edge"`（Mamba-3） |
| `mla` | True | 全注意力层用 MLA（`False` 则 GQA+RoPE） |
| `d_latent_q` / `d_latent_kv` / `d_rope` | 48 / 32 / 8 | MLA 压缩维与解耦 RoPE 维 |
| `n_experts` / `moe_top_k` / `shared_expert` / `moe_every` | 4 / 2 / True / 2 | MoE 配置 |
| `d_ff` / `d_ff_expert` | 192 / 96 | FFN 隐藏维 |
| `conv_kernel` | 4 | 短卷积核 |
| `mtp_depth` / `mtp_lambda` | 1 / 0.1 | 多令牌预测 |
| `tie_embeddings` | True | 绑定 embedding 与 LM head |
| `max_len` / `rope_base` | 512 / 10000.0 | 位置编码 |

属性：`d_head`、`dim_inner`。

### `class TianjiModel(cfg, params=None, seed=0)`

| 成员 | 说明 |
|---|---|
| `forward(tokens, targets=None, cache=None, pos=0) -> dict` | 返回 `{"logits","xf","loss","aux","mtp_logits"}` |
| `backward(g_logits=None) -> dict[str, np.ndarray]` | 参数梯度；传 `g_logits` 则对齐路径 |
| `param_count() -> int` | 参数量 |
| `layers` / `is_moe_layer(i)` / `head_w()` | 结构查询 |
| `balancers` | MoE 层的 `LoadBalancer` 字典 |

`cache` 用于增量解码；`pos` 是**绝对位置**（RoPE 必须用绝对位置，否则长文本质量崩掉但不报错）。

### `init_params(cfg, seed=0) -> dict`
残差分支的 out_proj 按 `1/√(2·n_layers)` 缩放（GPT-2 的 residual scaling）。

### `build_layer_layout(n_layers, ratio=3, layout="interleave") -> list[str]`
返回 `"gdn"` / `"attn"` 序列。

---

## 持久化

```python
from tianji import save_checkpoint, load_checkpoint, checkpoint_info, CheckpointError
```

### `save_checkpoint(path, cfg, params, *, meta=None, tokenizer_path=None) -> sha256`
落盘 `<path>.npz`（npz 压缩）与 `<path>.npz.meta.json`（配置 + 参数表 + 指纹 + 校验和）。
路径不带 `.npz` 后缀时自动补全（`save`/`load`/`checkpoint_info` 三路同一规范化，
传什么名字进去就原样传出来）。**原子落盘**：先写临时文件再 `os.replace`，
保存中途崩溃不留半成品，旧 checkpoint 始终配对完好（对齐 safetensors 模式）。

### `load_checkpoint(path, *, verify=True) -> (params, info)`
`verify=True` 时强制校验 sha256 与参数指纹，**不一致直接抛 `CheckpointError`**。
损坏的 meta.json（截断/非法 JSON）、损坏的权重、参数集合/指纹失配、版本不兼容、
文件缺失 —— 全部转成 `CheckpointError`，调用方只需捕获这一种。

### `checkpoint_info(path) -> dict`
只读元数据，不加载权重。损坏同样抛 `CheckpointError`。

---

## 数据

```python
from tianji import TINY_TALE, default_corpus, markov_tokens, markov_entropy_rate
```

- `TINY_TALE` —— 极小的中英混合文本
- `default_corpus()` —— 默认训练语料（约 93.5KB，71280 字符）
- `markov_tokens(n, vocab, seed=0, order=2)` —— 确定性合成马尔可夫序列（基准语料源；与 `bench/run_bench.py` 同源，已去重）
- `markov_entropy_rate(vocab=64, seed=0, order=2, n_sample=200_000, tol=5e-3)` —— 计算该马尔可夫源的**熵率下界 H**，返回 dict（`H / H_analytic / H_oracle / abs_diff / consistent` 等）。**两套独立算法互相印证**（解析平稳分布法 + oracle 交叉熵法），是「降到多少才算学会」的可机检判据，由门禁 G24 守着

全部离线，无任何下载。

---

## 门禁

```python
from tianji import run_all_gates, GateReport, GateResult
rep = run_all_gates(seed=0, steps=40)
print(rep.table())        # G1..G24 的 PASS/FAIL 与实测值
rep.all_green             # 是否 24/24
```

---

## 命令行

```bash
python -m tianji.cli gates      # 24 道交付门禁
python -m tianji.cli test       # 模块梯度检验冒烟
python -m tianji.cli bench      # 完整基准
python -m tianji.cli microops   # 微指令统计（--show 列出全部家族）
python -m tianji.cli train      # 端到端训练 + 生成

python -m tianji.cli train --steps 300 --d-model 96 --layers 8 \
    --layout edge --optim muon2 --prompt "the quick brown"
```

各子命令均支持 `--steps` / `--seed`（必须在子命令之后传递）。

---

## 完整示例

```python
import numpy as np
from tianji import (BPE, TianjiConfig, TianjiModel, init_params,
                    DataLoader, Trainer, HybridOptimizer, generate,
                    save_checkpoint, load_checkpoint, run_all_gates)

corpus = "the quick brown fox jumps over the lazy dog. " * 100
bpe = BPE(vocab_size=512).train(corpus)
ids = np.asarray(bpe.encode(corpus), dtype=np.int64)

cfg = TianjiConfig(vocab_size=max(bpe.size, 512), d_model=96, n_layers=8)
model = TianjiModel(cfg, init_params(cfg, seed=0))

ld = DataLoader(ids, seq_len=64, batch_size=8, seed=0)
Trainer(model, HybridOptimizer(), ld, total_steps=200,
        warmup=20, lr_max=0.03).run(200)

out_ids = generate(model, np.asarray(bpe.encode("the quick")), max_new=40)
print(bpe.decode(np.asarray(out_ids)))

save_checkpoint("runs/model.npz", cfg, model.P)
P2, info = load_checkpoint("runs/model.npz")     # 自动校验 sha256

print(run_all_gates().table())
```
