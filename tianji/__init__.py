"""天玑 TIANJI —— 混合架构语言模型全栈。

十层架构（每层只做一件事，层与层之间只经明确契约通信）：

    L0  数值内核      稳定 logsumexp / softmax / Newton-Schulz / 谱范数 / 确定性种子
    L1  反向契约      显式 backward 契约 + 中心差分梯度检验（不用 tape AD，理由见模块）
    L2  分词          字节级 BPE（GPT-2 谱系，含 CJK 单字切分改进）
    L3  架构原语      RMSNorm / RoPE / SwiGLU / GQA / MLA / 交叉熵
    L4  序列混合器    Gated DeltaNet（delta rule + 遗忘门）
    L5  稀疏化        DeepSeek-V3 风格 MoE（无辅助损失负载均衡）
    L6  优化器        AdamW / Muon / Muon2 与混合路由
    L7  训练          调度 / 数据管道 / 梯度累积 / 度量
    L8  推理          KV cache / 量化 / 投机解码
    L9  对齐          DPO / IPO / SimPO / KTO
    L10 微指令+门禁   1024 条可执行微指令 + 26 道交付门禁

快速开始：
    >>> from tianji import TianjiConfig, TianjiModel, init_params
    >>> cfg = TianjiConfig(vocab_size=512, d_model=96, n_layers=8)
    >>> m = TianjiModel(cfg, init_params(cfg, seed=0))
    >>> out = m.forward(tokens, targets)
    >>> grads = m.backward()
"""

from .l0_numerics import (
    stable_hash, spawn_seed, rng, logsumexp, softmax, log_softmax,
    spectral_norm, newton_schulz, rms, topk, clip_grad_norm,
)
from .l1_contract import Ctx, Module, gradcheck, GradReport
from .l2_tokenizer import BPE, pretokenize
from .l3_primitives import (
    Linear, RMSNorm, RoPE, SwiGLU, CausalAttention, CrossEntropy,
    rope_tables, causal_mask, causal_mask_rect, silu, gelu,
)
from .l4_deltanet import GatedDeltaNet, CausalConv1D, l2norm, L2Norm
from .l5_moe import MoELayer, LoadBalancer, sigmoid, sequence_balance_loss
from .l6_optimizer import AdamW, Muon, Muon2, HybridOptimizer, orthogonalization_quality
from .l7_train import cosine_schedule, wsd_schedule, DataLoader, Trainer, evaluate
from .l8_infer import (
    quantize_per_channel, quantize_groupwise, dequantize,
    sample_top_p, generate, speculative_decode, SpecStats, TinyDraftGDN,
)
from .l9_align import (
    seq_logprob, mean_logprob, grad_logprob, Aligner, freeze_copy,
    dpo_terms, ipo_terms, simpo_terms, kto_terms,
)
from .data import (TINY_TALE, default_corpus, markov_tokens,           # noqa: F401
                   markov_entropy_rate, markov_transitions, markov_train_val)
from .l10_microops import MICROOPS, FAMILIES, FAMILY_KIND, run_all as run_microops
from .model import TianjiConfig, TianjiModel, init_params, build_layer_layout
from .persist import save_checkpoint, load_checkpoint, checkpoint_info, CheckpointError
from .verify import run_all_gates, GateReport, GateResult

__version__ = "1.0.0"
__author__ = "晨星 (Chenxing)"
__license__ = "MIT"

__all__ = [
    # L0
    "stable_hash", "spawn_seed", "rng", "logsumexp", "softmax", "log_softmax",
    "spectral_norm", "newton_schulz", "rms", "topk", "clip_grad_norm",
    # L1
    "Ctx", "Module", "gradcheck", "GradReport",
    # L2
    "BPE", "pretokenize",
    # L3
    "Linear", "RMSNorm", "RoPE", "SwiGLU", "CausalAttention", "CrossEntropy",
    "rope_tables", "causal_mask", "causal_mask_rect", "silu", "gelu",
    # L4
    "GatedDeltaNet", "CausalConv1D", "l2norm", "L2Norm",
    # L5
    "MoELayer", "LoadBalancer", "sigmoid", "sequence_balance_loss",
    # L6
    "AdamW", "Muon", "Muon2", "HybridOptimizer", "orthogonalization_quality",
    # L7
    "cosine_schedule", "wsd_schedule", "DataLoader", "Trainer", "evaluate",
    # L8
    "quantize_per_channel", "quantize_groupwise", "dequantize",
    "sample_top_p", "generate", "speculative_decode", "SpecStats",
    # L9
    "seq_logprob", "mean_logprob", "grad_logprob", "Aligner", "freeze_copy",
    "dpo_terms", "ipo_terms", "simpo_terms", "kto_terms",
    # L10
    "MICROOPS", "FAMILIES", "FAMILY_KIND", "run_microops",
    # 数据与持久化
    "TINY_TALE", "default_corpus", "markov_tokens", "markov_entropy_rate",
    "markov_transitions", "markov_train_val",
    "save_checkpoint", "load_checkpoint", "checkpoint_info", "CheckpointError",
    # 模型与门禁
    "TianjiConfig", "TianjiModel", "init_params", "build_layer_layout",
    "run_all_gates", "GateReport", "GateResult",
    "__version__", "__author__", "__license__",
]
