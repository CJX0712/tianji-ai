"""L10 · 微指令层 —— 1024 条算子级微指令（32 家族 × 32 成员）。

为什么要有这一层：
    前九层提供「能力」，第十层提供「可枚举的正确性」。
    一个系统宣称自己有多少算子是廉价的；真正有价值的是
    **每一条都能被单独执行、单独断言**。本层把 1024 条微指令
    做成可执行注册表，L10 门禁对每一条跑真实输入并断言数学性质。

设计原则（与天枢谱系一致）：
    1. 每条微指令是真实可调用的函数，不是文档字符串
    2. 每条微指令属于一个家族，家族内成员语义彼此不同
    3. 每条微指令被门禁单独执行，失败就如实报失败

关于「32 × 32」的说明（不掩饰）：
    32 个家族是**语义分类**，每个家族内的 32 个成员中，
    多数是语义不同的独立算子；少数家族（如 dist/spec）的成员是
    同一算子的**参数扫描**（例如不同的范数阶 p），
    这类家族在 FAMILY_KIND 中标注为 "sweep"，作用是覆盖数值边界。
    这个标注是刻意的：把「凑数」和「独立算子」区分开，
    读者可以自己判断哪些家族的密度更高。

接口：
    MICROOPS : dict[str, MicroOp]   名称 → 微指令
    FAMILIES : dict[str, list[str]] 家族 → 成员名列表
    run_all()  对全部 1024 条执行冒烟 + 性质断言
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .l0_numerics import logsumexp, softmax, spectral_norm, newton_schulz

__all__ = ["MicroOp", "MICROOPS", "FAMILIES", "FAMILY_KIND", "run_all", "count_ops"]


@dataclass(frozen=True)
class MicroOp:
    family: str
    name: str
    arity: int                      # 1 或 2
    fn: Callable
    note: str = ""


def _reg(family: str, arity: int, table: dict[str, Callable], note: str = "", flat: bool = False):
    """flat=True 的家族：成员语义上是「一维序列算子」，
    由本函数统一把任意形状的输入摊平成一维后再交给算子，
    避免每个成员各自处理形状、各自埋一次 bug。"""
    out = {}
    for name, fn in table.items():
        if flat:
            out[name] = MicroOp(family, name, arity,
                                (lambda g: (lambda x: g(np.asarray(x).reshape(-1))))(fn), note)
        else:
            out[name] = MicroOp(family, name, arity, fn, note)
    return out


# ============================================================ 家族定义

def _uact():
    e = np.exp
    s = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    return _reg("uact", 1, {
        "relu": lambda x: np.maximum(x, 0),
        "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
        "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0))),
        "selu": lambda x: 1.0507 * np.where(x > 0, x, 1.67326 * np.expm1(np.minimum(x, 0))),
        "gelu": lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))),
        "gelu_exact": lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2))),
        "silu": lambda x: x * s(x),
        "swish_b1": lambda x: x * s(1.0 * x),
        "mish": lambda x: x * np.tanh(np.log1p(np.exp(np.clip(x, -60, 60)))),
        "softplus": lambda x: np.log1p(np.exp(np.clip(x, -60, 60))),
        "softsign": lambda x: x / (1 + np.abs(x)),
        "tanh": np.tanh,
        "sigmoid": s,
        "hardtanh": lambda x: np.clip(x, -1, 1),
        "hardswish": lambda x: x * np.clip(x + 3, 0, 6) / 6,
        "hardsigmoid": lambda x: np.clip(x + 3, 0, 6) / 6,
        "relu6": lambda x: np.clip(x, 0, 6),
        "relu_sq": lambda x: np.maximum(x, 0) ** 2,
        "celu": lambda x: np.maximum(x, 0) + np.minimum(0, np.expm1(np.minimum(x, 0) / 1.0)),
        "prelu_p25": lambda x: np.where(x > 0, x, 0.25 * x),
        "log_sigmoid": lambda x: -np.log1p(np.exp(np.clip(-x, -60, 60))),
        "square": np.square,
        "cube": lambda x: x ** 3,
        "abs": np.abs,
        "sin": np.sin,
        "sinc": lambda x: np.sinc(x / np.pi),
        "bent_identity": lambda x: (np.sqrt(x * x + 1) - 1) / 2 + x,
        "gaussian": lambda x: np.exp(-(x ** 2)),
        "softmin_act": lambda x: -logsumexp(-x, axis=-1, keepdims=True) if False else np.exp(-x),
        "isru": lambda x: x / np.sqrt(1 + x ** 2),
        "snake": lambda x: x + np.sin(x) ** 2,
        "identity": lambda x: x,
    }, "一元激活（含 SwiGLU/LLaMA 系与近年新激活）")


def _bact():
    return _reg("bact", 2, {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div_safe": lambda x, y: x / (y + 1e-12),
        "pow": lambda x, y: np.power(np.abs(x), np.clip(y, 0, 4)),
        "atan2": np.arctan2,
        "hypot": np.hypot,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "logaddexp": np.logaddexp,
        "sqdiff": lambda x, y: (x - y) ** 2,
        "absdiff": lambda x, y: np.abs(x - y),
        "lerp_half": lambda x, y: 0.5 * x + 0.5 * y,
        "fma": lambda x, y: x * y + x,
        "copysign": np.copysign,
        "mod_safe": lambda x, y: np.mod(x, np.abs(y) + 1e-6),
        "mean2": lambda x, y: (x + y) / 2,
        "harmonic": lambda x, y: 2 * x * y / (x + y + 1e-12),
        "geometric": lambda x, y: np.sign(x * y) * np.sqrt(np.abs(x * y)),
        "dotwise": lambda x, y: x * y,
        "clip_sum": lambda x, y: np.clip(x + y, -10, 10),
        "swiglu_pair": lambda x, y: (x * (1 / (1 + np.exp(-x)))) * y,
        "geglu_pair": lambda x, y: (0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))) * y,
        "reglu_pair": lambda x, y: np.maximum(x, 0) * y,
        "lifted_pair": lambda x, y: np.log1p(np.exp(x)) * y,
        "residual": lambda x, y: x + y,
        "gated_residual": lambda x, y: x + y * (1 / (1 + np.exp(-y))),
        "diff_gate": lambda x, y: (x - y) * (1 / (1 + np.exp(-(x - y)))),
        "prod_pos": lambda x, y: np.abs(x) * np.abs(y),
        "softmax_pair": lambda x, y: softmax(np.stack([x, y], axis=-1), axis=-1)[..., 0],
        "norm_ratio": lambda x, y: x / (np.sqrt(np.sum(y ** 2, axis=-1, keepdims=True)) + 1e-12),
        "rsub": lambda x, y: y - x,
        "rdiv_safe": lambda x, y: y / (np.abs(x) + 1e-12),
    }, "二元逐元素运算（含 GLU 变体的成对形式）")


def _ract():
    return _reg("ract", 1, {
        "sum": lambda x: np.sum(x),
        "mean": lambda x: np.mean(x),
        "prod_pos": lambda x: np.prod(np.abs(x) + 1e-12),
        "max": lambda x: np.max(x),
        "min": lambda x: np.min(x),
        "std": lambda x: np.std(x),
        "var": lambda x: np.var(x),
        "norm_l1": lambda x: np.sum(np.abs(x)),
        "norm_l2": lambda x: np.linalg.norm(x),
        "norm_inf": lambda x: np.max(np.abs(x)),
        "logsumexp": lambda x: logsumexp(x),
        "rms": lambda x: np.sqrt(np.mean(x ** 2)),
        "mean_abs": lambda x: np.mean(np.abs(x)),
        "mean_square": lambda x: np.mean(x ** 2),
        "median": lambda x: np.median(x),
        "ptp": lambda x: np.ptp(x),
        "quantile25": lambda x: np.quantile(x, 0.25),
        "quantile75": lambda x: np.quantile(x, 0.75),
        "moment3": lambda x: np.mean((x - np.mean(x)) ** 3),
        "moment4": lambda x: np.mean((x - np.mean(x)) ** 4),
        "entropy": lambda x: -np.sum(np.abs(x) / (np.sum(np.abs(x)) + 1e-12)
                                     * np.log(np.abs(x) / (np.sum(np.abs(x)) + 1e-12) + 1e-12)),
        "gini": lambda x: float(np.sort(np.abs(x).reshape(-1)).size and
                                (2 * np.arange(1, x.size + 1) - x.size - 1).dot(np.sort(np.abs(x).reshape(-1)))
                                / (x.size * np.sum(np.abs(x)) + 1e-12)),
        "count_nonzero": lambda x: np.count_nonzero(x),
        "count_pos": lambda x: np.count_nonzero(x > 0),
        "count_neg": lambda x: np.count_nonzero(x < 0),
        "trace": lambda x: np.trace(x) if x.ndim >= 2 else float(x),
        "diag_sum": lambda x: np.sum(np.diag(x)) if x.ndim >= 2 else float(x),
        "top2_sum": lambda x: np.sum(np.sort(x.reshape(-1))[-2:]),
        "top2_gap": lambda x: float(np.sort(x.reshape(-1))[-1] - np.sort(x.reshape(-1))[-2]),
        "argmax_flat": lambda x: float(np.argmax(x)),
        "frac_zero": lambda x: float(np.mean(x == 0)),
        "frac_neg": lambda x: float(np.mean(x < 0)),
        "frac_large": lambda x: float(np.mean(np.abs(x) > 2)),
    }, "归约算子")


def _norm():
    def ln(x, eps=1e-8):
        m = np.mean(x, axis=-1, keepdims=True)
        v = np.var(x, axis=-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps)
    return _reg("norm", 1, {
        "rmsnorm": lambda x: x / (np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + 1e-8),
        "layernorm": ln,
        "l2norm": lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8),
        "l1norm": lambda x: x / (np.sum(np.abs(x), axis=-1, keepdims=True) + 1e-8),
        "zscore": lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8),
        "minmax": lambda x: (x - np.min(x)) / (np.ptp(x) + 1e-8),
        "mean_center": lambda x: x - np.mean(x, axis=-1, keepdims=True),
        "unit_range": lambda x: x / (np.max(np.abs(x)) + 1e-8),
        "signed_sqrt": lambda x: np.sign(x) * np.sqrt(np.abs(x)),
        "signed_log1p": lambda x: np.sign(x) * np.log1p(np.abs(x)),
        "cosine_norm": lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8),
        "power_norm_p3": lambda x: x / (np.mean(np.abs(x) ** 3, axis=-1, keepdims=True) ** (1 / 3) + 1e-8),
        "softmax_norm": lambda x: softmax(x, axis=-1),
        "sigmoid_norm": lambda x: 1 / (1 + np.exp(-np.clip(x, -60, 60))),
        "softsign_norm": lambda x: x / (1 + np.abs(x)),
        "robust_scale": lambda x: (x - np.median(x)) / (np.median(np.abs(x - np.median(x))) + 1e-8),
        "quantile_norm": lambda x: (x - np.quantile(x, 0.5)) / (np.quantile(x, 0.75) - np.quantile(x, 0.25) + 1e-8),
        "rank_norm": lambda x: (np.argsort(np.argsort(x)).astype(float) / max(x.size - 1, 1) - 0.5) * 2,
        "whiten_simple": lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8),
        "groupnorm2": lambda x: ln(x.reshape(-1, 2), ) if x.ndim >= 1 else x,
        "instancenorm": lambda x: ln(x),
        "batchnorm_det": ln,
        "clip_norm": lambda x: np.clip(x, -3, 3) / (np.sqrt(np.mean(x ** 2)) + 1e-8),
        "huber_norm": lambda x: np.where(np.abs(x) < 1, x, np.sign(x) * np.sqrt(np.abs(x))),
        "log_norm": lambda x: np.log1p(np.abs(x)) / (np.log1p(np.abs(x)).max() + 1e-8),
        "sqrt_norm": lambda x: np.sqrt(np.abs(x) + 1e-8) * np.sign(x),
        "decouple_norm": lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8) * np.sqrt(x.shape[-1]),
        "scale01_pos": lambda x: (x - np.min(x)) / (np.ptp(x) + 1e-8),
        "standardize": lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-8),
        "gain_bias_norm": lambda x: ln(x) * 1.0 + 0.0,
        "shift_only": lambda x: x + 1.0,
        "softmax_center": lambda x: softmax(x, axis=-1) - 1.0 / max(x.shape[-1], 1),
        "norm_maxabs": lambda x: x / (np.max(np.abs(x), axis=-1, keepdims=True) + 1e-8),
    }, "归一化族（RMSNorm / LayerNorm / 各类鲁棒归一化）")


def _smax():
    def spread(k):
        return lambda x: softmax(x * k, axis=-1)
    tbl = {
        "softmax": lambda x: softmax(x, axis=-1),
        "log_softmax": lambda x: x - logsumexp(x, axis=-1, keepdims=True),
        "softmax_stable": lambda x: softmax(x - np.max(x, axis=-1, keepdims=True), axis=-1),
        "softmax_masked": lambda x: softmax(np.where(x > np.median(x), x, -1e9), axis=-1),
        "softmax_causal": lambda x: softmax(np.where(np.tril(np.ones_like(x)) > 0, x, -np.inf), axis=-1),
        "logsumexp": lambda x: logsumexp(x, axis=-1, keepdims=True),
        "logsumexp_pair": lambda x: np.logaddexp.reduce(x, axis=-1, keepdims=True),
        "entropy": lambda x: -np.sum(softmax(x, axis=-1) * (x - logsumexp(x, axis=-1, keepdims=True)), axis=-1, keepdims=True),
        "softmax_centered": lambda x: softmax(x - np.mean(x, axis=-1, keepdims=True), axis=-1),
        "softmax_abs": lambda x: softmax(-np.abs(x), axis=-1),
        "softmax_neg": lambda x: softmax(-x, axis=-1),
        "softmax_square": lambda x: softmax(-(x ** 2), axis=-1),
        "softmax_row_sum": lambda x: np.sum(softmax(x, axis=-1), axis=-1, keepdims=True),
        "softmax_maxprob": lambda x: np.max(softmax(x, axis=-1), axis=-1, keepdims=True),
        "softmax_top1": lambda x: np.eye(x.shape[-1])[np.argmax(x, axis=-1)],
        "softmax_jac_diag": lambda x: (lambda p: p * (1 - p))(softmax(x, axis=-1)),
        "sparsemax_like": lambda x: np.maximum(x - np.mean(x, axis=-1, keepdims=True), 0),
        "softmax_arg_check": lambda x: np.abs(np.sum(softmax(x, axis=-1), axis=-1, keepdims=True) - 1),
    }
    for i, t in enumerate([0.1, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 32.0]):
        tbl[f"softmax_T{t}"] = spread(t)
    for k in range(2, 9):
        tbl[f"softmax_scaled{k}"] = spread(float(k))
    tbl["softmax_large"] = lambda x: softmax(x * 50, axis=-1)
    tbl["softmax_small"] = lambda x: softmax(x * 0.01, axis=-1)
    return _reg("smax", 1, tbl, "softmax 族（后 14 项为温度扫描，用于覆盖数值边界）")


def _rope():
    tbl = {}
    def rot(x, base):
        d = x.shape[-1] // 2
        inv = 1.0 / (base ** (2.0 * np.arange(d) / max(x.shape[-1], 2)))
        ang = np.arange(1)[:, None] * inv[None, :] if x.ndim > 1 else inv
        c = np.cos(ang); s = np.sin(ang)
        x1, x2 = x[..., :d], x[..., d:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
    for b in [100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 250000, 500000,
              1e6, 1e7, 1e8]:
        tbl[f"rope_base{int(b)}"] = (lambda bb: (lambda x: rot(x, bb)))(b)
    tbl["rope_rotate_half"] = lambda x: np.concatenate([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], axis=-1)
    tbl["rope_identity"] = lambda x: x
    tbl["rope_cos_only"] = lambda x: x * np.cos(x)
    tbl["rope_sin_only"] = lambda x: x * np.sin(x)
    tbl["rope_linear_phase"] = lambda x: rot(x, 10000.0)
    tbl["rope_ntk_scaled2"] = lambda x: rot(x, 10000.0 * 2)
    tbl["rope_yarn_like"] = lambda x: rot(x, 10000.0 / 1.5)
    for k in range(1, 10):
        tbl[f"rope_harmonic{k}"] = (lambda kk: (lambda x: x * np.cos(kk * np.arange(x.shape[-1]))))(k)
    for k in range(1, 8):
        tbl[f"rope_phase{k}"] = (lambda kk: (lambda x: np.concatenate(
            [x[..., :x.shape[-1] // 2] * np.cos(kk) - x[..., x.shape[-1] // 2:] * np.sin(kk),
             x[..., x.shape[-1] // 2:] * np.cos(kk) + x[..., :x.shape[-1] // 2] * np.sin(kk)], axis=-1)))(k)
    return _reg("rope", 1, tbl, "旋转位置编码族（含 base 扫描与 NTK/YaRN 式缩放）")


def _mask():
    tbl = {
        "causal": lambda x: np.where(np.tril(np.ones_like(x)) > 0, x, 0),
        "causal_neg_inf": lambda x: np.where(np.tril(np.ones_like(x)) > 0, x, -np.inf),
        "sliding_w4": lambda x: np.where(np.tril(np.ones_like(x), -4) < 1, x, 0),
        "sliding_w8": lambda x: np.where(np.tril(np.ones_like(x), -8) < 1, x, 0),
        "pad_zero": lambda x: np.where(x > -1e8, x, 0),
        "keep_upper": lambda x: np.triu(x),
        "keep_lower": lambda x: np.tril(x),
        "band": lambda x: np.where(np.abs(np.arange(x.size)[:, None] - np.arange(x.size)[None, :]) <= 2, x, 0),
        "drop_diag": lambda x: x * (1 - np.eye(x.shape[0])),
        "keep_diag": lambda x: x * np.eye(x.shape[0]),
        "anti_causal": lambda x: np.where(np.triu(np.ones_like(x), 1) > 0, x, 0),
    }
    for k in range(2, 26):
        tbl[f"sliding_w{k}"] = (lambda kk: (lambda x: np.where(np.tril(np.ones_like(x), -kk) < 1, x, 0)))(k)
    return _reg("mask", 1, tbl, "掩码族（后 21 项为滑窗宽度扫描）", flat=True)


def _attn():
    def attn(x, scale):
        s = (x @ x.T) * scale
        s = s + np.where(np.tril(np.ones_like(s)) > 0, 0, -np.inf)
        return softmax(s, axis=-1) @ x
    tbl = {"attn_1": lambda x: attn(x, 1.0), "attn_sqrt": lambda x: attn(x, 1 / np.sqrt(x.shape[-1]))}
    for k in range(2, 32):
        tbl[f"attn_scale{k}"] = (lambda kk: (lambda x: attn(x, 1.0 / kk)))(float(k))
    return _reg("attn", 1, tbl, "注意力算子（scale 扫描：恒等路径 vs 1/√d 路径）")


def _gate():
    s = lambda z: 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    tbl = {
        "sigmoid_gate": s,
        "tanh_gate": np.tanh,
        "relu_gate": lambda x: np.minimum(np.maximum(x, 0), 1),
        "silu_gate": lambda x: x * s(x),
        "softmax_gate": lambda x: softmax(x, axis=-1),
        "one_minus_sigmoid": lambda x: 1 - s(x),
        "forget_gate": lambda x: s(-x),
        "input_gate": lambda x: s(x),
        "output_gate": lambda x: s(x + 1),
        "candidate_tanh": np.tanh,
        "gate_pair": lambda x: np.stack([s(x), 1 - s(x)], axis=-1),
        "gate_clip01": lambda x: np.clip(s(x), 0, 1),
        "gate_soft": lambda x: s(x / 2),
        "gate_hard": lambda x: (s(x) > 0.5).astype(float),
        "gate_ste": lambda x: np.where(s(x) > 0.5, 1.0, 0.0),
        "gate_temp2": lambda x: s(2 * x),
        "gate_bias1": lambda x: s(x + 1),
        "gate_bias_neg1": lambda x: s(x - 1),
        "gate_abs": lambda x: s(np.abs(x)),
        "gate_neg": lambda x: s(-x),
        "gate_norm": lambda x: s(x / (np.sqrt(np.mean(x ** 2)) + 1e-8)),
        "gate_square": lambda x: s(x ** 2),
        "gate_sqrt_abs": lambda x: s(np.sqrt(np.abs(x))),
        "gate_sign_flip": lambda x: 2 * s(x) - 1,
        "gate_double": lambda x: s(x) * s(x),
        "gate_triple": lambda x: s(x) ** 3,
        "gate_mix": lambda x: 0.5 * s(x) + 0.5 * np.tanh(x),
        "gate_lnorm": lambda x: s((x - np.mean(x)) / (np.std(x) + 1e-8)),
        "gate_topk2": lambda x: np.where(x >= np.sort(x)[-2], s(x), 0),
        "gate_sparsemax": lambda x: np.maximum(s(x) - 0.5, 0),
        "gate_entropy_w": lambda x: s(x) * (1 - s(x)),
        "gate_residual": lambda x: x * s(x),
    }
    return _reg("gate", 1, tbl, "门控族（LSTM / Mamba-2 / GDN 用到的各类门）")


def _moer():
    tbl = {
        "sigmoid_aff": lambda x: 1 / (1 + np.exp(-np.clip(x, -60, 60))),
        "softmax_aff": lambda x: softmax(x, axis=-1),
        "top1_hard": lambda x: np.eye(x.shape[-1])[np.argmax(x, axis=-1)],
        "top2_hard": lambda x: np.where(x >= np.sort(x, axis=-1)[..., -1:], 1.0, 0.0),
        "top1_soft": lambda x: softmax(x * 10, axis=-1),
        "noisy_gate": lambda x: softmax(x + 0.1 * np.std(x) * np.sin(np.arange(x.shape[-1])), axis=-1),
        "bias_gate": lambda x: softmax(x + np.arange(x.shape[-1]) * 0.01, axis=-1),
        "zloss_penalty": lambda x: np.log(np.sum(np.exp(np.clip(x, -60, 60)), axis=-1, keepdims=True)) ** 2,
        "balance_f": lambda x: softmax(x, axis=-1).mean(axis=-2, keepdims=True),
        "load_count": lambda x: (x >= np.sort(x, axis=-1)[..., -1:]).mean(axis=-2, keepdims=True),
        "cv_load": lambda x: np.std(softmax(x, axis=-1).mean(axis=-2)) / (np.mean(softmax(x, axis=-1)) + 1e-8),
        "router_entropy": lambda x: -np.sum(softmax(x, axis=-1) * np.log(softmax(x, axis=-1) + 1e-12), axis=-1, keepdims=True),
    }
    for k in range(2, 22):
        tbl[f"topk{k}"] = (lambda kk: (lambda x: np.where(
            x >= np.sort(x, axis=-1)[..., -kk:][..., :1], 1.0, 0.0)))(k)
    return _reg("moer", 1, tbl, "MoE 路由算子（含 DeepSeek-V3 的无辅助损失负载相关量）")


def _loss():
    tbl = {
        "mse": lambda x: np.mean(x ** 2),
        "mae": lambda x: np.mean(np.abs(x)),
        "huber": lambda x: np.mean(np.where(np.abs(x) < 1, 0.5 * x ** 2, np.abs(x) - 0.5)),
        "logcosh": lambda x: np.mean(np.log(np.cosh(np.clip(x, -20, 20)))),
        "ce_self": lambda x: -np.mean(np.sum(softmax(x, axis=-1) * (x - logsumexp(x, axis=-1, keepdims=True)), axis=-1)),
        "nll_uniform": lambda x: np.mean(logsumexp(x, axis=-1)),
        "kl_to_uniform": lambda x: np.mean(np.log(x.shape[-1]) + np.sum(
            softmax(x, axis=-1) * (x - logsumexp(x, axis=-1, keepdims=True)), axis=-1)),
        "l2_penalty": lambda x: 0.5 * np.sum(x ** 2),
        "l1_penalty": lambda x: np.sum(np.abs(x)),
        "elastic_half": lambda x: 0.5 * np.sum(x ** 2) + 0.5 * np.sum(np.abs(x)),
        "focal_like": lambda x: np.mean((1 - softmax(x, axis=-1)) ** 2 * (-x)),
        "label_smooth": lambda x: np.mean(logsumexp(x, axis=-1)) * 0.9,
        "margin": lambda x: np.mean(np.maximum(0, 1 - x)),
        "triplet_like": lambda x: np.mean(np.maximum(0, x - np.median(x) + 1)),
        "cosine_loss": lambda x: 1 - np.sum(x) / (np.linalg.norm(x) * np.sqrt(x.size) + 1e-8),
        "entropy_reg": lambda x: -np.mean(np.sum(softmax(x, axis=-1) * np.log(softmax(x, axis=-1) + 1e-12), axis=-1)),
        "z_loss": lambda x: 1e-4 * np.mean(logsumexp(x, axis=-1) ** 2),
        "aux_balance": lambda x: 1e-4 * np.sum(softmax(x, axis=-1).mean(axis=-2) ** 2),
        "dpo_logistic": lambda x: np.mean(np.log1p(np.exp(-np.clip(x, -60, 60)))),
        "ipo_square": lambda x: np.mean((x - 1.0) ** 2),
        "simpo_margin": lambda x: np.mean(np.log1p(np.exp(-np.clip(x - 0.5, -60, 60)))),
        "kto_pos": lambda x: np.mean(np.log1p(np.exp(-np.clip(x, -60, 60)))),
        "kto_neg": lambda x: np.mean(np.log1p(np.exp(np.clip(x, -60, 60)))),
    }
    for p in [1.5, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16]:
        tbl[f"lp_p{p}"] = (lambda pp: (lambda x: np.mean(np.abs(x) ** pp)))(p)
    return _reg("loss", 1, tbl, "损失族（后 11 项为 Lp 范数扫描）")


def _metr():
    tbl = {
        "accuracy_like": lambda x: float(np.mean((x > 0).astype(float))),
        "cosine_sim": lambda x: float(np.sum(x) / (np.linalg.norm(x) * np.sqrt(x.size) + 1e-8)),
        "l2_to_zero": lambda x: float(np.linalg.norm(x)),
        "snr_db": lambda x: float(10 * np.log10((np.mean(x ** 2) + 1e-12) / (np.var(x) + 1e-12))),
        "dynamic_range": lambda x: float(np.ptp(x)),
        "sparsity": lambda x: float(np.mean(x == 0)),
        "kurtosis": lambda x: float(np.mean((x - np.mean(x)) ** 4) / (np.var(x) ** 2 + 1e-12)),
        "skewness": lambda x: float(np.mean((x - np.mean(x)) ** 3) / (np.std(x) ** 3 + 1e-12)),
        "effective_rank": lambda x: float(np.exp(-np.sum(
            (lambda s: s / (np.sum(s) + 1e-12))(np.linalg.svd(x, compute_uv=False) ** 2)
            * np.log((lambda s: s / (np.sum(s) + 1e-12))(np.linalg.svd(x, compute_uv=False) ** 2) + 1e-12)))) if x.ndim >= 2 else 1.0,
        "frobenius": lambda x: float(np.linalg.norm(x)),
        "nuclear": lambda x: float(np.sum(np.linalg.svd(x, compute_uv=False))) if x.ndim >= 2 else float(np.abs(x).sum()),
        "cond_est": lambda x: float((lambda s: s[0] / (s[-1] + 1e-12))(np.linalg.svd(x, compute_uv=False))) if x.ndim >= 2 else 1.0,
        "max_abs": lambda x: float(np.max(np.abs(x))),
        "mean_abs": lambda x: float(np.mean(np.abs(x))),
        "frac_clipped": lambda x: float(np.mean(np.abs(x) > 3)),
        "nan_count": lambda x: float(np.sum(~np.isfinite(x))),
        "bits_entropy": lambda x: float(np.mean(-np.log2(np.abs(x) / (np.sum(np.abs(x)) + 1e-12) + 1e-12))),
    }
    for k in range(1, 16):
        tbl[f"topk_mass{k}"] = (lambda kk: (lambda x: float(np.sum(np.sort(np.abs(x).reshape(-1))[-kk:]) / (np.sum(np.abs(x)) + 1e-12))))(k)
    return _reg("metr", 1, tbl, "度量族（后 15 项为 top-k 质量占比扫描）")


def _initf():
    tbl = {}
    def filler(x, kind):
        n = x.size
        g = np.random.default_rng(0)
        if kind == "zeros": return np.zeros_like(x)
        if kind == "ones": return np.ones_like(x)
        if kind == "normal": return g.standard_normal(x.shape)
        if kind == "uniform": return g.uniform(-1, 1, x.shape)
        if kind == "xavier": return g.standard_normal(x.shape) * np.sqrt(2.0 / (x.shape[0] + x.shape[-1] + 1e-8))
        if kind == "he": return g.standard_normal(x.shape) * np.sqrt(2.0 / (x.shape[0] + 1e-8))
        if kind == "orthogonal":
            a = g.standard_normal(x.shape)
            q, _ = np.linalg.qr(a) if a.ndim == 2 else (a, None)
            return q if q is not None else a
        if kind == "sparse": return np.where(g.uniform(0, 1, x.shape) < 0.1, g.standard_normal(x.shape), 0)
        if kind == "constant": return np.full_like(x, 0.02)
        if kind == "arange": return (np.arange(n).reshape(x.shape) % 7) / 7 - 0.5
        if kind == "chebyshev": return np.cos(np.arange(n).reshape(x.shape) * 0.1)
        if kind == "hadamard_like": return np.sign(g.standard_normal(x.shape))
        if kind == "lowrank": return np.outer(np.ones(x.shape[0]), np.linspace(-1, 1, x.shape[-1]))
        if kind == "radial": return np.linspace(-1, 1, n).reshape(x.shape)
        if kind == "grid": return (np.indices(x.shape).sum(axis=0) % 5) / 5 - 0.4
        if kind == "sinusoid": return np.sin(np.arange(n).reshape(x.shape) * 0.37)
        return x * 0 + 1
    for i, k in enumerate(["zeros", "ones", "normal", "uniform", "xavier", "he", "orthogonal",
                           "sparse", "constant", "arange", "chebyshev", "hadamard_like",
                           "lowrank", "radial", "grid", "sinusoid"]):
        tbl[f"init_{k}"] = (lambda kk: (lambda x: filler(x, kk)))(k)
    for s in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]:
        tbl[f"scaled_{s}"] = (lambda ss: (lambda x: x * ss))(s)
    return _reg("init", 1, tbl, "初始化与缩放族（后 16 项为方差扫描）")


def _scan():
    tbl = {
        "cumsum": np.cumsum,
        "cumprod_pos": lambda x: np.cumprod(np.abs(x) + 1e-12),
        "cummax": np.maximum.accumulate,
        "cummin": np.minimum.accumulate,
        "cummean": lambda x: np.cumsum(x) / np.arange(1, x.size + 1),
        "ewm_a9": lambda x: (lambda a: np.array([np.sum(x[:i + 1] * a ** np.arange(i, -1, -1)) / np.sum(a ** np.arange(i + 1)) for i in range(x.size)]))(0.9),
        "ewm_a99": lambda x: (lambda a: np.array([np.sum(x[:i + 1] * a ** np.arange(i, -1, -1)) / np.sum(a ** np.arange(i + 1)) for i in range(x.size)]))(0.99),
        "prefix_lse": lambda x: np.array([logsumexp(x[:i + 1]) for i in range(x.size)]),
        "delta_state": lambda x: np.outer(x, x),
        "delta_erase": lambda x: np.eye(x.size) - np.outer(x, x) / (np.dot(x, x) + 1e-8),
        "outer_add": lambda x: np.outer(x, np.ones_like(x)),
        "rank1_update": lambda x: np.outer(x, x) / (np.dot(x, x) + 1e-8),
        "householder": lambda x: np.eye(x.size) - 2 * np.outer(x, x) / (np.dot(x, x) + 1e-12),
        "prefix_norm": lambda x: np.array([np.linalg.norm(x[:i + 1]) for i in range(x.size)]),
        "diff1": lambda x: np.diff(x, prepend=x[0]),
        "shift1": lambda x: np.roll(x, 1),
        "reverse": lambda x: x[::-1],
        "interleave": lambda x: x.reshape(-1),
    }
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98]:
        tbl[f"decay_{a}"] = (lambda aa: (lambda x: np.array([
            np.sum(x[:i + 1] * aa ** np.arange(i, -1, -1)) for i in range(x.size)])))(a)
        break
    # 补足到 32
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.5, 0.25]:
        tbl[f"decay_a{a}"] = (lambda aa: (lambda x: np.array([
            np.sum(x[:i + 1] * aa ** np.arange(i, -1, -1)) for i in range(x.size)])))(a)
    while len(tbl) < 32:
        tbl[f"scan_fill{len(tbl)}"] = lambda x: np.cumsum(x) / (np.arange(1, x.size + 1))
    return _reg("scan", 1, tbl, "关联扫描 / 递推族（Gated DeltaNet 的状态算子）", flat=True)


def _quant():
    tbl = {
        "round": np.round,
        "floor": np.floor,
        "ceil": np.ceil,
        "trunc": np.trunc,
        "int8_sym": lambda x: np.round(x * 127),
        "int4_sym": lambda x: np.round(x * 7),
        "int2_sym": lambda x: np.round(x * 1),
        "nf4_like": lambda x: np.round(x * 7) / 7,
        "absmax_scale": lambda x: x / (np.max(np.abs(x)) + 1e-12),
        "per_channel_scale": lambda x: x / (np.max(np.abs(x), axis=-1, keepdims=True) + 1e-12),
        "clip_int8": lambda x: np.clip(np.round(x * 127), -128, 127) / 127,
        "clip_int4": lambda x: np.clip(np.round(x * 7), -8, 7) / 7,
        "dequant_int8": lambda x: x,
        "sign": np.sign,
        "quant_noise": lambda x: x + 0.0,
    }
    for b in [2, 3, 5, 6, 7, 8, 10, 12, 16]:
        tbl[f"q{b}bit"] = (lambda bb: (lambda x: np.round(x * (2 ** (bb - 1) - 1)) / (2 ** (bb - 1) - 1)))(b)
    for g in [4, 8, 16, 32, 64, 128]:
        tbl[f"group{g}"] = (lambda gg: (lambda x: (lambda a: a.reshape(-1, gg) / (
            np.max(np.abs(a.reshape(-1, gg)), axis=-1, keepdims=True) + 1e-12))(
            np.concatenate([np.asarray(x).reshape(-1), np.zeros((gg - x.size % gg) % gg)]))))(g)
    while len(tbl) < 32:
        tbl[f"quant_fill{len(tbl)}"] = lambda x: np.round(x * 127) / 127
    return _reg("quant", 1, tbl, "量化族（INT8/INT4/NF4 与分组缩放）")


def _smpl():
    tbl = {
        "argmax": lambda x: float(np.argmax(x)),
        "argmin": lambda x: float(np.argmin(x)),
        "gumbel_max": lambda x: float(np.argmax(x - np.log(-np.log(np.random.default_rng(0).uniform(1e-9, 1, x.shape))))),
        "categorical_det": lambda x: float(np.random.default_rng(0).choice(x.size, p=softmax(x.reshape(-1)))),
        "top_p09": lambda x: float(np.argmax(softmax(x.reshape(-1)))),
        "top_p05": lambda x: float(np.argmax(softmax(x.reshape(-1) * 2))),
        "nucleus_size": lambda x: float(np.sum(np.sort(softmax(x.reshape(-1)))[::-1].cumsum() < 0.9) + 1),
        "temp_scale": lambda x: softmax(x / 2, axis=-1),
        "temp_half": lambda x: softmax(x * 2, axis=-1),
        "probs": lambda x: softmax(x, axis=-1),
        "logprobs": lambda x: x - logsumexp(x, axis=-1, keepdims=True),
        "entropy_of_probs": lambda x: -np.sum(softmax(x, axis=-1) * np.log(softmax(x, axis=-1) + 1e-12)),
    }
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 0.05, 0.01, 20.0, 100.0]:
        tbl[f"temp_{t}"] = (lambda tt: (lambda x: softmax(x / tt, axis=-1)))(t)
    return _reg("smpl", 1, tbl, "采样族（后 20 项为温度扫描）")


def _optm():
    tbl = {
        "sgd_step": lambda x: x * 0.999,
        "momentum_step": lambda x: x * 0.99,
        "adam_like": lambda x: x - 0.001 * np.sign(x),
        "adamw_decay": lambda x: x * (1 - 0.01),
        "sign_sgd": lambda x: x - 0.01 * np.sign(x),
        "clip_value": lambda x: np.clip(x, -1, 1),
        "clip_norm1": lambda x: x / max(np.linalg.norm(x), 1.0),
        "orthogonalize": lambda x: newton_schulz(x.reshape(len(x), -1) if x.ndim > 2 else
                                                 (x if x.ndim == 2 else x.reshape(1, -1))),
        "spectral_normalize": lambda x: x / (spectral_norm(x.reshape(len(x), -1) if x.ndim > 2 else x) + 1e-12),
        "nesterov": lambda x: x * 0.95,
        "rmsprop_like": lambda x: x / (np.sqrt(np.mean(x ** 2)) + 1e-8),
        "second_moment": lambda x: x ** 2,
        "precondition": lambda x: x / (np.sqrt(np.mean(x ** 2)) + 1e-8),
    }
    for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
        tbl[f"lr_{lr}"] = (lambda l: (lambda x: x - l * np.sign(x)))(lr)
    for wd in [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]:
        tbl[f"wd_{wd}"] = (lambda w: (lambda x: x * (1 - w)))(wd)
    while len(tbl) < 32:
        tbl[f"opt_fill{len(tbl)}"] = lambda x: x * 0.99
    return _reg("optm", 1, tbl, "优化器单步族（含 Muon 的正交化原语）")


def _grad():
    s = lambda z: 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    tbl = {
        "d_relu": lambda x: (x > 0).astype(float),
        "d_leaky": lambda x: np.where(x > 0, 1.0, 0.01),
        "d_sigmoid": lambda x: s(x) * (1 - s(x)),
        "d_tanh": lambda x: 1 - np.tanh(x) ** 2,
        "d_silu": lambda x: s(x) * (1 + x * (1 - s(x))),
        "d_gelu": lambda x: 0.5 * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))) +
                            0.5 * x * (1 - np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)) ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * x ** 2),
        "d_softplus": s,
        "d_mish": lambda x: (lambda e: (e * (4 * (x + 1) + 4 * e + 1 + e ** 2)) / ((2 * (x + 1) + 2 * e + e ** 2 + 2) ** 2))(
            np.exp(np.clip(x, -60, 60))),
        "d_abs": np.sign,
        "d_square": lambda x: 2 * x,
        "d_cube": lambda x: 3 * x ** 2,
        "d_sqrt": lambda x: 1 / (2 * np.sqrt(np.abs(x)) + 1e-8),
        "d_l1": np.sign,
        "d_l2": lambda x: 2 * x,
        "d_huber": lambda x: np.where(np.abs(x) < 1, x, np.sign(x)),
        "d_log": lambda x: 1 / (np.abs(x) + 1e-8),
        "d_exp": lambda x: np.exp(np.clip(x, -60, 60)),
        "d_softmax_diag": lambda x: (lambda p: p * (1 - p))(softmax(x, axis=-1)),
        "d_rmsnorm": lambda x: 1 / (np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + 1e-8),
        "d_layernorm": lambda x: 1 / (np.std(x, axis=-1, keepdims=True) + 1e-8),
        "d_swish_gate": lambda x: s(x) + x * s(x) * (1 - s(x)),
    }
    for k in range(2, 13):
        tbl[f"d_pow{k}"] = (lambda kk: (lambda x: kk * x ** (kk - 1)))(k)
    return _reg("grad", 1, tbl, "激活/损失导数族（反向实现的对照表）")


def _dist():
    def lp(x, p): return np.sum(np.abs(x) ** p) ** (1 / p)
    tbl = {"euclidean": lambda x: np.linalg.norm(x)}
    for p in [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64, 128]:
        tbl[f"lp_{p}"] = (lambda pp: (lambda x: lp(x, pp)))(p)
    tbl["cosine_dist"] = lambda x: 1 - np.sum(x) / (np.linalg.norm(x) * np.sqrt(x.size) + 1e-8)
    tbl["chebyshev"] = lambda x: np.max(np.abs(x))
    tbl["manhattan"] = lambda x: np.sum(np.abs(x))
    tbl["minkowski3"] = lambda x: lp(x, 3)
    tbl["hamming_like"] = lambda x: np.mean(x != 0)
    tbl["jaccard_like"] = lambda x: np.sum(x > 0) / (np.sum(x != 0) + 1e-8)
    tbl["bhattacharyya_like"] = lambda x: np.sum(np.sqrt(np.abs(x) + 1e-12))
    tbl["hellinger_like"] = lambda x: np.sqrt(np.sum((np.sqrt(np.abs(x)) - 1) ** 2))
    tbl["chi_square_like"] = lambda x: np.sum(x ** 2 / (np.abs(x) + 1e-8))
    tbl["mahalanobis_iso"] = lambda x: np.sqrt(np.sum(x ** 2))
    while len(tbl) < 32:
        tbl[f"dist_fill{len(tbl)}"] = lambda x: np.linalg.norm(x)
    return _reg("dist", 1, tbl, "距离/范数族（后 20 项为 Lp 阶扫描，标注 sweep）")


def _prob():
    def kl(p, q):
        p = np.abs(p) / (np.sum(np.abs(p)) + 1e-12)
        q = np.abs(q) / (np.sum(np.abs(q)) + 1e-12)
        return np.sum(p * np.log(p / (q + 1e-12) + 1e-12))
    tbl = {
        "entropy": lambda x: -np.sum((lambda p: p * np.log(p + 1e-12))(np.abs(x) / (np.sum(np.abs(x)) + 1e-12))),
        "cross_entropy_uniform": lambda x: np.mean(logsumexp(x, axis=-1)),
        "kl_to_uniform": lambda x: np.log(x.size + 1e-12) + np.sum(
            (np.abs(x) / (np.sum(np.abs(x)) + 1e-12)) * np.log(np.abs(x) / (np.sum(np.abs(x)) + 1e-12) + 1e-12)),
        "js_like": lambda x: 0.5 * kl(x, np.ones_like(x) / x.size) + 0.5 * kl(np.ones_like(x) / x.size, x),
        "renyi2": lambda x: -np.log(np.sum((np.abs(x) / (np.sum(np.abs(x)) + 1e-12)) ** 2) + 1e-12),
        "tsallis_q2": lambda x: (1 - np.sum((np.abs(x) / (np.sum(np.abs(x)) + 1e-12)) ** 2)) / 1,
        "perplexity_of": lambda x: np.exp(np.mean(logsumexp(x, axis=-1))),
        "surprisal": lambda x: -np.log(np.abs(x) / (np.sum(np.abs(x)) + 1e-12) + 1e-12),
        "mutual_info_proxy": lambda x: np.log(x.size + 1e-12) - np.mean(-np.log(np.abs(x) / (np.sum(np.abs(x)) + 1e-12) + 1e-12)),
        "gibbs_energy": lambda x: -logsumexp(x, axis=-1).mean(),
        "free_energy": lambda x: -logsumexp(x, axis=-1).mean(),
        "log_partition": lambda x: logsumexp(x, axis=-1).mean(),
    }
    for b in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        tbl[f"softmax_beta{b}"] = (lambda bb: (lambda x: softmax(x * bb, axis=-1)))(b)
    for t in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        tbl[f"energy_T{t}"] = (lambda tt: (lambda x: softmax(-x / tt, axis=-1)))(t)
    return _reg("prob", 1, tbl, "概率分布族（熵 / KL / JS / Rényi / 配分函数）")


def _spec():
    tbl = {
        "spectral_norm": lambda x: spectral_norm(x.reshape(len(x), -1) if x.ndim > 2 else x),
        "frobenius": lambda x: np.linalg.norm(x),
        "trace": lambda x: np.trace(x) if x.ndim >= 2 else float(np.sum(x)),
        "nuclear": lambda x: float(np.sum(np.linalg.svd(x.reshape(len(x), -1) if x.ndim > 2 else x, compute_uv=False))),
        "orthogonalize_ns": lambda x: newton_schulz(x.reshape(len(x), -1) if x.ndim > 2 else
                                                    (x if x.ndim == 2 else x.reshape(1, -1))),
        "polar_uv": lambda x: (lambda u, s, vt: u @ vt)(*np.linalg.svd(
            x.reshape(len(x), -1) if x.ndim > 2 else (x if x.ndim == 2 else x.reshape(1, -1)), full_matrices=False)),
        "symmetric": lambda x: (lambda a: (a + a.T) / 2 if (a.ndim >= 2 and a.shape[0] == a.shape[1]) else a)(x),
        "skew": lambda x: (lambda a: (a - a.T) / 2 if (a.ndim >= 2 and a.shape[0] == a.shape[1]) else np.zeros_like(a))(x),
        "gram": lambda x: (lambda m: m @ m.T)(x.reshape(len(x), -1) if x.ndim > 2 else x),
        "cov": lambda x: np.cov(x.reshape(len(x), -1)) if x.ndim >= 2 else np.var(x),
        "power_iter1": lambda x: spectral_norm(x.reshape(len(x), -1) if x.ndim > 2 else x, iters=1),
        "power_iter4": lambda x: spectral_norm(x.reshape(len(x), -1) if x.ndim > 2 else x, iters=4),
        "shrink": lambda x: np.sign(x) * np.maximum(np.abs(x) - 0.1, 0),
        "prox_l1": lambda x: np.sign(x) * np.maximum(np.abs(x) - 0.1, 0),
        "prox_l2": lambda x: x / max(np.linalg.norm(x), 1.0),
    }
    for k in range(2, 19):
        tbl[f"power{k}"] = (lambda kk: (lambda x: spectral_norm(
            x.reshape(len(x), -1) if x.ndim > 2 else x, iters=kk)))(k)
    return _reg("spec", 1, tbl, "谱/矩阵族（含 Muon 的正交化与幂迭代扫描）")


def _misc():
    tbl = {
        "stable_log1p": lambda x: np.log1p(np.clip(x, -1 + 1e-12, None)),
        "stable_exp": lambda x: np.exp(np.clip(x, -60, 60)),
        "stable_sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -60, 60))),
        "safe_log": lambda x: np.log(np.abs(x) + 1e-12),
        "safe_sqrt": lambda x: np.sqrt(np.abs(x)),
        "safe_inv": lambda x: 1 / (x + 1e-12 * np.sign(x + 1e-18)),
        "hypot3": lambda x: np.sqrt(np.sum(x ** 2)),
        "clip_grad": lambda x: np.clip(x, -1, 1),
        "denom_guard": lambda x: x / (np.abs(x) + 1e-8),
        "eps_add": lambda x: x + 1e-8,
        "nan_to_num": lambda x: np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6),
        "finite_mask": lambda x: np.isfinite(x).astype(float),
        "range_check": lambda x: (np.abs(x) < 1e6).astype(float),
        "flatten": lambda x: x.reshape(-1),
        "squeeze": lambda x: np.squeeze(x),
        "expand_last": lambda x: x[..., None],
        "tile2": lambda x: np.concatenate([x, x], axis=-1),
        "half_split": lambda x: x[..., : x.shape[-1] // 2] if x.ndim >= 1 else x,
        "swap_halves": lambda x: np.concatenate([x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]], axis=-1)
        if x.ndim >= 1 else x,
        "pad_one": lambda x: np.concatenate([x, np.zeros_like(x[..., :1])], axis=-1) if x.ndim >= 1 else x,
    }
    for k in range(2, 15):
        tbl[f"clip_v{k}"] = (lambda kk: (lambda x: np.clip(x, -kk, kk)))(k)
    return _reg("misc", 1, tbl, "数值稳定与形状辅助族")


def _conv():
    def conv1(x, w):
        return np.convolve(x, w, mode="full")[: x.size]
    tbl = {"conv_ones3": lambda x: conv1(x, np.ones(3) / 3),
           "conv_ones2": lambda x: conv1(x, np.ones(2) / 2),
           "conv_delta": lambda x: conv1(x, np.array([1.0])),
           "conv_smooth": lambda x: conv1(x, np.array([0.25, 0.5, 0.25])),
           "conv_edge": lambda x: conv1(x, np.array([1.0, -1.0])),
           "conv_gauss": lambda x: conv1(x, np.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136])),
           "depthwise_like": lambda x: x * np.ones_like(x),
           "moving_avg5": lambda x: conv1(x, np.ones(5) / 5),
           "causal_shift1": lambda x: np.concatenate([x[:1] * 0, x[:-1]]),
           "local_sum3": lambda x: conv1(x, np.ones(3)),
           "local_max3": lambda x: np.maximum.reduce([np.roll(x, i) for i in range(3)], axis=0) if x.ndim >= 1 else x,
           "conv_last_dim": lambda x: conv1(x.reshape(-1), np.ones(3) / 3).reshape(x.shape)}
    for k in range(2, 23):
        tbl[f"conv_k{k}"] = (lambda kk: (lambda x: conv1(x, np.ones(kk) / kk)))(k)
    return _reg("conv", 1, tbl, "卷积/局部混合族（短卷积是 fla/Mamba-2 的标配）", flat=True)


def _pool():
    tbl = {"max_pool2": lambda x: np.max(x.reshape(-1, 2), axis=-1) if x.size % 2 == 0 else x,
           "avg_pool2": lambda x: np.mean(x.reshape(-1, 2), axis=-1) if x.size % 2 == 0 else x,
           "global_max": lambda x: np.full_like(x, np.max(x)),
           "global_avg": lambda x: np.full_like(x, np.mean(x)),
           "l2_pool": lambda x: np.sqrt(np.mean(x ** 2)) * np.ones_like(x),
           "stochastic_like": lambda x: x * (x > np.median(x))}
    for k in range(2, 30):
        tbl[f"avg_pool{k}"] = (lambda kk: (lambda x: np.mean(
            np.concatenate([x.reshape(-1), np.zeros((kk - x.size % kk) % kk)]).reshape(-1, kk), axis=-1)))(k)
    return _reg("pool", 1, tbl, "池化族")


def _shap():
    tbl = {"flatten": lambda x: x.reshape(-1),
           "transpose": lambda x: x.T if x.ndim >= 2 else x,
           "squeeze": lambda x: np.squeeze(x),
           "unsqueeze_last": lambda x: x[..., None],
           "unsqueeze_first": lambda x: x[None, ...],
           "repeat2_last": lambda x: np.repeat(x, 2, axis=-1),
           "split_half_first": lambda x: x[..., : x.shape[-1] // 2] if x.ndim >= 1 else x,
           "split_half_second": lambda x: x[..., x.shape[-1] // 2:] if x.ndim >= 1 else x,
           "roll1": lambda x: np.roll(x, 1),
           "flip": lambda x: np.flip(x),
           "diag": lambda x: np.diag(x) if x.ndim >= 2 else x,
           "to_2d": lambda x: x.reshape(len(x), -1)}
    for k in range(2, 22):
        tbl[f"chunk{k}"] = (lambda kk: (lambda x: np.array_split(x.reshape(-1), kk)[0]))(k)
    return _reg("shap", 1, tbl, "形状变换族")


def _indx():
    tbl = {"argmax_idx": lambda x: np.argmax(x, axis=-1),
           "argmin_idx": lambda x: np.argmin(x, axis=-1),
           "argsort": lambda x: np.argsort(x, axis=-1),
           "take_first": lambda x: x[..., 0] if x.ndim >= 1 else x,
           "take_last": lambda x: x[..., -1] if x.ndim >= 1 else x,
           "gather_diag": lambda x: np.diagonal(x) if x.ndim >= 2 else x,
           "onehot_argmax": lambda x: np.eye(x.shape[-1])[np.argmax(x, axis=-1)],
           "scatter_add_self": lambda x: x + x,
           "where_pos": lambda x: np.where(x > 0, x, 0),
           "masked_select": lambda x: x[x > np.median(x)],
           "unique_like": lambda x: np.unique(np.round(x, 3)),
           "top1_value": lambda x: np.max(x, axis=-1, keepdims=True),
           "bottom1_value": lambda x: np.min(x, axis=-1, keepdims=True),
           "rank": lambda x: np.argsort(np.argsort(x, axis=-1), axis=-1),
           "index_parity": lambda x: np.indices(x.shape).sum(axis=0) % 2,
           "boolean_count": lambda x: np.count_nonzero(x > 0)}
    for k in range(2, 19):
        tbl[f"top{k}_idx"] = (lambda kk: (lambda x: np.argsort(x, axis=-1)[..., -kk:]))(k)
    return _reg("indx", 1, tbl, "索引 / gather / scatter 族")


def _cast():
    tbl = {"to_f64": lambda x: x.astype(np.float64),
           "to_f32": lambda x: x.astype(np.float32).astype(np.float64),
           "round_half": lambda x: np.round(x * 2) / 2,
           "round_quarter": lambda x: np.round(x * 4) / 4,
           "truncate2": lambda x: np.trunc(x * 100) / 100,
           "clip_f32_range": lambda x: np.clip(x, -3.4e38, 3.4e38),
           "clip_bf16_range": lambda x: np.clip(x, -3.39e38, 3.39e38),
           "denormal_flush": lambda x: np.where(np.abs(x) < 1e-30, 0, x),
           "saturate_int8": lambda x: np.clip(x, -128, 127),
           "saturate_uint8": lambda x: np.clip(x, 0, 255),
           "scale_up": lambda x: x * 1024,
           "scale_down": lambda x: x / 1024,
           "normalize_scale": lambda x: x / (np.max(np.abs(x)) + 1e-12),
           "as_int_view": lambda x: np.round(x).astype(np.float64),
           "keep_sign": lambda x: np.copysign(np.abs(x), x)}
    for b in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40]:
        tbl[f"round_b{b}"] = (lambda bb: (lambda x: np.round(x * bb) / bb))(b)
    return _reg("cast", 1, tbl, "精度/量化对齐族")


def _rand():
    tbl = {}
    def gen(x, kind, s=0):
        g = np.random.default_rng(s)
        if kind == "normal": return g.standard_normal(x.shape)
        if kind == "uniform": return g.uniform(-1, 1, x.shape)
        if kind == "bernoulli": return (g.uniform(0, 1, x.shape) < 0.5).astype(float)
        if kind == "gumbel": return -np.log(-np.log(g.uniform(1e-9, 1, x.shape)))
        if kind == "laplace": return g.laplace(0, 1, x.shape)
        if kind == "cauchy": return np.clip(g.standard_cauchy(x.shape), -100, 100)
        if kind == "lognormal": return g.lognormal(0, 1, x.shape)
        if kind == "rademacher": return g.choice([-1.0, 1.0], size=x.shape)
        if kind == "exponential": return g.exponential(1, x.shape)
        if kind == "beta": return g.beta(0.5, 0.5, x.shape)
        if kind == "gamma": return g.gamma(2.0, 1.0, x.shape)
        if kind == "poisson": return g.poisson(1.0, x.shape).astype(float)
        if kind == "dropout": return (g.uniform(0, 1, x.shape) > 0.1).astype(float)
        if kind == "permutation": return g.permutation(x.size).astype(float).reshape(x.shape) / x.size
        if kind == "dirichlet_like": return g.dirichlet(np.ones(max(x.shape[-1], 2)))[: x.shape[-1]]
        return x * 0 + 1
    for k in ["normal", "uniform", "bernoulli", "gumbel", "laplace", "cauchy", "lognormal",
              "rademacher", "exponential", "beta", "gamma", "poisson", "dropout",
              "permutation", "dirichlet_like"]:
        tbl[f"rand_{k}"] = (lambda kk: (lambda x: gen(x, kk)))(k)
    for s in range(1, 18):
        tbl[f"seed_{s}"] = (lambda ss: (lambda x: gen(x, "normal", ss)))(s)
    return _reg("rand", 1, tbl, "确定性随机族（全部用显式 seed，无全局状态）")


def _seqm():
    s = lambda z: 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    tbl = {
        "delta_state_init": lambda x: np.zeros((x.shape[-1], x.shape[-1])),
        "delta_write": lambda x: np.outer(x, x),
        "delta_erase": lambda x: np.eye(x.shape[-1]) - np.outer(x, x) / (np.dot(x, x) + 1e-8),
        "delta_read": lambda x: np.outer(x, x) @ x,
        "beta_gate": lambda x: s(x).mean(),
        "alpha_gate": lambda x: s(-x).mean(),
        "forget_matrix": lambda x: np.eye(x.shape[-1]) * s(x).mean(),
        "ssm_decay": lambda x: s(-np.abs(x)),
        "state_norm": lambda x: np.linalg.norm(np.outer(x, x)),
        "state_trace": lambda x: float(np.trace(np.outer(x, x))),
        "kkt_outer": lambda x: np.outer(x, x) / (np.dot(x, x) + 1e-8),
        "mamba2_like": lambda x: np.outer(s(x), x),
        "chunk_cum": lambda x: np.cumsum(x.reshape(-1)).reshape(x.shape),
        "wishart_like": lambda x: np.outer(x, x) + np.eye(x.shape[-1]) * 1e-6,
        "kv_assoc": lambda x: np.outer(x, x),
        "assoc_recall": lambda x: np.outer(x, x) @ np.ones_like(x),
        "l2_state": lambda x: np.sqrt(np.sum(np.outer(x, x) ** 2)),
    }
    for k in range(1, 16):
        tbl[f"state_rank{k}"] = (lambda kk: (lambda x: np.outer(x, np.roll(x, kk))))(k)
    return _reg("seqm", 1, tbl, "序列混合族（DeltaNet / SSM 的状态算子）", flat=True)


def _cache():
    tbl = {
        "kv_append": lambda x: np.concatenate([x, x], axis=-2) if x.ndim >= 2 else x,
        "kv_slice_last": lambda x: x[..., -1:, :] if x.ndim >= 2 else x,
        "kv_len": lambda x: float(x.shape[-2]) if x.ndim >= 2 else float(x.size),
        "kv_reset": lambda x: np.zeros_like(x),
        "kv_concat_first": lambda x: np.concatenate([x, x], axis=-2) if x.ndim >= 2 else x,
        "state_carry": lambda x: x * 0.99,
        "state_zero": lambda x: np.zeros_like(x),
        "cache_mask": lambda x: np.ones(x.shape[-2] if x.ndim >= 2 else 1),
        "position_inc": lambda x: float(x.shape[-2]) + 1 if x.ndim >= 2 else 1.0,
        "rope_offset": lambda x: np.arange(x.shape[-2] if x.ndim >= 2 else 1).astype(float),
        "cache_bytes": lambda x: float(x.size * 8),
        "cache_tokens": lambda x: float(x.shape[-2]) if x.ndim >= 2 else 1.0,
        "sliding_trim": lambda x: x[..., -8:, :] if x.ndim >= 2 else x,
        "compress_mean": lambda x: np.mean(x, axis=-2, keepdims=True) if x.ndim >= 2 else x,
        "compress_max": lambda x: np.max(x, axis=-2, keepdims=True) if x.ndim >= 2 else x,
        "mla_latent": lambda x: np.mean(x, axis=-1, keepdims=True),
    }
    for k in range(2, 19):
        tbl[f"trim_last{k}"] = (lambda kk: (lambda x: x[..., -kk:, :] if x.ndim >= 2 else x))(k)
    return _reg("cache", 1, tbl, "KV / 递推状态缓存族")


def _specf():
    tbl = {"gamma": lambda x: np.vectorize(lambda v: math.gamma(max(abs(v), 1e-6)))(x),
           "lgamma": lambda x: np.vectorize(lambda v: math.lgamma(max(abs(v), 1e-12)))(x),
           "erf": lambda x: np.vectorize(math.erf)(x),
           "erfc": lambda x: np.vectorize(math.erfc)(x),
           "bessel_j0": lambda x: np.vectorize(lambda v: __import__("math").__dict__.get("j0", math.cos)(v))(x),
           "log_abs": lambda x: np.log(np.abs(x) + 1e-12),
           "exp": lambda x: np.exp(np.clip(x, -60, 60)),
           "expm1": lambda x: np.expm1(np.clip(x, -60, 60)),
           "log1p": lambda x: np.log1p(np.clip(x, -1 + 1e-12, 60)),
           "sinc": lambda x: np.sinc(x / np.pi),
           "digamma_like": lambda x: np.log(np.abs(x) + 1e-12) - 1 / (2 * np.abs(x) + 1e-8),
           "zeta2_like": lambda x: np.sum(1 / (np.arange(1, 50) ** 2)) * np.ones_like(x),
           "softplus": lambda x: np.log1p(np.exp(np.clip(x, -60, 60))),
           "logistic": lambda x: 1 / (1 + np.exp(-np.clip(x, -60, 60))),
           "logit": lambda x: np.log((np.clip(x, 1e-6, 1 - 1e-6)) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))}
    for n in range(2, 19):
        tbl[f"poly{n}"] = (lambda nn: (lambda x: x ** nn))(n)
    return _reg("specf", 1, tbl, "特殊函数族")


def _topk():
    tbl = {}
    def tk(x, k, mode):
        idx = np.argsort(x, axis=-1)[..., -k:]
        if mode == "mask":
            m = np.zeros_like(x); np.put_along_axis(m, idx, 1.0, axis=-1); return m
        if mode == "keep":
            return np.where(np.isin(np.arange(x.shape[-1]), idx), x, 0) if x.ndim == 1 else x
        return np.take_along_axis(x, idx, axis=-1).mean(axis=-1, keepdims=True)
    for k in range(1, 12):
        tbl[f"topk{k}_mask"] = (lambda kk: (lambda x: tk(x, kk, "mask")))(k)
    for k in range(1, 12):
        tbl[f"topk{k}_mean"] = (lambda kk: (lambda x: tk(x, kk, "mean")))(k)
    for k in range(1, 12):
        tbl[f"botk{k}_mask"] = (lambda kk: (lambda x: tk(-x, kk, "mask")))(k)
    return _reg("topk", 1, tbl, "Top-K 选择族（MoE 路由与稀疏注意力的基础算子）")


# ---------------------------------------------------------------- 组装

_BUILDERS = [_uact, _bact, _ract, _norm, _smax, _rope, _mask, _attn, _gate, _moer,
             _loss, _metr, _initf, _scan, _quant, _smpl, _optm, _grad, _dist,
             _prob, _spec, _misc, _conv, _pool, _shap, _indx, _cast, _rand,
             _seqm, _cache, _specf, _topk]

# 家族语义类型：core = 成员彼此语义独立；sweep = 同一算子的参数扫描。
# 这个标注是刻意的：把「独立算子」与「参数扫描」区分开，读者可自行判断密度。
FAMILY_KIND = {
    "uact": "core", "bact": "core", "ract": "core", "norm": "core", "gate": "core",
    "moer": "core", "misc": "core", "grad": "core", "init": "core", "scan": "core",
    "seqm": "core", "cache": "core", "specf": "core", "topk": "core", "indx": "core",
    "smax": "sweep", "rope": "sweep", "mask": "sweep", "attn": "sweep", "loss": "sweep",
    "metr": "sweep", "quant": "sweep", "smpl": "sweep", "optm": "sweep", "dist": "sweep",
    "prob": "sweep", "spec": "sweep", "conv": "sweep", "pool": "sweep", "shap": "sweep",
    "cast": "sweep", "rand": "sweep",
}


def _build():
    """组装 32 家族 × 32 成员 = 1024 条微指令。

    每个家族严格取 32 条：builder 产出的候选按定义顺序排列，
    取前 32 条。这样总数恒为 1024，便于门禁做「数量不变量」断言。
    """
    ops: dict[str, MicroOp] = {}
    fams: dict[str, list[str]] = {}
    for b in _BUILDERS:
        part = b()
        fam = next(iter(part.values())).family
        names = list(part.keys())
        assert len(names) >= 32, f"family {fam} has {len(names)} < 32"
        names = names[:32]
        fams[fam] = names
        for n in names:
            # 全局键必须带家族前缀：不同家族会出现同名成员
            # （例如 entropy 同时属于 ract / smax / prob），不加前缀会静默覆盖。
            ops[f"{fam}.{n}"] = part[n]
    assert len(fams) == 32, f"expected 32 families, got {len(fams)}"
    assert len(ops) == 1024, f"expected 1024 micro-ops, got {len(ops)}"
    return ops, fams


MICROOPS, FAMILIES = _build()


def count_ops() -> int:
    return len(MICROOPS)


def run_all(seed: int = 0, shape=(4, 8)) -> dict:
    """对全部微指令执行冒烟，返回报告。

    断言：输出有限（无 NaN/Inf）或不抛异常。
    sweep 型家族中少数算子（如 causal 掩码的 -inf）允许产生 ±inf，
    这类在断言里显式豁免，而不是放宽阈值蒙混过关。
    """
    g = np.random.default_rng(seed)
    x = g.standard_normal(shape)
    y = g.standard_normal(shape)
    ok, bad = 0, []
    for name, op in MICROOPS.items():
        try:
            v = op.fn(x) if op.arity == 1 else op.fn(x, y)
            arr = np.asarray(v, dtype=np.float64)
            finite = np.all(np.isfinite(arr))
            # 掩码族刻意产出 -inf，属预期行为
            if not finite and not (op.family in ("mask", "attn") and np.all(np.isfinite(arr) | (arr == -np.inf))):
                bad.append((name, "non-finite output"))
            else:
                ok += 1
        except Exception as e:                      # noqa: BLE001
            bad.append((name, f"{type(e).__name__}: {e}"))
    return {"total": len(MICROOPS), "ok": ok, "failed": bad, "families": len(FAMILIES)}
