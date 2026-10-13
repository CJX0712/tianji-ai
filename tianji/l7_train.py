"""L7 · 训练 —— 调度、数据管道、梯度累积与训练循环。

站在谁之上（零重写）：
    - 余弦学习率 + 线性 warmup    —— GPT-3 / Chinchilla 的标准配方，
                                     也是 nanoGPT / Karpathy 系的默认调度
    - 梯度累积                     —— 用「小显存换大 batch」的标准工程手段，
                                     CPU 上同样成立（小 batch 的 BLAS 效率差）
    - 梯度范数裁剪                  —— Pascanu et al. 2013；对 MoE + 深层混合架构
                                     尤其必要（MoE 的梯度分布重尾）
    - bits-per-byte / perplexity   —— 语言模型的标准可比指标

不变量（门禁复检）：
    T1 调度器单调：warmup 后学习率单调不增，且端点值精确等于设定值
    T2 训练有效：固定 seed 下，N 步后训练损失显著低于初始（相对下降 > 20%）
    T3 过拟合可证：在极小数据集上能把训练损失压到接近 0（说明反向真的在传播）
    T4 确定性：同 seed 两次训练，最终损失逐位一致
    T5 MoE 负载均衡：训练中 expert 负载的变异系数下降（偏置控制器生效）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .l5_moe import LoadBalancer

__all__ = ["cosine_schedule", "wsd_schedule", "DataLoader", "Trainer", "evaluate"]


def cosine_schedule(step: int, total: int, warmup: int, lr_max: float,
                    lr_min: float = 0.0) -> float:
    """线性 warmup + 余弦衰减。

    为什么必须 warmup：训练初期 RMSNorm 增益与 router 偏置还在随机态，
    此时用满学习率会把它们一次性推到很远，损失出现不可逆的尖峰。
    """
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    if step >= total:
        return lr_min
    prog = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * prog))


def wsd_schedule(step: int, total: int, warmup: int, lr_max: float,
                 lr_min: float = 0.0, decay_frac: float = 0.1) -> float:
    """WSD（Warmup-Stable-Decay）—— 2024 年之后大模型预训练的主流替代。

    与余弦的差别：余弦要求**预先知道总步数**，中途改计划就得重训；
    WSD 把「稳定期」与「衰减期」解耦，随时可以从稳定期的 checkpoint
    启动衰减，因此更适合探索式训练。本项目保留两种，由调用方选择。
    """
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    if step < decay_start:
        return lr_max
    prog = (step - decay_start) / max(total - decay_start, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * min(prog, 1.0)))


class DataLoader:
    """从 token 序列随机切 batch。

    为什么用「随机起点 + 连续切片」而不是随机采样单个 token：
    语言建模需要连续的上下文，随机单点采样会破坏序列结构，
    让模型只学到边际分布。
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, batch_size: int, seed: int = 0):
        self.data = np.asarray(tokens, dtype=np.int64)
        self.T = int(seq_len)
        self.B = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def batch(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.data) - self.T - 1
        starts = self.rng.integers(0, max(n, 1), size=self.B)
        x = np.stack([self.data[s:s + self.T] for s in starts])
        y = np.stack([self.data[s + 1:s + self.T + 1] for s in starts])
        return x, y

    def __len__(self) -> int:
        return max(len(self.data) - self.T - 1, 1)


@dataclass
class TrainLog:
    step: int = 0
    losses: list[float] = field(default_factory=list)
    val: list[tuple[int, float]] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    seconds: float = 0.0
    lr_trace: list[float] = field(default_factory=list)


class Trainer:
    def __init__(self, model, opt, loader: DataLoader, *, total_steps: int,
                 warmup: int = 20, lr_max: float = 0.02, lr_min: float = 0.0,
                 accum: int = 1, schedule: str = "cosine",
                 val_loader: DataLoader | None = None, eval_every: int = 0,
                 use_balancer: bool = True):
        self.model = model
        self.opt = opt
        self.loader = loader
        self.total = int(total_steps)
        self.warmup = int(warmup)
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.accum = max(1, int(accum))
        self.schedule = schedule
        self.val_loader = val_loader
        self.eval_every = int(eval_every)
        self.log = TrainLog()
        self._grad_accum: dict[str, np.ndarray] | None = None
        if use_balancer:
            cfg = model.cfg
            for i in range(cfg.n_layers):
                if model.is_moe_layer(i):
                    model.balancers[i] = LoadBalancer(cfg.n_experts)

    def lr_at(self, step: int) -> float:
        fn = cosine_schedule if self.schedule == "cosine" else wsd_schedule
        return fn(step, self.total, self.warmup, self.lr_max, self.lr_min)

    def step(self, step: int) -> float:
        m = self.model
        total_loss = 0.0
        for k in range(self.accum):
            x, y = self.loader.batch()
            out = m.forward(x, y)
            G = m.backward()
            total_loss += out["loss"] / self.accum
            if self._grad_accum is None:
                self._grad_accum = {k2: v.copy() for k2, v in G.items()}
            else:
                for k2, v in G.items():
                    if k2 in self._grad_accum:
                        self._grad_accum[k2] += v
                    else:
                        self._grad_accum[k2] = v.copy()
        for k2 in self._grad_accum:
            self._grad_accum[k2] /= self.accum
        lr = self.lr_at(step)
        # 学习率注入：Muon/AdamW 都从 optimizer 自身的 lr 读取，这里统一覆写
        self.opt.muon.lr = lr
        self.opt.adam.lr = lr
        self.opt.step(m.P, self._grad_accum)
        self._grad_accum = None
        self.log.grad_norms.append(self.opt.last_grad_norm)
        self.log.lr_trace.append(lr)
        return total_loss

    def run(self, steps: int | None = None, verbose: bool = True,
            print_every: int = 20) -> TrainLog:
        steps = steps or self.total
        t0 = time.time()
        for s in range(steps):
            loss = self.step(s)
            self.log.losses.append(loss)
            self.log.step = s + 1
            if verbose and (s % print_every == 0 or s == steps - 1):
                msg = f"step {s:>5}/{steps}  loss {loss:.4f}  lr {self.lr_at(s):.5f}"
                if self.log.grad_norms:
                    msg += f"  |g| {self.log.grad_norms[-1]:.3f}"
                print(msg, flush=True)
            if self.val_loader is not None and self.eval_every and (s % self.eval_every == 0):
                v = evaluate(self.model, self.val_loader, n_batch=4)
                self.log.val.append((s, v))
                if verbose:
                    print(f"        val loss {v:.4f}", flush=True)
        self.log.seconds = time.time() - t0
        return self.log


def evaluate(model, loader: DataLoader, n_batch: int = 8) -> float:
    tot, n = 0.0, 0
    for _ in range(n_batch):
        x, y = loader.batch()
        out = model.forward(x, y)
        tot += out["aux"].get("loss_main", out["loss"])
        n += 1
    return tot / max(n, 1)


def perplexity(loss: float) -> float:
    return float(np.exp(loss))


def bits_per_byte(loss: float, bytes_per_token: float) -> float:
    """bits-per-byte —— 跨分词器可比的指标。

    为什么不能只报 perplexity：perplexity 依赖词表大小与分词粒度，
    换一个 BPE 词表数字就不可比。bpb 把「每字节需要的比特数」作为单位，
    是跨实现对比的唯一公平口径。
    """
    return float(loss / (np.log(2.0) * bytes_per_token))
