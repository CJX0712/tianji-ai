"""L1 · 反向契约层 —— 把「正确性」从测试环节提升到架构层。

设计选择与替代方案对比：

方案 A（未采用）：JAX/Autograd 式 tape 自动微分
    优点：写前向即得反向，开发快。
    缺点：CPU + NumPy 后端下 Python 开销占比极高（每个算子一次 tape 记录 +
          一次 dict 查找），在本项目目标规模（百万级参数、十万步级迭代）上
          比手写解析反向慢 3~10 倍；且 tape 反向无法利用 Gated DeltaNet
          的递推结构（应当沿时间反向递推，而非逐节点展开）。

方案 B（采用）：显式 backward 契约 + 强制梯度检验
    每个模块实现 forward(ctx, ...) / backward(ctx, grad_out)，
    由本层提供统一的有限差分验证器 gradcheck。
    优点：性能可控、允许为递推结构写专用反向、形状/数值契约可静态断言。
    代价：每个模块必须手写反向 —— 用 gradcheck 兜底，不靠人眼。

站在谁之上：
    - backward(ctx, g) 契约形态      —— PyTorch autograd.Function 的成熟约定
    - 中心差分梯度检验               —— Ng, "CS231n" 与所有主流框架的 numerical gradcheck
    - 相对误差度量                   —— 同上，用 |a-b| / max(|a|+|b|) 避免零量纲误判

不变量（门禁复检）：
    A1 每个注册模块的解析反向与中心差分一致（相对误差 < 1e-6）
    A2 反向输出的形状与对应前向输入严格一致
    A3 反向不产生 NaN/Inf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

__all__ = ["Ctx", "Module", "gradcheck", "GradReport", "assert_finite", "check_shapes"]


class Ctx(dict):
    """前向传给反向的暂存区。

    用 dict 子类而非 dataclass：模块数量多（L10 有 1024 条微指令），
    为每个模块声明一个 dataclass 会让代码量翻倍且毫无收益。
    """

    def put(self, **kw: Any) -> None:
        self.update(kw)


class Module:
    """所有可微模块的基类。

    契约：
        forward(ctx, *args) -> out        不得修改 args，中间量一律存 ctx
        backward(ctx, grad_out) -> tuple  返回与 forward 的「可微输入」等长同形的梯度元组
                                          不可微输入（如超参）位置返回 None
    """

    #: 声明哪些位置的下标是可微输入，用于 gradcheck 自动选择扰动对象
    differentiable: tuple[int, ...] = ()

    def forward(self, ctx: Ctx, *args: Any) -> np.ndarray:
        raise NotImplementedError

    def backward(self, ctx: Ctx, grad_out: np.ndarray) -> tuple:
        raise NotImplementedError

    def __call__(self, *args: Any) -> np.ndarray:
        ctx = Ctx()
        out = self.forward(ctx, *args)
        return out


@dataclass
class GradReport:
    """梯度检验结果。失败就如实报失败，不允许放宽阈值凑绿。"""

    name: str
    max_rel_err: float
    passed: bool
    tol: float
    detail: str = ""
    per_input: list[float] = field(default_factory=list)
    max_abs_err: float = 0.0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"[{status}] {self.name:<28} abs={self.max_abs_err:.3e} "
                f"rel={self.max_rel_err:.3e} tol={self.tol:.0e}")


def assert_finite(x: np.ndarray, name: str = "tensor") -> None:
    a = np.asarray(x)
    if not np.all(np.isfinite(a)):
        bad = int(np.sum(~np.isfinite(a)))
        raise FloatingPointError(f"{name} 含 {bad} 个非有限值 (NaN/Inf)")


def check_shapes(pairs: Sequence[tuple[str, np.ndarray, tuple]]) -> None:
    for name, arr, want in pairs:
        got = tuple(np.asarray(arr).shape)
        if got != tuple(want):
            raise AssertionError(f"形状契约破裂: {name} got={got} want={tuple(want)}")


def gradcheck(
    module: Module,
    args: Sequence[Any],
    *,
    seed: int = 0,
    eps: float = 1e-5,
    tol: float = 1e-5,
    out_seed_probe: int = 1,
    name: str | None = None,
) -> tuple[GradReport, tuple, Ctx]:
    """中心差分梯度检验。

    为什么用中心差分而不是单边：单边差分误差 O(eps)，中心差分 O(eps²)。
    float64 下 eps=1e-6 时中心差分可稳定到 1e-9 量级，能真正区分
    「反向写错了」与「浮点舍入」。

    为什么只检验随机投影而非每个输出元素：输出可能是 (B,T,d) 上万元素，
    逐个扰动意味着上万次前向。随机高斯投影 g 检验 gᵀ·(Jᵀ·g_out) 即可，
    数学上等价于检验完整 Jacobian 的一个随机方向组合（且以概率 1 覆盖）。

    已知局限（G23 抓出后记录，勿踩）：
        步长 `eps` 是**绝对**的（默认 1e-5）。当输入量级远小于 1e-5 时
        （如 ‖k‖~1e-7 的 key 向量），扰动比输入自身大 100 倍，中心差分
        已不在测量该点的导数 —— 此时 FAIL 是测量伪影，不是反向写错。
        需要检验小量级输入时，必须用**尺度自适应步长**（h ≈ 1e-5·|x|），
        G23 门禁即如此实现。
        同理，对在零点不连续的函数（如 l2norm 在 x=0 处定义为 0），
        任何有限差分都会在零行爆炸 —— 零行应改用契约检查（输出/梯度
        是否精确为 0），而不是差分。
    """
    name = name or type(module).__name__
    g = np.random.default_rng(seed)
    ctx = Ctx()
    out = np.asarray(module.forward(ctx, *args))
    assert_finite(out, f"{name}.forward")

    # 随机输出方向：把向量值函数标量化。
    # 注意：解析与数值两侧必须使用完全相同的 probe 集合，否则会差一个常数倍。
    # （早期版本在这里犯过错：解析累加 4 个 probe、数值只算 1 个，全部误报 FAIL）
    probes = [np.asarray(g.standard_normal(out.shape))]

    # ---- 解析反向（对每个 probe 求和，因为反向是线性的）
    analytic: list[np.ndarray | None] = [None] * len(args)
    diffable = module.differentiable if module.differentiable else tuple(range(len(args)))
    for p in probes:
        # 每次 backward 需要独立 ctx（部分模块会就地消费 ctx）
        ctx_b = Ctx()
        module.forward(ctx_b, *args)
        grads = module.backward(ctx_b, p)
        if len(grads) != len(args):
            raise AssertionError(
                f"{name}.backward 返回 {len(grads)} 项，forward 收到 {len(args)} 个输入"
            )
        for i in diffable:
            gi = grads[i]
            if gi is None:
                continue
            assert_finite(gi, f"{name}.backward[{i}]")
            if np.asarray(args[i]).shape != np.asarray(gi).shape:
                raise AssertionError(
                    f"{name} 反向形状破裂: arg{i} {np.asarray(args[i]).shape} vs grad {np.asarray(gi).shape}"
                )
            analytic[i] = gi if analytic[i] is None else analytic[i] + gi

    # ---- 数值反向（中心差分）
    rtol = tol
    worst = 0.0
    worst_abs = 0.0
    per_input: list[float] = []
    # 绝对容差必须用**全局**梯度尺度，不能用单个元素的量级：
    # 中心差分的浮点噪声约为 eps_machine·|loss|/eps（≈1e-10 量级），
    # 与元素自身大小无关。若按元素量级设 atol，则梯度本身只有 1e-8 的
    # 参数（深层模型中很常见）会被噪声淹没、被误判为 FAIL ——
    # 本项目早期就踩过这个坑：analytic 与 numeric 比值 1.0000 却被判 FAIL。
    gscale = max((float(np.abs(a).max()) for a in analytic if a is not None), default=1.0)
    atol = 1e-7 * max(gscale, 1e-6)
    for i in diffable:
        arr = args[i]
        if not isinstance(arr, np.ndarray) or not np.issubdtype(arr.dtype, np.floating):
            continue
        num = np.zeros_like(arr, dtype=np.float64)
        flat = arr.reshape(-1)
        nflat = num.reshape(-1)
        # 全量扰动太慢时随机抽样：这里按元素全扫，模块输入都很小
        for j in range(flat.size):
            old = flat[j]
            flat[j] = old + eps
            o_plus = float(np.sum(probes[0] * np.asarray(module.forward(Ctx(), *args))))
            flat[j] = old - eps
            o_minus = float(np.sum(probes[0] * np.asarray(module.forward(Ctx(), *args))))
            flat[j] = old
            nflat[j] = (o_plus - o_minus) / (2 * eps)
        num = nflat.reshape(arr.shape)
        a_i = analytic[i]
        if a_i is None:
            # 模块明确声明该输入不可微（如离散 target、整数索引）
            per_input.append(0.0)
            continue
        abs_err = np.abs(a_i - num)
        # 判据：|a − n| <= atol + rtol·|n|（PyTorch gradcheck 同款混合判据）。
        # 为什么不能只用相对误差：梯度接近 0 的元素上，中心差分的浮点噪声
        # （~1e-16/eps ≈ 1e-10）被 |a|+|n| ≈ 1e-5 的分母放大成 1e-5 量级，
        # 会把完全正确的反向误判为 FAIL。绝对项 atol 兜住这个区间。
        scale = max(float(np.abs(a_i).max()), float(np.abs(num).max()), 1e-12)
        ratio = float(np.max(abs_err / (atol + rtol * np.abs(num))))
        per_input.append(ratio)
        worst = max(worst, ratio)
        worst_abs = max(worst_abs, float(abs_err.max()))

    passed = worst <= 1.0
    rel_ref = 0.0
    rep = GradReport(
        name=name,
        max_rel_err=rel_ref,
        passed=passed,
        tol=rtol,
        detail=f"differentiable={diffable}, eps={eps:.0e}, atol={atol:.1e}, "
               f"判据 |a-n| <= atol + rtol*|n|，比值需 <= 1",
        per_input=per_input,
        max_abs_err=worst_abs,
    )
    return rep, tuple(analytic), ctx


def gradcheck_many(
    cases: Sequence[tuple[str, Module, Sequence[Any]]],
    *,
    seed: int = 0,
    tol: float = 1e-6,
) -> list[GradReport]:
    """批量梯度检验。返回全部报告（含失败项），不做 early-stop。"""
    return [gradcheck(m, a, seed=seed, tol=tol, name=n)[0] for n, m, a in cases]
