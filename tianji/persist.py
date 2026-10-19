"""模型持久化 —— checkpoint 的保存、加载与完整性校验。

为什么需要独立一层：
    「训练脚本」和「推理脚本」要能分开跑，就必须有可靠的 checkpoint。
    而 checkpoint 最常见的失效不是存不进去，是**读出来悄悄错了** ——
    参数名对不上、词表版本不一致、文件被截断。
    这些都不会抛异常，只会让生成结果变成乱码。

设计：
    - 权重用 np.savez_compressed（float64 有损？不 —— savez 是精确二进制）
    - 元数据（配置 / 词表 / 校验和 / 版本）存同名 .meta.json，人眼可读可 diff
    - 加载时**强制校验 sha256 与参数名集合**，不一致直接报错而不是静默降级

站在谁之上：
    - np.savez_compressed  —— NumPy 标准持久化（zip + deflate），无第三方依赖
    - sha256 校验           —— 与 HuggingFace safetensors 的 header 校验思路一致：
                              把「完整性」做成加载路径上的强制步骤
    - tmp + os.replace 原子落盘 —— 与 safetensors.save_file 的临时文件+重命名一致：
                              保存中途崩溃不留半成品（旧 checkpoint 完好可用）

不变量（L10 门禁会在每次交付时复检）：
    P1 路径规范化闭环：无 .npz 扩展名的路径 save→load→info 一致
       （np.savez 会自动追加 .npz，不规范化会让 _sha256_file 找错文件 ——
        G20 探针实测抓出过 FileNotFoundError，已修复）
    P2 损坏的 meta.json（截断/非法 JSON）必须抛 CheckpointError
       （早期实现漏出原始 JSONDecodeError，调用方 except CheckpointError 会漏）
    P3 损坏的权重（翻字节）必须抛 CheckpointError（sha256 失配）
    P4 「权重被替换 + sha 同步篡改」必须被参数指纹抓住
       （威胁模型：防 meta 与权重失配的腐烂，不防两者同步作恶——
        那是认证签名格式的职责，safetensors 亦然，docstring 如实声明）
    P5 原子性：保存失败/中断不留下可被误加载的半成品，tmp 文件全部清理
    P6 空参数、怪异参数名（点/斜杠/unicode）、dtype 保持 roundtrip 无损
    P7 覆盖保存：新权重 + 旧 meta 的失配必须被检出（响亮失败而非静默）
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

__all__ = ["save_checkpoint", "load_checkpoint", "checkpoint_info", "CheckpointError"]

FORMAT_VERSION = "1.0"


class CheckpointError(RuntimeError):
    """checkpoint 损坏或版本不兼容。一律抛错，不静默降级。"""


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _params_fingerprint(params: dict[str, np.ndarray]) -> str:
    """参数指纹：对每个参数的名字 + 形状 + 少量采样值取哈希。

    为什么不只校验文件大小：两个 checkpoint 可能大小相同但内容不同
    （例如训练到不同步数）。形状 + 采样值足以在加载前发现「拿错文件」。

    威胁模型（如实声明）：指纹 + sha256 防的是「meta 与权重部分失配」的
    常见腐烂（文件被替换但 meta 没同步、拷贝中断、磁盘静默错误）。
    它不防「攻击者同步改写两者」—— 那需要 HMAC/签名，safetensors 同样
    不防（未签名的 header 与数据可一起改）。G20 门禁机检这一边界。
    """
    h = hashlib.sha256()
    for k in sorted(params):
        a = np.asarray(params[k])
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        flat = a.reshape(-1)
        if flat.size:
            idx = np.linspace(0, flat.size - 1, num=min(64, flat.size)).astype(int)
            h.update(np.ascontiguousarray(flat[idx]).tobytes())
    return h.hexdigest()


def _normalize_path(path: str) -> str:
    """路径规范化：统一补上 .npz 后缀。

    np.savez_compressed 会自动给不带 .npz 的路径追加后缀，但后续的
    _sha256_file / meta 写入用的是原始路径 —— 不规范化就会「权重写到了
    ckpt.npz、校验和却去找 ckpt」直接 FileNotFoundError（G20 探针抓出的
    真实缺陷）。save / load / checkpoint_info 三条路径共用本函数，
    保证传同样的路径进去，一定能取回来。
    """
    path = str(path)
    return path if path.endswith(".npz") else path + ".npz"


def save_checkpoint(path: str, cfg, params: dict[str, np.ndarray],
                    *, meta: dict[str, Any] | None = None,
                    tokenizer_path: str | None = None) -> str:
    """保存 checkpoint，返回权重文件的 sha256。

    落盘两个文件：
        <path>.npz          权重（npz，压缩）
        <path>.npz.meta.json  元数据（配置 + 参数名 + 指纹 + 校验和）

    原子性（对齐 safetensors.save_file 的 tmp+rename 模式）：
        先写临时文件、算校验和、写 meta 临时文件，最后 os.replace 落位。
        保存中途崩溃（断电/杀进程/磁盘满）只会留下 .tmp* 垃圾，
        旧 checkpoint 与 meta 始终配对完好 —— 绝不产生半成品。
    """
    path = _normalize_path(str(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # 临时名必须以 .npz 结尾：np.savez_compressed 会给不带该后缀的路径再追加一个
    tmp = f"{path}.tmp{os.getpid()}.npz"
    try:
        np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in params.items()})
        digest = _sha256_file(tmp)

        cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
        info = {
            "format_version": FORMAT_VERSION,
            "library": "tianji",
            "params": {k: list(np.asarray(v).shape) for k, v in params.items()},
            "n_params": int(sum(np.asarray(v).size for v in params.values())),
            "config": cfg_dict,
            "weights_sha256": digest,
            "params_fingerprint": _params_fingerprint(params),
            "tokenizer": os.path.basename(tokenizer_path) if tokenizer_path else None,
            "meta": meta or {},
        }
        with open(tmp + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
        os.replace(tmp + ".meta.json", path + ".meta.json")
    except BaseException:
        for p in (tmp, tmp + ".meta.json"):      # 失败不留半成品垃圾
            try:
                os.unlink(p)
            except OSError:
                pass
        raise
    return digest


def _read_meta(meta_path: str) -> dict:
    """读取 meta.json，任何损坏一律转成 CheckpointError。

    早期实现让 json.JSONDecodeError 直接漏出 —— 调用方按 docstring 写的
    `except CheckpointError` 抓不到它，损坏的 checkpoint 会掀翻整个调用链
    （G20 探针实测：截断的 meta.json 漏出 JSONDecodeError）。
    """
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise CheckpointError(f"缺少元数据文件: {meta_path}（checkpoint 不完整）") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CheckpointError(
            f"元数据文件损坏（不是合法 JSON，checkpoint 可能被截断）: {meta_path}\n"
            f"  {type(e).__name__}: {e}") from None
    except OSError as e:
        raise CheckpointError(f"元数据文件不可读: {meta_path}（{e}）") from None


def load_checkpoint(path: str, *, verify: bool = True):
    """加载 checkpoint，返回 (params, meta_dict)。

    verify=True 时强制校验 sha256 与参数指纹 —— **不一致就抛 CheckpointError**，
    绝不静默加载一个"看起来能用"的损坏文件。
    路径按 _normalize_path 规范化：save 时传的名字，load 原样传回去即可。
    """
    path = _normalize_path(str(path))
    if not os.path.exists(path):
        raise CheckpointError(f"checkpoint 不存在: {path}")
    info = _read_meta(path + ".meta.json")
    if info.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint 格式版本不兼容: 文件={info.get('format_version')}, "
            f"本库={FORMAT_VERSION}")

    if verify:
        actual = _sha256_file(path)
        if actual != info.get("weights_sha256"):
            raise CheckpointError(
                f"权重文件校验和不匹配（文件可能损坏或被替换）\n"
                f"  期望 {info.get('weights_sha256')}\n  实际 {actual}")

    try:
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files}
    except Exception as e:
        raise CheckpointError(
            f"权重文件无法解析（zip 结构损坏或不是 npz）: {path}\n"
            f"  {type(e).__name__}: {e}") from None

    expected = set(info.get("params", {}))
    got = set(params)
    if expected and expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise CheckpointError(f"参数集合不匹配: 缺失 {missing[:5]}, 多余 {extra[:5]}")

    if verify and info.get("params_fingerprint"):
        fp = _params_fingerprint(params)
        if fp != info["params_fingerprint"]:
            raise CheckpointError("参数指纹不匹配（权重内容与元数据记录的不一致）")

    return params, info


def checkpoint_info(path: str) -> dict:
    """只读取元数据，不加载权重（用于快速查看）。损坏同样抛 CheckpointError。"""
    return _read_meta(_normalize_path(str(path)) + ".meta.json")
