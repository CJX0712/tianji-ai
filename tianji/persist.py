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


def save_checkpoint(path: str, cfg, params: dict[str, np.ndarray],
                    *, meta: dict[str, Any] | None = None,
                    tokenizer_path: str | None = None) -> str:
    """保存 checkpoint，返回权重文件的 sha256。

    落盘两个文件：
        <path>          权重（npz，压缩）
        <path>.meta.json  元数据（配置 + 参数名 + 指纹 + 校验和）
    """
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in params.items()})
    digest = _sha256_file(path)

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
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, sort_keys=True)
    return digest


def load_checkpoint(path: str, *, verify: bool = True):
    """加载 checkpoint，返回 (params, meta_dict)。

    verify=True 时强制校验 sha256 与参数指纹 —— **不一致就抛 CheckpointError**，
    绝不静默加载一个"看起来能用"的损坏文件。
    """
    path = str(path)
    if not os.path.exists(path):
        raise CheckpointError(f"checkpoint 不存在: {path}")
    meta_path = path + ".meta.json"
    if not os.path.exists(meta_path):
        raise CheckpointError(f"缺少元数据文件: {meta_path}（checkpoint 不完整）")

    with open(meta_path, encoding="utf-8") as f:
        info = json.load(f)
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

    with np.load(path, allow_pickle=False) as z:
        params = {k: z[k] for k in z.files}

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
    """只读取元数据，不加载权重（用于快速查看）。"""
    with open(str(path) + ".meta.json", encoding="utf-8") as f:
        return json.load(f)
