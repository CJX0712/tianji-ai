#!/usr/bin/env bash
# 天玑 TIANJI · 一键复现（无需 Docker）
#
#   ./scripts/reproduce.sh            # 完整复现：建环境 → 测试 → 门禁 → 示例 → 训练 → 推理
#   ./scripts/reproduce.sh --quick    # 只跑测试 + 门禁 + 最小示例
#
# 依赖：Python 3.10+（自带 venv）。全程无需 root，不污染系统环境。
# 为什么提供这个脚本：Docker 需要能访问 Docker Hub；在网络受限的机器上，
# 本脚本提供等价的「一条命令复现」能力。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
VENV=".venv"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

log() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }

log "1/6 创建虚拟环境"
if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install -q --disable-pip-version-check -r requirements.lock.txt
python -c "import numpy; print('numpy', numpy.__version__)"

log "2/6 单元测试"
python -m pytest tests/ -q --no-header

log "3/6 交付门禁（10 道）"
python -m tianji.cli gates

log "4/6 最小可运行示例"
python scripts/quickstart.py

if [ "$QUICK" = "1" ]; then
    log "完成（--quick 模式）"
    exit 0
fi

log "5/6 训练（120 步，约 15 秒）"
python scripts/train.py --steps 120 --d-model 64 --layers 4 \
    --seq 32 --batch 8 --out runs/reproduce

log "6/6 推理"
python scripts/generate.py --ckpt runs/reproduce/model.npz \
    --tokenizer runs/reproduce/tokenizer.txt \
    --prompt "to be or not" -n 50 --temperature 0.7

printf '\n\033[32m✅ 全部复现步骤通过\033[0m\n'
printf '产物：runs/reproduce/{model.npz, model.npz.meta.json, tokenizer.txt, train_log.json}\n'
