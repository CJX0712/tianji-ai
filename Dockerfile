# 天玑 TIANJI —— 一键复现环境
#
# 构建： docker build -t tianji:latest .
# 运行： docker run --rm tianji:latest                  # 默认跑 quickstart 自检
#        docker run --rm tianji:latest pytest           # 跑测试
#        docker run --rm tianji:latest train --steps 200
#
# 设计取舍：
#   - 基础镜像用 python:3.11-slim（而非 alpine）：NumPy 在 alpine 上需要
#     自行编译（musl + OpenBLAS），构建时间从 30s 变成 10 分钟以上，
#     且容易在不同机器上产出不同的浮点结果，破坏"可复现"这条硬约束。
#   - 依赖用锁定的 requirements.lock.txt 而不是 requirements.txt：
#     浮点运算的逐位一致性依赖具体的 BLAS 与 NumPy 版本。
#   - 不装 GPU 栈：本项目是纯 CPU/NumPy 实现，装 CUDA 只会让镜像大 3GB。

FROM python:3.11-slim

LABEL org.opencontainers.image.title="tianji-ai"
LABEL org.opencontainers.image.description="天玑 TIANJI · 混合架构语言模型全栈"
LABEL org.opencontainers.image.authors="晨星 (Chenxing)"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/CJX0712/tianji-ai"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=4

WORKDIR /app

# 依赖先复制、先安装：利用 Docker 层缓存，改代码不触发重装
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.lock.txt

COPY pyproject.toml README.md LICENSE ./
COPY tianji ./tianji
COPY tests ./tests
COPY bench ./bench
COPY scripts ./scripts

# 非 root 运行（最小权限原则）
RUN useradd -m -u 1000 tianji && mkdir -p /app/runs && chown -R tianji:tianji /app
USER tianji

# 入口：封装了 quickstart / gates / bench / pytest / train / generate
ENTRYPOINT ["python", "-m", "tianji.cli"]
CMD ["gates"]

# 冒烟：构建后立即验证 import 与门禁可用（失败则镜像不可用）
HEALTHCHECK --interval=60s --timeout=120s --start-period=20s --retries=2 \
    CMD python -c "import sys; import tianji; from tianji.verify import run_all_gates; \
        sys.exit(0 if run_all_gates(steps=10).all_green else 1)"
