# 天玑 TIANJI · 一键复现
# 全部目标均无需网络（依赖已在镜像/venv 中），可在断网机器上运行。

PY ?= python
DOCKER ?= docker
IMAGE ?= tianji:latest

.DEFAULT_GOAL := help

.PHONY: help install test gates bench quickstart train generate \
        docker-build docker-test docker-shell clean verify-all

help:            ## 显示所有可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:         ## 安装依赖（含 dev）
	$(PY) -m pip install -r requirements.lock.txt

test:            ## 跑单元测试（48 个）
	$(PY) -m pytest tests/ -q --no-header

gates:           ## 跑 16 道交付门禁
	$(PY) -m tianji.cli gates

bench:           ## 跑完整基准（约 2 分钟）
	$(PY) bench/run_bench.py --steps 120

quickstart:      ## 最小可运行示例（约 10 秒）
	$(PY) scripts/quickstart.py

train:           ## 训练（默认内置语料 300 步）
	$(PY) scripts/train.py --steps 300 --out runs/default

generate:        ## 用上一步的 checkpoint 生成文本
	$(PY) scripts/generate.py --ckpt runs/default/model.npz \
		--tokenizer runs/default/tokenizer.txt --prompt "the quick brown" -n 60

# ---------------------------------------------------------------- Docker

docker-build:    ## 构建镜像
	$(DOCKER) build -t $(IMAGE) .

docker-test:     ## 在容器内跑测试与门禁（全新环境验证）
	$(DOCKER) run --rm $(IMAGE) test
	$(DOCKER) run --rm $(IMAGE) gates

docker-shell:    ## 进入容器交互环境
	$(DOCKER) run -it --rm --entrypoint /bin/bash $(IMAGE)

# ---------------------------------------------------------------- 交付自查

verify-all: test gates quickstart    ## 交付前自查：测试 + 门禁 + 最小示例
	@echo ""
	@echo "✅ 自查完成：单元测试、16 道门禁、最小可运行示例"

clean:           ## 清理临时产物
	rm -rf .pytest_cache runs/quickstart bench/results/*.json
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
