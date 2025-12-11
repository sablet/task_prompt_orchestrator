install:
	uv sync
	uv pip install -e .

test:
	uv run pytest tests

# =============================================================================
# Code quality
# =============================================================================
LINT_DIRS = src

.PHONY: format lint

format: ## Format code with ruff
	uv run ruff format $(LINT_DIRS)

lint: ## Lintチェック
	uv run ruff check --fix --unsafe-fixes $(LINT_DIRS)

typecheck: ## 型チェック
	uv run mypy $(LINT_DIRS)

complexity: ## 複雑度チェック
	uv run xenon -b C -m B -a A $(LINT_DIRS)

duplication: ## 重複コードチェック
	npx jscpd --config .jscpd.json

# CALLGRAPH_ARGS ?=
# callgraph: ## 静的コールグラフ解析
# 	uv run python tools/static_callgraph.py $(CALLGRAPH_ARGS)

deps: ## 依存関係チェック
	uv run deptry .

module-lines: ## モジュール行数チェック
	uv run pylint $(LINT_DIRS) --rcfile=pyproject.toml

check: format duplication module-lines lint typecheck complexity #callgraph

loc: ## Pythonファイル行数（output/venv/0行除く、500行以上は赤字）
	@find . -name "*.py" -type f ! -path "./.venv/*" ! -path "./venv/*" ! -path "./*env/*" ! -path "./output/*" | while read f; do \
		lines=$$(wc -l < "$$f"); \
		if [ "$$lines" -gt 0 ]; then \
			if [ "$$lines" -ge 500 ]; then \
				printf "\033[31m%6d %s\033[0m\n" "$$lines" "$$f"; \
			else \
				printf "%6d %s\n" "$$lines" "$$f"; \
			fi; \
		fi; \
	done | sort -t/ -k2,2 -k3,3 -k4,4 -k5,5

