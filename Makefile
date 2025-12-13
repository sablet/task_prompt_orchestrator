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

# check: format duplication module-lines lint typecheck complexity
check: format duplication lint-filesize lint typecheck complexity

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

lint-filesize: ## ファイルサイズチェック（500行超:WARN, 1000行超:ERROR）
	@has_error=0; has_warn=0; \
	find $(LINT_DIRS) -name "*.py" -type f | while read f; do \
		lines=$$(wc -l < "$$f"); \
		if [ "$$lines" -gt 1000 ]; then \
			printf "\033[31mERROR: %s (%d lines)\033[0m\n" "$$f" "$$lines"; \
			echo "error" > /tmp/lint-filesize-status; \
		elif [ "$$lines" -gt 500 ]; then \
			printf "\033[33mWARN:  %s (%d lines)\033[0m\n" "$$f" "$$lines"; \
			[ ! -f /tmp/lint-filesize-status ] && echo "warn" > /tmp/lint-filesize-status; \
		fi; \
	done; \
	status=$$(cat /tmp/lint-filesize-status 2>/dev/null); rm -f /tmp/lint-filesize-status; \
	if [ "$$status" = "error" ]; then echo "\n1000行超のファイルがあります"; exit 1; \
	elif [ "$$status" = "warn" ]; then echo "\n500行超のファイルがあります（リファクタリング推奨、1000行オーバーは必須）"; fi

