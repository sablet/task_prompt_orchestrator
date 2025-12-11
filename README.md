# Task Prompt Orchestrator

Claude Agent SDKを使用したマルチステップタスク自動化ツール。YAMLで定義されたタスクを順番に実行し、各タスクの検証結果に応じてフローを制御する。

## クイックスタート

### 1. タスク定義ファイルを作成

Claude Code でスラッシュコマンドを使うと、目的からタスク定義ファイルを自動生成できる：

```bash
# Claude Code で実行
/plan_task_prompts make check を通るようにリファクタリングがしたい
```

手動で作成する場合：

```yaml
# tasks.yaml
tasks:
  - id: task_1
    name: タスク名
    instruction: |
      実行内容の指示
    validation:
      - "検証条件1"
      - "検証条件2"

  - id: task_2
    name: 次のタスク
    depends_on: [task_1]
    instruction: |
      task_1完了後に実行
    validation:
      - "検証条件"
```

### 2. 実行

```bash
task-orchestrator run tasks.yaml
```

## インストール

### 開発版を別プロジェクトで使う場合

```bash
# editable インストール（コード変更が即座に反映）
uv add --editable /path/to/task_prompt_orchestrator
```

### このリポジトリで開発する場合

```bash
uv sync
uv pip install -e .
```

## コマンド

```bash
# タスク実行
task-orchestrator run tasks.yaml

# dry-run (プロンプト確認のみ)
task-orchestrator run tasks.yaml --dry-run

# サンプルYAML生成
task-orchestrator sample > /tmp/tasks.yaml
```

### オプション

| オプション | 説明 |
|-----------|------|
| `--max-retries N` | タスクあたりの最大リトライ回数 (default: 3) |
| `--max-total-retries N` | 全体の最大リトライ回数 (default: 10) |
| `--cwd PATH` | 作業ディレクトリ |
| `--model MODEL` | 使用モデル |
| `--no-web` | WebFetch/WebSearch無効化 |
| `--bypass-permissions` | 全ツール自動承認 (要注意) |
| `--dry-run` | 実行せずプロンプト表示 |
| `-o FILE` | 結果JSONの出力先 |

## フロー制御

```
instruction → validation
    ↓
approved → 次のタスクへ
declined → feedbackを付けてリトライ (max_retries まで)
permission_denied → 即終了 (リトライなし)
```

## 履歴管理・途中再開

実行履歴は `.task-orchestrator-history/` に自動保存され、失敗時に途中から再開できる。

```bash
# 未完了の履歴一覧
task-orchestrator history

# 途中から再開
task-orchestrator resume <history_id>

# 特定ポイントから再開
task-orchestrator resume <history_id> --from task2_validation
```

再開ポイントは `<task_id>_instruction` または `<task_id>_validation` の形式。

### その他の履歴操作

```bash
# 全履歴一覧
task-orchestrator history --all

# 詳細表示
task-orchestrator history --show <history_id>

# 履歴削除
task-orchestrator history --delete <history_id>

# 履歴無効化
task-orchestrator run tasks.yaml --no-history
```

## Permission Mode

| モード | 説明 |
|--------|------|
| `acceptEdits` (default) | ファイル編集・ファイルシステム操作を自動承認 |
| `bypassPermissions` | 全ツール自動承認 |

デフォルトは`acceptEdits`。任意のBashコマンド実行が必要な場合は`--bypass-permissions`を使用。

## 許可ツール

`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `NotebookEdit`, `WebFetch`, `WebSearch`

`--no-web`で`WebFetch`, `WebSearch`を除外可能。
