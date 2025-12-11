# Task Prompt Orchestrator

Claude Agent SDKを使用したマルチステップタスク自動化ツール。YAMLで定義されたタスクを順番に実行し、各タスクの検証結果に応じてフローを制御する。

## 2つの実行モード

| モード | 用途 |
|--------|------|
| Loop C | タスク定義ファイルを順次実行 |
| Loop B | 要件定義から自動でタスク生成・実行・検証を繰り返す |

**ファイル内容から自動判定**されるため、コマンドは共通:

```bash
task-orchestrator run <file.yaml>
```

## クイックスタート

### Loop C: タスク定義ファイルを実行

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

```bash
task-orchestrator run tasks.yaml
```

### Loop B: 要件定義から自動実行

要件定義ファイルを作成：

```yaml
# requirements.yaml
requirements:
  - id: req_1
    name: ユーザー認証機能を実装する
    acceptance_criteria:
      - ログイン/ログアウトが動作する
      - パスワードがハッシュ化されて保存される

  - id: req_2
    name: テストカバレッジ80%以上
    acceptance_criteria:
      - pytest --cov でカバレッジ80%以上
```

```bash
task-orchestrator run requirements.yaml
```

Loop Bは以下のサイクルを自動で繰り返す：
1. 要件からタスク定義を生成
2. タスクを実行（Loop C）
3. 要件の達成状況を検証
4. 未達成の要件があれば追加タスクを生成して再実行

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

### 基本コマンド

```bash
# タスク/要件ファイルを実行（ファイル内容で自動判定）
task-orchestrator run <file.yaml>

# dry-run（実行フローとプロンプトを確認）
task-orchestrator run <file.yaml> --dry-run

# サンプルYAML生成
task-orchestrator sample > /tmp/tasks.yaml
```

### オプション

| オプション | 説明 |
|-----------|------|
| `--loopb` | 強制的にLoop Bモードで実行（通常は自動判定） |
| `--max-retries N` | タスクあたりの最大リトライ回数 (default: 3) |
| `--max-total-retries N` | 全体の最大リトライ回数 (default: 10) |
| `--cwd PATH` | 作業ディレクトリ |
| `--model MODEL` | 使用モデル |
| `--no-web` | WebFetch/WebSearch無効化 |
| `--bypass-permissions` | 全ツール自動承認 (要注意) |
| `-o FILE` | 結果JSONの出力先 |
| `--dry-run` | 実行フローとプロンプトを表示（実行しない） |

## フロー制御

```
instruction → validation
    ↓
approved → 次のタスクへ
declined → feedbackを付けてリトライ (max_retries まで)
permission_denied → 即終了 (リトライなし)
```

## 履歴管理・途中再開

実行履歴は `.task-orchestrator-history/` に自動保存される。

### 自動resume

**同じファイルを再度 `run` すると、未完了の実行があれば自動的に途中から再開する：**

```bash
# 初回実行（途中で中断）
task-orchestrator run tasks.yaml

# 再度実行 → 自動的に途中から再開
task-orchestrator run tasks.yaml
```

### 手動でのresume

```bash
# 未完了の履歴一覧
task-orchestrator history

# 履歴IDを指定して再開
task-orchestrator resume <history_id>

# 特定ポイントから再開（Loop Cのみ）
task-orchestrator resume <history_id> --from task2_validation

# Loop Bの再開
task-orchestrator resume <history_id> --loopb
```

再開ポイント（Loop C）は `<task_id>_instruction` または `<task_id>_validation` の形式。

### 履歴操作

```bash
# 全履歴一覧（完了済み含む）
task-orchestrator history --all
task-orchestrator history --loopb --all

# 詳細表示
task-orchestrator history --show <history_id>
task-orchestrator history --loopb --show <loopb_history_id>

# 紐づくLoop C履歴を表示
task-orchestrator history --loopb-children <loopb_history_id>
```

### 履歴ファイルの削除

履歴は以下の構造で保存される:
```
.task-orchestrator-history/
├── <history_id>.json          # Loop C履歴
└── loopb/
    ├── <history_id>.json      # Loop B履歴
    └── tasks/                 # 生成されたタスクYAML
```

```bash
# 特定の履歴を削除
rm .task-orchestrator-history/<history_id>.json
rm .task-orchestrator-history/loopb/<history_id>.json

# 全履歴クリア（新規実行したい場合）
rm -rf .task-orchestrator-history
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
