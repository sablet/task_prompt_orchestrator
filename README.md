# Task Prompt Orchestrator

Claude Agent SDKを使用したマルチステップタスク自動化ツール。YAMLで定義されたタスクを順番に実行し、各タスクの検証結果に応じてフローを制御する。

## 2つの実行モード

| モード | コマンド | 用途 |
|--------|----------|------|
| Loop C | `task-orchestrator run tasks.yaml` | タスク定義ファイルを順次実行 |
| Loop B | `task-orchestrator run-requirements requirements.yaml` | 要件定義から自動でタスク生成・実行・検証を繰り返す |

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
    description: ユーザー認証機能を実装する
    acceptance_criteria:
      - ログイン/ログアウトが動作する
      - パスワードがハッシュ化されて保存される

  - id: req_2
    description: テストカバレッジ80%以上
    acceptance_criteria:
      - pytest --cov でカバレッジ80%以上
```

```bash
task-orchestrator run-requirements requirements.yaml
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

### Loop C (タスク実行)

```bash
# タスク実行
task-orchestrator run tasks.yaml

# dry-run (プロンプト確認のみ)
task-orchestrator run tasks.yaml --dry-run

# サンプルYAML生成
task-orchestrator sample > /tmp/tasks.yaml
```

### Loop B (要件駆動実行)

```bash
# 要件定義から実行
task-orchestrator run-requirements requirements.yaml

# イテレーション上限を指定
task-orchestrator run-requirements requirements.yaml --max-iterations 5
```

### 共通オプション

| オプション | 説明 | 対象 |
|-----------|------|------|
| `--max-retries N` | タスクあたりの最大リトライ回数 (default: 3) | run, run-requirements |
| `--max-total-retries N` | 全体の最大リトライ回数 (default: 10) | run, run-requirements |
| `--cwd PATH` | 作業ディレクトリ | 全コマンド |
| `--model MODEL` | 使用モデル | run, run-requirements |
| `--no-web` | WebFetch/WebSearch無効化 | run, run-requirements |
| `--bypass-permissions` | 全ツール自動承認 (要注意) | run, run-requirements |
| `--dry-run` | 実行せずプロンプト表示 | run |
| `-o FILE` | 結果JSONの出力先 | run, run-requirements |

### Loop B専用オプション

| オプション | 説明 |
|-----------|------|
| `--max-iterations N` | 最大イテレーション回数 (default: 5) |
| `--tasks-output-dir DIR` | 生成タスクYAMLの保存先 |

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

### Loop C履歴

```bash
# 未完了の履歴一覧
task-orchestrator history

# 途中から再開
task-orchestrator resume <history_id>

# 特定ポイントから再開
task-orchestrator resume <history_id> --from task2_validation
```

再開ポイントは `<task_id>_instruction` または `<task_id>_validation` の形式。

### Loop B履歴

```bash
# Loop B履歴一覧
task-orchestrator history --loopb

# 詳細表示
task-orchestrator history --loopb --show <loopb_history_id>

# 紐づくLoop C履歴を表示
task-orchestrator history --loopb-children <loopb_history_id>
```

### その他の履歴操作

```bash
# 全履歴一覧
task-orchestrator history --all
task-orchestrator history --loopb --all

# 詳細表示
task-orchestrator history --show <history_id>

# 履歴削除
task-orchestrator history --delete <history_id>
task-orchestrator history --loopb --delete <loopb_history_id>

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
