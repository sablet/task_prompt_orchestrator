# Loop B (要件充足ループ) 追加 実装計画

## 概要

現在のオーケストレーター（ループC）を外側から制御するループBを追加する。

### アーキテクチャ原則

```
┌─────────────────────────────────────────────────────────┐
│ Loop B (requirements_orchestrator)                       │
│   - requirements.yaml を入力                             │
│   - tasks.yaml を生成・管理                              │
│   - Loop C を呼び出し                                    │
│   - 要件充足を確認                                       │
│                                                          │
│   ┌───────────────────────────────────────────────────┐ │
│   │ Loop C (orchestrator) ← 従来通り                   │ │
│   │   - tasks.yaml を入力（変更なし）                  │ │
│   │   - タスク実行＋検証                               │ │
│   │   - .task-orchestrator-history/ に履歴保存        │ │
│   └───────────────────────────────────────────────────┘ │
│                                                          │
│   Loop Bは Loop C の履歴を参照・追跡可能                │
└─────────────────────────────────────────────────────────┘
```

**重要**: Loop Cは従来通りYAMLファイルを入力として独立動作する。Loop Bはそれを外側から制御し、必要に応じてYAMLファイルを生成・更新する。

### フロー詳細
```
Loop B:
  1. requirements.yaml読み込み
  2. LLMにtasks.yamlを生成させる → ファイル出力
  3. Loop C実行（task-orchestrator run tasks.yaml）
  4. Loop Cの履歴を確認
  5. LLMに要件充足確認
  6a. 充足 or 3回試行 → 終了
  6b. 未充足 → 追加tasks.yamlを生成 → 3へ
```

## ファイル構成

```
src/task_prompt_orchestrator/
├── schema.py                    # 既存 + Requirement*, LoopB* 追加
├── orchestrator.py              # 既存のまま (ループC)
├── requirements_orchestrator.py # 新規: ループB管理
├── prompts.py                   # 新規: プロンプト構築
└── cli.py                       # 既存 + run-requirements追加
```

## Phase 1: スキーマ拡張 (schema.py)

### 追加するクラス

```python
@dataclass
class Requirement:
    id: str
    description: str
    acceptance_criteria: list[str]
    priority: str = "required"

@dataclass
class RequirementDefinition:
    name: str
    context: str
    requirements: list[Requirement]
    constraints: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RequirementDefinition": ...

class LoopBStatus(Enum):
    GENERATING_TASKS = "generating_tasks"
    EXECUTING_TASKS = "executing_tasks"
    VERIFYING_REQUIREMENTS = "verifying_requirements"
    GENERATING_ADDITIONAL_TASKS = "generating_additional_tasks"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class LoopBIteration:
    iteration_number: int
    generated_tasks: list[Task]
    loop_c_result: OrchestratorResult | None
    verification_result: dict[str, Any] | None
    unmet_requirements: list[str]

@dataclass
class LoopBExecutionHistory:
    history_id: str
    requirements_path: str
    started_at: str
    updated_at: str
    status: LoopBStatus
    max_iterations: int
    current_iteration: int
    iterations: list[LoopBIteration]
    completed_task_ids: list[str]
    final_result: dict[str, Any] | None
    error: str | None = None
```

### requirements.yaml フォーマット

```yaml
name: "機能名"
context: |
  プロジェクトのコンテキスト情報
requirements:
  - id: req_1
    description: "要件の説明"
    acceptance_criteria:
      - "受け入れ基準1"
      - "受け入れ基準2"
    priority: required
constraints:
  - "制約条件"
```

## Phase 2: プロンプト実装 (prompts.py 新規作成)

### 関数一覧

1. `build_task_generation_prompt(requirements, completed_tasks?, previous_feedback?)` - 要件→タスク生成
2. `build_requirement_verification_prompt(requirements, completed_tasks)` - 要件充足確認
3. `parse_generated_tasks(output) -> list[Task]` - LLM出力からタスク抽出
4. `parse_verification_result(output) -> dict` - LLM出力から検証結果抽出

### タスク生成プロンプト（初回）
- プロジェクトコンテキスト
- 要件一覧（ID、説明、受け入れ基準）
- 制約条件
- YAML形式での出力指示

### タスク生成プロンプト（追加）
- 上記 + 完了済みタスク一覧 + 前回フィードバック
- 「既存タスクは維持、新しいタスクのみ出力」の指示

### 要件充足確認プロンプト
- 要件一覧
- 完了タスクとその結果
- JSON形式での出力指示（all_requirements_met, requirement_status配列）

## Phase 3: ループBオーケストレーター (requirements_orchestrator.py 新規作成)

### クラス構成

```python
@dataclass
class RequirementsOrchestratorConfig:
    max_iterations: int = 3
    orchestrator_config: OrchestratorConfig | None = None

class LoopBHistoryManager:
    """ループB履歴管理（.task-orchestrator-history/loopb/）"""
    ...

class RequirementsOrchestrator:
    async def run(self) -> LoopBExecutionHistory:
        for iteration in range(max_iterations):
            tasks = await self._generate_tasks(iteration)
            loop_c_result = await self._execute_loop_c(tasks)
            verification = await self._verify_requirements()

            if verification["all_requirements_met"]:
                return SUCCESS
        return FAILED

    async def _generate_tasks(self, iteration: int) -> list[Task]: ...
    async def _execute_loop_c(self, tasks: list[Task]) -> OrchestratorResult: ...
    async def _verify_requirements(self) -> dict: ...
```

## Phase 4: CLI統合 (cli.py)

### 新コマンド

```bash
task-orchestrator run-requirements requirements.yaml [options]
```

### オプション
- `--max-iterations`: ループB最大試行回数（デフォルト3）
- 既存runコマンドと同様のオプション（--cwd, --model, --max-retries等）

## Phase 5: テスト・サンプル

1. `examples/sample_requirements.yaml` 作成
2. `create_sample_requirements_yaml()` 関数追加
3. 各コンポーネントのユニットテスト

## 実装順序

| Step | 内容 | ファイル |
|------|------|---------|
| 1 | Requirement, RequirementDefinition追加 | schema.py |
| 2 | LoopBStatus, LoopBIteration, LoopBExecutionHistory追加 | schema.py |
| 3 | プロンプト構築関数実装 | prompts.py (新規) |
| 4 | パーサー関数実装 | prompts.py |
| 5 | LoopBHistoryManager実装 | requirements_orchestrator.py (新規) |
| 6 | RequirementsOrchestrator実装 | requirements_orchestrator.py |
| 7 | run-requirementsコマンド追加 | cli.py |
| 8 | サンプルファイル作成 | examples/, schema.py |
| 9 | __init__.py更新 | __init__.py |

## 変更ファイル一覧

- `src/task_prompt_orchestrator/schema.py` - スキーマ拡張
- `src/task_prompt_orchestrator/prompts.py` - 新規作成
- `src/task_prompt_orchestrator/requirements_orchestrator.py` - 新規作成
- `src/task_prompt_orchestrator/cli.py` - コマンド追加
- `src/task_prompt_orchestrator/__init__.py` - エクスポート追加
- `examples/sample_requirements.yaml` - 新規作成

## 履歴管理の設計

### ディレクトリ構造

```
.task-orchestrator-history/
├── loopb/
│   └── {requirements_name}_{timestamp}.json   # Loop B履歴
└── {tasks_name}_{timestamp}.json              # Loop C履歴（従来通り）
```

### Loop B履歴 → Loop C履歴の参照

```python
@dataclass
class LoopBIteration:
    iteration_number: int
    tasks_yaml_path: str              # 生成したtasks.yamlのパス
    loop_c_history_id: str            # 対応するLoop C履歴ID
    verification_result: dict | None
    unmet_requirements: list[str]

@dataclass
class LoopBExecutionHistory:
    history_id: str
    requirements_path: str
    tasks_output_dir: str             # tasks.yaml出力先ディレクトリ
    iterations: list[LoopBIteration]
    loop_c_history_ids: list[str]     # 全Loop C履歴IDのリスト
    ...
```

### 履歴参照の流れ

```
Loop B履歴                          Loop C履歴
──────────                          ──────────
loopb/req_20241211_120000.json
  └─ iterations[0]
       └─ loop_c_history_id ────────► tasks_iter1_20241211_120001.json
  └─ iterations[1]
       └─ loop_c_history_id ────────► tasks_iter2_20241211_120500.json
```

### CLIでの履歴確認

```bash
# Loop B履歴一覧
task-orchestrator history --loopb

# Loop B履歴詳細（紐づくLoop C履歴も表示）
task-orchestrator history --loopb --show <loopb_history_id>

# 特定Loop B配下のLoop C履歴のみ表示
task-orchestrator history --loopb-children <loopb_history_id>
```

## タスクYAML生成の管理

### 生成ファイルの配置

```
doc/design/
├── {feature}-requirements.yaml      # 入力: 要件定義
└── generated/
    ├── {feature}-tasks-iter1.yaml   # 生成: 1回目のタスク
    ├── {feature}-tasks-iter2.yaml   # 生成: 追加タスク
    └── ...
```

### 追加タスク生成時の動作

1. 前回のtasks.yamlを読み込み
2. 完了済みタスクIDを把握（Loop C履歴から）
3. 追加タスクのみを含む新しいYAMLを生成
4. 新YAMLでLoop Cを実行

```yaml
# {feature}-tasks-iter2.yaml（追加タスク）
# 前回完了: task_1, task_2, task_3
# 今回追加分のみ

tasks:
  - id: task_4
    name: "追加タスク1"
    depends_on: [task_2]  # 完了済みタスクへの依存OK
    instruction: |
      ...
```

## CLI設計

### 新コマンド

```bash
# Loop B実行（requirements.yamlから開始）
task-orchestrator run-requirements requirements.yaml [options]

Options:
  --max-iterations N     # Loop B最大試行回数（デフォルト3）
  --tasks-output-dir DIR # 生成tasks.yamlの出力先
  --cwd DIR              # 作業ディレクトリ
  --model MODEL          # 使用モデル
  --max-retries N        # Loop C内タスクリトライ回数
```

### 既存コマンドは変更なし

```bash
# 従来通りのLoop C単体実行
task-orchestrator run tasks.yaml [options]

# 従来通りの履歴操作
task-orchestrator history [options]
task-orchestrator resume <history_id> [options]
```
