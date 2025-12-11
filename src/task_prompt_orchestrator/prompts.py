"""Prompt building and parsing functions for Loop B."""

import json
import re
from typing import Any

import yaml

from .schema import Requirement, RequirementDefinition, Task, TaskResult, ValidationItem


def _format_requirements_with_ids(requirements: list[Requirement]) -> str:
    """Format requirements list with criterion IDs for coverage mapping."""
    lines = []
    for req in requirements:
        lines.append(f"### {req.id}: {req.name}")
        lines.append("**Acceptance Criteria**:")
        for i, criterion in enumerate(req.acceptance_criteria, 1):
            criterion_id = f"{req.id}.{i}"
            lines.append(f"  - `{criterion_id}`: {criterion}")
        lines.append("")
    return "\n".join(lines)


def _format_requirements(requirements: list[Requirement]) -> str:
    """Format requirements list for prompt (without IDs, for verification)."""
    lines = []
    for req in requirements:
        lines.append(f"### {req.id}: {req.name}")
        lines.append("**Acceptance Criteria**:")
        for criterion in req.acceptance_criteria:
            lines.append(f"  - {criterion}")
        lines.append("")
    return "\n".join(lines)


def _format_completed_tasks(tasks: list[Task]) -> str:
    """Format completed tasks for prompt."""
    if not tasks:
        return "(none)"
    lines = []
    for task in tasks:
        lines.append(f"- **{task.id}**: {task.name}")
    return "\n".join(lines)


def _format_task_results(results: list[TaskResult]) -> str:
    """Format task results for verification prompt."""
    if not results:
        return "(none)"
    lines = []
    for result in results:
        lines.append(f"### {result.task_id}")
        lines.append(f"**Status**: {result.status.value}")
        if result.instruction_output:
            output_preview = result.instruction_output[:500]
            if len(result.instruction_output) > 500:
                output_preview += "..."
            lines.append(f"**Output**:\n```\n{output_preview}\n```")
        lines.append("")
    return "\n".join(lines)


TASK_GENERATION_BASE = """## タスク設計の原則

1. **単一責任**: 1タスク = 1つの明確な成果物
2. **検証可能**: 各タスクに機械的に確認可能な validation を定義
3. **依存関係の明示**: depends_on で順序を制御
4. **自己完結**: 各タスクの instruction は単独で実行可能な情報を含む
5. **セットアップと作業の分離**: ツールのインストール等のセットアップは、それを使う作業と同一タスクにしない
6. **カバレッジ必須**: 全ての acceptance_criteria が少なくとも1つの validation でカバーされること

## 出力フォーマット

各 validation には `covers` で対応する acceptance_criteria ID を明示してください。

```yaml
tasks:
  - id: task_1
    name: {タスク名}
    depends_on: []
    instruction: |
      {具体的な作業指示}

      ## 変更対象
      - `src/{path/to/file}`: {変更内容}

      ## 関連パス
      - `src/{path/to/related}`: {役割・関係性}
    validation:
      - criterion: "{検証項目1}"
        covers: [req_id.1, req_id.2]
      - criterion: "{検証項目2}"
        covers: [req_id.3]
```
"""


def build_task_generation_prompt(
    requirements: RequirementDefinition,
    completed_tasks: list[Task] | None = None,
    previous_feedback: str | None = None,
) -> str:
    """Build prompt for generating tasks from requirements."""
    req_text = _format_requirements_with_ids(requirements.requirements)

    if completed_tasks:
        return f"""# 追加タスク生成

## 要件
{req_text}

## 完了済みタスク
{_format_completed_tasks(completed_tasks)}

## 前回のフィードバック
{previous_feedback or "(none)"}

## 指示
未達成の要件を満たすための追加タスクを生成してください。
完了済みタスクは変更せず、新しいタスクは既存のtask IDに依存可能です。

{TASK_GENERATION_BASE}"""

    return f"""# タスク生成

## 要件
{req_text}

## 指示
上記の要件をすべて満たすタスクを生成してください。

{TASK_GENERATION_BASE}"""


def build_requirement_verification_prompt(
    requirements: RequirementDefinition,
    completed_task_results: list[TaskResult],
) -> str:
    """Build prompt for verifying if requirements are met."""
    return f"""## Requirement Verification

### Requirements
{_format_requirements(requirements.requirements)}

### Completed Tasks
{_format_task_results(completed_task_results)}

### Instructions
Verify if each requirement's acceptance criteria are met based on completed tasks.

Output in JSON format:
```json
{{
  "all_requirements_met": true/false,
  "requirement_status": [
    {{
      "requirement_id": "req_1",
      "met": true/false,
      "evidence": "Evidence or explanation"
    }}
  ],
  "summary": "Overall summary",
  "feedback_for_additional_tasks": "Feedback for next iteration if requirements not met"
}}
```
"""


def _parse_validation_items(
    validation_data: list[Any],
) -> tuple[list[str], list[ValidationItem]]:
    """Parse validation data into both string list and ValidationItem list.

    Handles both formats:
    - Simple strings: ["criterion1", "criterion2"]
    - Objects with covers: [{"criterion": "...", "covers": ["req.1"]}]
    """
    validation_strings: list[str] = []
    validation_items: list[ValidationItem] = []

    for v in validation_data:
        if isinstance(v, str):
            validation_strings.append(v)
            validation_items.append(ValidationItem(criterion=v, covers=[]))
        elif isinstance(v, dict):
            criterion = v.get("criterion", "")
            covers = v.get("covers", [])
            validation_strings.append(criterion)
            validation_items.append(ValidationItem(criterion=criterion, covers=covers))

    return validation_strings, validation_items


def parse_generated_tasks(output: str) -> list[Task]:
    """Parse LLM output to extract generated tasks."""
    # Find ```yaml ... ``` block, handling nested code blocks in instruction
    # Strategy: find ```yaml, then find the last ``` that closes it
    start_match = re.search(r"```yaml\s*\n", output)
    if start_match:
        content_start = start_match.end()
        # Find the closing ``` by looking for ``` at the start of a line (after newline)
        # This handles nested code blocks that are indented
        remaining = output[content_start:]
        # Match ``` that is either at start or preceded by newline (not indented)
        end_match = re.search(r"\n```\s*$", remaining, re.MULTILINE)
        yaml_content = remaining[: end_match.start()] if end_match else remaining
    else:
        yaml_content = output

    try:
        data = yaml.safe_load(yaml_content)
        if not data or "tasks" not in data:
            return []

        tasks = []
        for t in data["tasks"]:
            validation_data = t.get("validation", [])
            validation_strings, validation_items = _parse_validation_items(
                validation_data
            )
            tasks.append(
                Task(
                    id=t["id"],
                    name=t["name"],
                    instruction=t["instruction"],
                    validation=validation_strings,
                    depends_on=t.get("depends_on", []),
                    validation_items=validation_items,
                )
            )
        return tasks
    except (yaml.YAMLError, KeyError, TypeError):
        return []


def parse_verification_result(output: str) -> dict[str, Any]:
    """Parse LLM output to extract verification result."""
    json_match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
    if json_match:
        json_content = json_match.group(1)
    else:
        json_match = re.search(
            r'(\{[^{}]*"all_requirements_met"[^{}]*\})',
            output,
            re.DOTALL,
        )
        json_content = json_match.group(1) if json_match else output

    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        output_lower = output.lower()
        all_met = (
            "all_requirements_met" in output_lower and "true" in output_lower
        ) or ("all requirements" in output_lower and "met" in output_lower)

        return {
            "all_requirements_met": all_met,
            "requirement_status": [],
            "summary": "Could not parse verification result",
            "feedback_for_additional_tasks": output[:500] if not all_met else "",
        }


def get_unmet_requirement_ids(verification_result: dict[str, Any]) -> list[str]:
    """Extract unmet requirement IDs from verification result."""
    return [
        status.get("requirement_id", "unknown")
        for status in verification_result.get("requirement_status", [])
        if not status.get("met", False)
    ]


def get_all_criterion_ids(requirements: RequirementDefinition) -> set[str]:
    """Get all acceptance criterion IDs from requirements."""
    ids = set()
    for req in requirements.requirements:
        for i in range(1, len(req.acceptance_criteria) + 1):
            ids.add(f"{req.id}.{i}")
    return ids


def get_covered_criterion_ids(tasks: list[Task]) -> set[str]:
    """Get all criterion IDs covered by task validations."""
    covered = set()
    for task in tasks:
        for item in task.validation_items:
            covered.update(item.covers)
    return covered


def check_coverage(
    requirements: RequirementDefinition, tasks: list[Task]
) -> tuple[bool, list[str]]:
    """Check if all acceptance criteria are covered by task validations.

    Returns:
        Tuple of (all_covered, list_of_uncovered_ids)
    """
    all_ids = get_all_criterion_ids(requirements)
    covered_ids = get_covered_criterion_ids(tasks)
    uncovered = all_ids - covered_ids
    return len(uncovered) == 0, sorted(uncovered)
