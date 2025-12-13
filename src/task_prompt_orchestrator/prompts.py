"""Prompt building and parsing functions for Loop B."""

import json
import re
from typing import Any

import yaml

from .schema import (
    AcceptanceCriterion,
    Requirement,
    RequirementDefinition,
    Task,
    TaskResult,
    ValidationItem,
)
from .templates import render_template


def _format_criterion_text(ac: AcceptanceCriterion) -> str:
    """Format a single acceptance criterion with optional verify."""
    if ac.verify:
        return f"{ac.criterion}\n      - **Verify**: {ac.verify}"
    return ac.criterion


def _format_requirements_with_ids(requirements: list[Requirement]) -> str:
    """Format requirements list with criterion IDs for coverage mapping."""
    lines = []
    for req in requirements:
        lines.append(f"### {req.id}: {req.name}")
        if req.notes:
            lines.append(f"**Notes**: {req.notes}")
        lines.append("**Acceptance Criteria**:")
        for i, ac in enumerate(req.acceptance_criteria, 1):
            criterion_id = f"{req.id}.{i}"
            lines.append(f"  - `{criterion_id}`: {_format_criterion_text(ac)}")
        if req.design_decisions:
            lines.append("**Design Decisions**:")
            for i, decision in enumerate(req.design_decisions, 1):
                decision_id = f"{req.id}.d{i}"
                lines.append(f"  - `{decision_id}`: {decision}")
        lines.append("")
    return "\n".join(lines)


def _format_requirements(requirements: list[Requirement]) -> str:
    """Format requirements list for prompt (without IDs, for verification)."""
    lines = []
    for req in requirements:
        lines.append(f"### {req.id}: {req.name}")
        if req.notes:
            lines.append(f"**Notes**: {req.notes}")
        lines.append("**Acceptance Criteria**:")
        for ac in req.acceptance_criteria:
            lines.append(f"  - {_format_criterion_text(ac)}")
        if req.design_decisions:
            lines.append("**Design Decisions**:")
            for decision in req.design_decisions:
                lines.append(f"  - {decision}")
        lines.append("")
    return "\n".join(lines)


def _format_single_requirement(req: Requirement) -> str:
    """Format a single requirement for verification prompt."""
    lines = []
    lines.append(f"## {req.id}: {req.name}")
    if req.notes:
        lines.append(f"**Notes**: {req.notes}")
    lines.append("")
    lines.append("### Acceptance Criteria")
    for i, ac in enumerate(req.acceptance_criteria, 1):
        lines.append(f"**{i}. {ac.criterion}**")
        if ac.verify:
            lines.append(f"   - Verify: {ac.verify}")
        lines.append("")
    if req.design_decisions:
        lines.append("### Design Decisions")
        for i, decision in enumerate(req.design_decisions, 1):
            lines.append(f"{i}. {decision}")
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


def _format_common_validation(common_validation: list[str]) -> str:
    """Format common validation criteria for prompt."""
    if not common_validation:
        return ""
    return "\n".join(f"- {criterion}" for criterion in common_validation)


def build_task_generation_prompt(
    unmet_requirements: list[Requirement],
    all_requirements_path: str | None,
    common_validation: list[str],
    completed_tasks: list[Task] | None = None,
    previous_feedback: str | None = None,
    exploration_path: str | None = None,
) -> str:
    """Build prompt for generating tasks from unmet requirements.

    Args:
        unmet_requirements: Requirements that are not yet met (task generation target)
        all_requirements_path: Path to requirements YAML file (for context reference)
        common_validation: Common validation criteria for all tasks
        completed_tasks: Previously completed tasks (for additional task generation)
        previous_feedback: Feedback from previous verification
        exploration_path: Path to pre-collected exploration context file (optional)
    """
    req_text = _format_requirements_with_ids(unmet_requirements)
    common_val_text = _format_common_validation(common_validation)

    if completed_tasks:
        return render_template(
            "task_generation_additional.j2",
            requirements_text=req_text,
            all_requirements_path=all_requirements_path,
            common_validation_text=common_val_text,
            completed_tasks_text=_format_completed_tasks(completed_tasks),
            previous_feedback=previous_feedback or "(none)",
            exploration_path=exploration_path,
        )

    return render_template(
        "task_generation_initial.j2",
        requirements_text=req_text,
        all_requirements_path=all_requirements_path,
        common_validation_text=common_val_text,
        exploration_path=exploration_path,
    )


def build_requirement_verification_prompt(
    requirements: RequirementDefinition,
    completed_task_results: list[TaskResult],
) -> str:
    """Build prompt for verifying if requirements are met (legacy, all at once)."""
    return render_template(
        "requirement_verification.j2",
        requirements_text=_format_requirements(requirements.requirements),
        task_results_text=_format_task_results(completed_task_results),
    )


def build_shared_exploration_prompt(
    all_requirements_path: str,
) -> str:
    """Build prompt for shared codebase exploration before verification.

    Args:
        all_requirements_path: Path to requirements YAML file
    """
    return render_template(
        "shared_exploration.j2",
        all_requirements_path=all_requirements_path,
    )


def build_single_requirement_verification_prompt(
    req: Requirement,
    all_requirements_path: str | None = None,
    exploration_path: str | None = None,
) -> str:
    """Build prompt for verifying a single requirement.

    Args:
        req: The requirement to verify
        all_requirements_path: Path to requirements YAML file (for context reference)
        exploration_path: Path to pre-collected exploration context file (optional)
    """
    return render_template(
        "single_requirement_verification.j2",
        requirement_text=_format_single_requirement(req),
        all_requirements_path=all_requirements_path,
        requirement_id=req.id,
        criteria_count=len(req.acceptance_criteria),
        design_decisions_count=len(req.design_decisions),
        exploration_path=exploration_path,
    )


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
    """Get all acceptance criterion and design decision IDs from requirements."""
    ids = set()
    for req in requirements.requirements:
        for i in range(1, len(req.acceptance_criteria) + 1):
            ids.add(f"{req.id}.{i}")
        for i in range(1, len(req.design_decisions) + 1):
            ids.add(f"{req.id}.d{i}")
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
    """Check if all acceptance criteria and design decisions are covered by task validations.

    Returns:
        Tuple of (all_covered, list_of_uncovered_ids)
    """
    all_ids = get_all_criterion_ids(requirements)
    covered_ids = get_covered_criterion_ids(tasks)
    uncovered = all_ids - covered_ids
    return len(uncovered) == 0, sorted(uncovered)
