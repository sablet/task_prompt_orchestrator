"""Tests for prompts.py - prompt building and parsing functions."""

import pytest

from task_prompt_orchestrator.prompts import (
    build_requirement_verification_prompt,
    build_task_generation_prompt,
    get_unmet_requirement_ids,
    parse_generated_tasks,
    parse_verification_result,
)
from task_prompt_orchestrator.schema import (
    AcceptanceCriterion,
    Requirement,
    RequirementDefinition,
    Task,
    TaskResult,
    TaskStatus,
)


class TestParseGeneratedTasks:
    """Tests for parse_generated_tasks function."""

    def test_parse_valid_yaml(self) -> None:
        """Test parsing YAML with code block and raw YAML."""
        # With code block
        output_block = """```yaml
tasks:
  - id: task_1
    name: "Create user model"
    instruction: |
      Create a User model.
    validation:
      - "User model exists"
    depends_on: []
  - id: task_2
    name: "Add auth"
    instruction: Implement auth.
    validation:
      - "Auth works"
    depends_on: [task_1]
```"""
        tasks = parse_generated_tasks(output_block)
        assert len(tasks) == 2
        assert tasks[0].id == "task_1"
        assert tasks[1].depends_on == ["task_1"]

        # Raw YAML
        tasks_raw = parse_generated_tasks(
            "tasks:\n  - id: raw\n    name: Raw\n    instruction: Do\n    validation: [Done]"
        )
        assert len(tasks_raw) == 1
        assert tasks_raw[0].id == "raw"

    def test_parse_invalid_output(self) -> None:
        """Test parsing empty, invalid, or missing tasks key."""
        assert parse_generated_tasks("") == []
        assert parse_generated_tasks("no yaml here") == []
        assert parse_generated_tasks("```yaml\ninvalid: [") == []
        assert parse_generated_tasks("```yaml\nother_key: [1]\n```") == []


class TestParseVerificationResult:
    """Tests for parse_verification_result function."""

    def test_parse_json_responses(self) -> None:
        """Test parsing JSON with code block and fallback heuristic."""
        # Valid JSON
        output = """```json
{
  "all_requirements_met": false,
  "requirement_status": [{"requirement_id": "req_1", "met": false}],
  "summary": "Not done",
  "feedback_for_additional_tasks": "Need work"
}
```"""
        result = parse_verification_result(output)
        assert result["all_requirements_met"] is False
        assert result["feedback_for_additional_tasks"] == "Need work"

        # Heuristic fallback
        assert (
            parse_verification_result("All requirements met")["all_requirements_met"]
            is True
        )
        assert (
            parse_verification_result("Missing requirements")["all_requirements_met"]
            is False
        )


class TestGetUnmetRequirementIds:
    """Tests for get_unmet_requirement_ids function."""

    def test_extract_unmet_ids(self) -> None:
        """Test extracting unmet requirement IDs."""
        # All met
        assert (
            get_unmet_requirement_ids(
                {
                    "requirement_status": [
                        {"requirement_id": "req_1", "met": True},
                        {"requirement_id": "req_2", "met": True},
                    ]
                }
            )
            == []
        )

        # Some unmet
        unmet = get_unmet_requirement_ids(
            {
                "requirement_status": [
                    {"requirement_id": "req_1", "met": True},
                    {"requirement_id": "req_2", "met": False},
                ]
            }
        )
        assert unmet == ["req_2"]

        # Empty
        assert get_unmet_requirement_ids({"requirement_status": []}) == []


class TestBuildPrompts:
    """Tests for prompt building functions."""

    def test_task_generation_prompts(self) -> None:
        """Test initial and additional task generation prompts."""
        requirements = RequirementDefinition(
            requirements=[
                Requirement(
                    id="req_1",
                    name="Test Requirement",
                    acceptance_criteria=[
                        AcceptanceCriterion(criterion="Done", verify="Check it works")
                    ],
                )
            ],
        )

        # Initial generation
        prompt = build_task_generation_prompt(requirements)
        assert "req_1" in prompt
        assert "Test Requirement" in prompt

        # Additional generation
        completed = [
            Task(id="task_1", name="Done", instruction="Did", validation=["OK"])
        ]
        prompt_add = build_task_generation_prompt(
            requirements, completed_tasks=completed, previous_feedback="Need more"
        )
        assert "task_1" in prompt_add
        assert "Need more" in prompt_add

    def test_verification_prompt(self) -> None:
        """Test verification prompt generation."""
        requirements = RequirementDefinition(
            requirements=[
                Requirement(
                    id="req_1",
                    name="Test Req",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            criterion="Works", verify="Execute and check"
                        )
                    ],
                )
            ],
        )
        results = [TaskResult(task_id="task_1", status=TaskStatus.APPROVED)]
        prompt = build_requirement_verification_prompt(requirements, results)
        assert "req_1" in prompt
        assert "task_1" in prompt
