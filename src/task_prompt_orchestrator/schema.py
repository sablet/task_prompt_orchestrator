"""YAML task definition schema and dataclasses."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    pass


def _get_examples_dir() -> Path:
    """Get the examples directory path."""
    # Navigate from schema.py -> src/task_prompt_orchestrator -> src -> project root -> examples
    return Path(__file__).parent.parent.parent / "examples"


# =============================================================================
# Loop C (Task Execution) Schema
# =============================================================================


class ExecutionPhase(Enum):
    """Phase within a task execution."""

    INSTRUCTION = "instruction"
    VALIDATION = "validation"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    DECLINED = "declined"
    FAILED = "failed"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class ValidationItem:
    """Validation item with coverage mapping to acceptance criteria."""

    criterion: str
    covers: list[str] = field(default_factory=list)  # List of acceptance_criteria IDs


@dataclass
class Task:
    """Single task definition."""

    id: str
    name: str
    instruction: str
    validation: list[str]
    depends_on: list[str] = field(default_factory=list)
    # Extended validation with coverage info (used by Loop B, not persisted to YAML)
    validation_items: list[ValidationItem] = field(default_factory=list)


@dataclass
class TaskDefinition:
    """Complete task definition from YAML."""

    tasks: list[Task]
    common_validation: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TaskDefinition":
        """Load task definition from YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        tasks = []
        for task_data in data.get("tasks", []):
            tasks.append(
                Task(
                    id=task_data["id"],
                    name=task_data["name"],
                    instruction=task_data["instruction"],
                    validation=task_data.get("validation", []),
                    depends_on=task_data.get("depends_on", []),
                )
            )
        common_validation = data.get("common_validation", [])

        return cls(tasks=tasks, common_validation=common_validation)


@dataclass
class TaskResult:
    """Result of a single task execution."""

    task_id: str
    status: TaskStatus
    instruction_output: str | None = None
    validation_output: str | None = None
    validation_approved: bool = False
    attempts: int = 0
    error: str | None = None


@dataclass
class OrchestratorResult:
    """Complete orchestrator execution result."""

    task_results: list[TaskResult]
    success: bool
    total_attempts: int
    summary: str
    step_stopped: bool = False  # True if stopped due to --step mode


@dataclass
class ResumePoint:
    """Specifies where to resume execution from."""

    task_id: str
    phase: ExecutionPhase

    @classmethod
    def parse(cls, resume_str: str) -> "ResumePoint":
        """Parse resume string like 'task3_instruction' or 'task2_validation'."""
        parts = resume_str.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid resume format: {resume_str}. Expected: <task_id>_<phase>"
            )
        task_id, phase_str = parts
        try:
            phase = ExecutionPhase(phase_str)
        except ValueError as err:
            raise ValueError(
                f"Invalid phase: {phase_str}. Expected: instruction or validation"
            ) from err
        return cls(task_id=task_id, phase=phase)

    def __str__(self) -> str:
        return f"{self.task_id}_{self.phase.value}"


@dataclass
class ExecutionHistory:
    """Persistent execution history for resume capability."""

    history_id: str
    yaml_path: str
    started_at: str
    updated_at: str
    completed: bool
    task_results: list[TaskResult]
    completed_task_ids: list[str]
    current_task_id: str | None
    current_phase: str | None
    total_attempts: int
    config_snapshot: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "history_id": self.history_id,
            "yaml_path": self.yaml_path,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed": self.completed,
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "status": tr.status.value,
                    "instruction_output": tr.instruction_output,
                    "validation_output": tr.validation_output,
                    "validation_approved": tr.validation_approved,
                    "attempts": tr.attempts,
                    "error": tr.error,
                }
                for tr in self.task_results
            ],
            "completed_task_ids": self.completed_task_ids,
            "current_task_id": self.current_task_id,
            "current_phase": self.current_phase,
            "total_attempts": self.total_attempts,
            "config_snapshot": self.config_snapshot,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionHistory":
        """Deserialize from dictionary."""
        task_results = [
            TaskResult(
                task_id=tr["task_id"],
                status=TaskStatus(tr["status"]),
                instruction_output=tr.get("instruction_output"),
                validation_output=tr.get("validation_output"),
                validation_approved=tr.get("validation_approved", False),
                attempts=tr.get("attempts", 0),
                error=tr.get("error"),
            )
            for tr in data.get("task_results", [])
        ]
        return cls(
            history_id=data["history_id"],
            yaml_path=data["yaml_path"],
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            completed=data.get("completed", False),
            task_results=task_results,
            completed_task_ids=data.get("completed_task_ids", []),
            current_task_id=data.get("current_task_id"),
            current_phase=data.get("current_phase"),
            total_attempts=data.get("total_attempts", 0),
            config_snapshot=data.get("config_snapshot", {}),
            error=data.get("error"),
        )

    def get_resume_points(self) -> list[str]:
        """Get list of valid resume points based on current state."""
        points = []
        for tr in self.task_results:
            if tr.status == TaskStatus.APPROVED:
                continue
            if tr.instruction_output:
                points.append(f"{tr.task_id}_validation")
            points.append(f"{tr.task_id}_instruction")
        if self.current_task_id:
            if self.current_phase == ExecutionPhase.VALIDATION.value:
                points.append(f"{self.current_task_id}_validation")
            points.append(f"{self.current_task_id}_instruction")
        # Deduplicate while preserving order
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        return unique_points


class YamlType(Enum):
    """Type of YAML file."""

    LOOP_C = "loop_c"  # Task definition
    LOOP_B = "loop_b"  # Requirement definition
    UNKNOWN = "unknown"


def detect_yaml_type(yaml_path: str) -> YamlType:
    """Detect whether YAML file is Loop C (tasks) or Loop B (requirements)."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return YamlType.UNKNOWN

    if "tasks" in data:
        return YamlType.LOOP_C
    if "requirements" in data:
        return YamlType.LOOP_B

    return YamlType.UNKNOWN


def create_sample_task_yaml() -> str:
    """Return sample YAML content for reference."""
    return (_get_examples_dir() / "sample_tasks.yaml").read_text(encoding="utf-8")


# =============================================================================
# Loop B (Requirements Fulfillment) Schema
# =============================================================================


@dataclass
class AcceptanceCriterion:
    """Single acceptance criterion with verification method."""

    criterion: str
    verify: str  # How to verify: method + target + expected state


@dataclass
class Requirement:
    """Single requirement definition (similar to Task format)."""

    id: str
    name: str
    acceptance_criteria: list[AcceptanceCriterion]
    design_decisions: list[str] = field(
        default_factory=list
    )  # Technical decisions to verify
    notes: str | None = None  # Optional notes for additional context


def _parse_acceptance_criteria(criteria_data: list[Any]) -> list[AcceptanceCriterion]:
    """Parse acceptance criteria from YAML data.

    Supports both formats:
    - New format: [{"criterion": "...", "verify": "..."}, ...]
    - Legacy format: ["criterion1", "criterion2", ...]
    """
    result = []
    for item in criteria_data:
        if isinstance(item, str):
            # Legacy format: string only, no verify
            result.append(AcceptanceCriterion(criterion=item, verify=""))
        elif isinstance(item, dict):
            # New format: dict with criterion and verify
            result.append(
                AcceptanceCriterion(
                    criterion=item.get("criterion", ""),
                    verify=item.get("verify", ""),
                )
            )
    return result


@dataclass
class RequirementDefinition:
    """Complete requirement definition from YAML."""

    requirements: list[Requirement]
    common_validation: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RequirementDefinition":
        """Load requirement definition from YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        requirements = [
            Requirement(
                id=req["id"],
                name=req["name"],
                acceptance_criteria=_parse_acceptance_criteria(
                    req.get("acceptance_criteria", [])
                ),
                design_decisions=req.get("design_decisions", []),
                notes=req.get("notes"),
            )
            for req in data.get("requirements", [])
        ]
        common_validation = data.get("common_validation", [])

        return cls(requirements=requirements, common_validation=common_validation)


class LoopBStatus(Enum):
    """Loop B execution status."""

    GENERATING_TASKS = "generating_tasks"
    EXECUTING_TASKS = "executing_tasks"
    VERIFYING_REQUIREMENTS = "verifying_requirements"
    GENERATING_ADDITIONAL_TASKS = "generating_additional_tasks"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LoopBIteration:
    """Record of a single Loop B iteration."""

    iteration_number: int
    tasks_yaml_path: str
    loop_c_history_id: str | None
    verification_result: dict[str, Any] | None
    unmet_requirements: list[str]


@dataclass
class LoopBExecutionHistory:
    """Persistent execution history for Loop B."""

    history_id: str
    requirements_path: str
    tasks_output_dir: str
    started_at: str
    updated_at: str
    status: LoopBStatus
    max_iterations: int
    current_iteration: int
    iterations: list[LoopBIteration]
    completed_task_ids: list[str]
    loop_c_history_ids: list[str]
    final_result: dict[str, Any] | None = None
    error: str | None = None
    requirements_hash: str | None = None  # SHA256 hash of requirements YAML content
    # Verification progress tracking (for step mode resume)
    current_verification_index: int = 0  # Index of requirement being verified
    partial_verification_results: list[dict[str, Any]] | None = None
    # Shared exploration file path (codebase context, test results, artifacts)
    shared_exploration_path: str | None = None
    shared_exploration_iteration: int | None = None  # Which iteration this belongs to

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "history_id": self.history_id,
            "requirements_path": self.requirements_path,
            "tasks_output_dir": self.tasks_output_dir,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "iterations": [
                {
                    "iteration_number": it.iteration_number,
                    "tasks_yaml_path": it.tasks_yaml_path,
                    "loop_c_history_id": it.loop_c_history_id,
                    "verification_result": it.verification_result,
                    "unmet_requirements": it.unmet_requirements,
                }
                for it in self.iterations
            ],
            "completed_task_ids": self.completed_task_ids,
            "loop_c_history_ids": self.loop_c_history_ids,
            "final_result": self.final_result,
            "error": self.error,
            "requirements_hash": self.requirements_hash,
            "current_verification_index": self.current_verification_index,
            "partial_verification_results": self.partial_verification_results,
            "shared_exploration_path": self.shared_exploration_path,
            "shared_exploration_iteration": self.shared_exploration_iteration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopBExecutionHistory":
        """Deserialize from dictionary."""
        iterations = [
            LoopBIteration(
                iteration_number=it["iteration_number"],
                tasks_yaml_path=it["tasks_yaml_path"],
                loop_c_history_id=it.get("loop_c_history_id"),
                verification_result=it.get("verification_result"),
                unmet_requirements=it.get("unmet_requirements", []),
            )
            for it in data.get("iterations", [])
        ]
        return cls(
            history_id=data["history_id"],
            requirements_path=data["requirements_path"],
            tasks_output_dir=data.get("tasks_output_dir", ""),
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            status=LoopBStatus(data["status"]),
            max_iterations=data.get("max_iterations", 3),
            current_iteration=data.get("current_iteration", 0),
            iterations=iterations,
            completed_task_ids=data.get("completed_task_ids", []),
            loop_c_history_ids=data.get("loop_c_history_ids", []),
            final_result=data.get("final_result"),
            error=data.get("error"),
            requirements_hash=data.get("requirements_hash"),
            current_verification_index=data.get("current_verification_index", 0),
            partial_verification_results=data.get("partial_verification_results"),
            shared_exploration_path=data.get("shared_exploration_path"),
            shared_exploration_iteration=data.get("shared_exploration_iteration"),
        )


def create_sample_requirements_yaml() -> str:
    """Return sample requirements YAML content for reference."""
    return (_get_examples_dir() / "sample_requirements.yaml").read_text(
        encoding="utf-8"
    )
