"""Task orchestrator using Claude Agent SDK."""

import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)

from .schema import (
    ExecutionHistory,
    ExecutionPhase,
    OrchestratorResult,
    ResumePoint,
    Task,
    TaskDefinition,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ANSI color codes
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def default_stream_callback(text: str) -> None:
    """Default callback that prints to stdout."""
    sys.stdout.write(text)
    sys.stdout.flush()


@dataclass
class OrchestratorConfig:
    """Configuration for task orchestrator."""

    max_retries_per_task: int = 3
    max_total_retries: int = 10
    allowed_tools: list[str] = field(
        default_factory=lambda: [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "NotebookEdit",
            "WebFetch",
            "WebSearch",
        ]
    )
    cwd: str | None = None
    model: str | None = None
    permission_mode: str = "acceptEdits"
    stream_output: bool = True
    stream_callback: Callable[[str], None] | None = None

    def to_dict(self) -> dict:
        """Serialize config for history storage."""
        return {
            "max_retries_per_task": self.max_retries_per_task,
            "max_total_retries": self.max_total_retries,
            "allowed_tools": self.allowed_tools,
            "cwd": self.cwd,
            "model": self.model,
            "permission_mode": self.permission_mode,
        }


HISTORY_DIR = ".task-orchestrator-history"


def save_history_to_file(history_dir: Path, history_id: str, data: dict) -> None:
    """Save history data to a JSON file."""
    history_dir.mkdir(parents=True, exist_ok=True)
    file_path = history_dir / f"{history_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_history_from_file(history_dir: Path, history_id: str) -> dict:
    """Load history data from a JSON file."""
    file_path = history_dir / f"{history_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"History not found: {history_id}")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def list_history_files(history_dir: Path) -> list[dict]:
    """List all history files in a directory."""
    if not history_dir.exists():
        return []
    histories = []
    for file_path in history_dir.glob("*.json"):
        try:
            with open(file_path, encoding="utf-8") as f:
                histories.append(json.load(f))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load history {file_path}: {e}")
    return histories


def delete_history_file(history_dir: Path, history_id: str) -> bool:
    """Delete a history file."""
    file_path = history_dir / f"{history_id}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False


class HistoryManager:
    """Manages execution history persistence."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.history_dir = self.base_dir / HISTORY_DIR

    def _ensure_dir(self) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _generate_history_id(self, yaml_path: str) -> str:
        yaml_name = Path(yaml_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{yaml_name}_{timestamp}"

    def _history_file_path(self, history_id: str) -> Path:
        return self.history_dir / f"{history_id}.json"

    def create_history(
        self, yaml_path: str, config: OrchestratorConfig
    ) -> ExecutionHistory:
        """Create a new execution history."""
        self._ensure_dir()
        now = datetime.now().isoformat()
        history = ExecutionHistory(
            history_id=self._generate_history_id(yaml_path),
            yaml_path=str(Path(yaml_path).resolve()),
            started_at=now,
            updated_at=now,
            completed=False,
            task_results=[],
            completed_task_ids=[],
            current_task_id=None,
            current_phase=None,
            total_attempts=0,
            config_snapshot=config.to_dict(),
        )
        self.save_history(history)
        return history

    def save_history(self, history: ExecutionHistory) -> None:
        """Save execution history to file."""
        self._ensure_dir()
        history.updated_at = datetime.now().isoformat()
        file_path = self._history_file_path(history.history_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history.to_dict(), f, indent=2, ensure_ascii=False)

    def load_history(self, history_id: str) -> ExecutionHistory:
        """Load execution history from file."""
        file_path = self._history_file_path(history_id)
        if not file_path.exists():
            raise FileNotFoundError(f"History not found: {history_id}")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return ExecutionHistory.from_dict(data)

    def list_histories(self) -> list[ExecutionHistory]:
        """List all execution histories, sorted by date descending."""
        if not self.history_dir.exists():
            return []
        histories = []
        for file_path in self.history_dir.glob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                histories.append(ExecutionHistory.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load history {file_path}: {e}")
        histories.sort(key=lambda h: h.started_at, reverse=True)
        return histories

    def list_incomplete_histories(self) -> list[ExecutionHistory]:
        """List only incomplete (resumable) histories."""
        return [h for h in self.list_histories() if not h.completed]

    def delete_history(self, history_id: str) -> bool:
        """Delete a history file."""
        file_path = self._history_file_path(history_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False


def build_instruction_prompt(task: Task, cwd: str | None = None) -> str:
    """Build prompt for task instruction execution."""
    cwd_note = f"\n**Working directory: {cwd}**\nAll relative paths should be resolved from this directory.\n" if cwd else ""
    return f"""## Task: {task.name}
{cwd_note}
{task.instruction}

Please complete this task. When done, provide a summary of what was accomplished.
"""


def build_validation_prompt(task: Task, instruction_output: str) -> str:
    """Build prompt for task validation."""
    validation_criteria = "\n".join(f"- {v}" for v in task.validation)

    return f"""## Validation for Task: {task.name}

The following instruction was executed:
{task.instruction}

### Execution Output:
{instruction_output}

### Validation Criteria:
{validation_criteria}

Please verify ALL of the above validation criteria are met.

Respond with a JSON object in the following format:
```json
{{
  "approved": true/false,
  "checks": [
    {{"criterion": "criterion text", "passed": true/false, "details": "explanation"}}
  ],
  "summary": "overall validation summary",
  "feedback": "if not approved, specific feedback for retry"
}}
```
"""


def extract_validation_result(output: str) -> tuple[bool, str]:
    """Extract validation result from Claude output.

    Returns:
        (approved, feedback)
    """
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data.get("approved", False), data.get("feedback", "")
        except json.JSONDecodeError:
            pass

    # Fallback: look for raw JSON
    try:
        json_match = re.search(r"(\{[^{}]*\"approved\"[^{}]*\})", output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            return data.get("approved", False), data.get("feedback", "")
    except json.JSONDecodeError:
        pass

    # Heuristic fallback
    output_lower = output.lower()
    if "approved" in output_lower and "true" in output_lower:
        return True, ""
    if "all criteria" in output_lower and ("met" in output_lower or "pass" in output_lower):
        return True, ""

    return False, "Could not determine validation result"


def is_permission_error(feedback: str) -> bool:
    """Check if the feedback indicates a permission error."""
    if not feedback:
        return False
    feedback_lower = feedback.lower()
    permission_indicators = [
        "permission",
        "haven't granted",
        "not allowed",
        "access denied",
        "unauthorized",
    ]
    return any(indicator in feedback_lower for indicator in permission_indicators)


async def run_claude_query(prompt: str, config: OrchestratorConfig, phase: str = "") -> str:
    """Run a single Claude Code query and return text output with streaming."""
    options = ClaudeAgentOptions(
        allowed_tools=config.allowed_tools,
        permission_mode=config.permission_mode,
    )
    if config.cwd:
        options.cwd = config.cwd
    if config.model:
        options.model = config.model

    output_parts: list[str] = []
    callback = config.stream_callback or default_stream_callback
    stream = config.stream_output

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    output_parts.append(block.text)
                    if stream:
                        callback(block.text)
                elif isinstance(block, ToolUseBlock):
                    if stream:
                        tool_info = f"\n{CYAN}▶ {block.name}{RESET}"
                        if block.name in {"Bash", "Read", "Write", "Edit"}:
                            # Show relevant input for common tools
                            inp = block.input
                            if isinstance(inp, dict):
                                if "command" in inp:
                                    tool_info += f" {DIM}{inp['command'][:80]}{RESET}"
                                elif "file_path" in inp:
                                    tool_info += f" {DIM}{inp['file_path']}{RESET}"
                        callback(tool_info + "\n")
        elif isinstance(message, ToolResultBlock):
            if stream and message.content:
                content_str = str(message.content)
                # Truncate long tool results
                if len(content_str) > 200:
                    content_str = content_str[:200] + "..."
                callback(f"{DIM}  └─ {content_str}{RESET}\n")
        elif isinstance(message, ResultMessage) and stream:
            cost = getattr(message, "cost_usd", None)
            if cost:
                callback(f"{DIM}[cost: ${cost:.4f}]{RESET}\n")

    if stream:
        callback("\n")

    return "\n".join(output_parts)


async def execute_instruction_phase(
    task: Task,
    config: OrchestratorConfig,
    task_number: int,
    total_tasks: int,
    attempt: int,
    previous_feedback: str | None = None,
) -> str:
    """Execute the instruction phase of a task. Returns instruction output."""
    callback = config.stream_callback or default_stream_callback
    stream = config.stream_output

    instruction_prompt = build_instruction_prompt(task, cwd=config.cwd)
    if previous_feedback:
        instruction_prompt += f"\n\n### Previous Attempt Feedback:\n{previous_feedback}\n\nPlease address the issues noted above."

    logger.info(f"Executing instruction for task: {task.id}")

    if stream:
        callback(f"\n{BOLD}{'=' * 60}{RESET}\n")
        callback(f"{BOLD}📋 INSTRUCTION [{task_number}/{total_tasks}] (attempt {attempt}): {task.name}{RESET}\n")
        callback(f"{BOLD}{'=' * 60}{RESET}\n")
        callback(f"{DIM}>>> PROMPT >>>{RESET}\n")
        callback(f"{DIM}{instruction_prompt}{RESET}\n")
        callback(f"{DIM}<<< END PROMPT <<<{RESET}\n\n")

    return await run_claude_query(instruction_prompt, config)


async def execute_validation_phase(
    task: Task,
    config: OrchestratorConfig,
    task_number: int,
    total_tasks: int,
    instruction_output: str,
) -> tuple[str, bool, str]:
    """Execute the validation phase of a task. Returns (output, approved, feedback)."""
    callback = config.stream_callback or default_stream_callback
    stream = config.stream_output

    validation_prompt = build_validation_prompt(task, instruction_output)
    if stream:
        callback(f"\n{BOLD}{'-' * 60}{RESET}\n")
        callback(f"{YELLOW}🔍 VALIDATION [{task_number}/{total_tasks}]{RESET}\n")
        callback(f"{BOLD}{'-' * 60}{RESET}\n")
        callback(f"{DIM}>>> PROMPT >>>{RESET}\n")
        callback(f"{DIM}{validation_prompt}{RESET}\n")
        callback(f"{DIM}<<< END PROMPT <<<{RESET}\n\n")

    logger.info(f"Executing validation for task: {task.id}")
    validation_output = await run_claude_query(validation_prompt, config)

    approved, feedback = extract_validation_result(validation_output)
    return validation_output, approved, feedback


async def execute_task(
    task: Task,
    config: OrchestratorConfig,
    task_number: int = 1,
    total_tasks: int = 1,
    attempt: int = 1,
    previous_feedback: str | None = None,
    skip_instruction: bool = False,
    existing_instruction_output: str | None = None,
) -> TaskResult:
    """Execute a single task with instruction and validation.

    Args:
        skip_instruction: If True, skip instruction phase and use existing_instruction_output
        existing_instruction_output: Pre-existing instruction output for resume from validation
    """
    result = TaskResult(task_id=task.id, status=TaskStatus.IN_PROGRESS)
    callback = config.stream_callback or default_stream_callback
    stream = config.stream_output

    try:
        if skip_instruction and existing_instruction_output:
            result.instruction_output = existing_instruction_output
            if stream:
                callback(f"\n{BOLD}{'=' * 60}{RESET}\n")
                callback(f"{BOLD}📋 RESUMING from VALIDATION [{task_number}/{total_tasks}]: {task.name}{RESET}\n")
                callback(f"{BOLD}{'=' * 60}{RESET}\n")
        else:
            result.instruction_output = await execute_instruction_phase(
                task, config, task_number, total_tasks, attempt, previous_feedback
            )

        validation_output, approved, feedback = await execute_validation_phase(
            task, config, task_number, total_tasks, result.instruction_output
        )
        result.validation_output = validation_output
        result.validation_approved = approved

        if approved:
            result.status = TaskStatus.APPROVED
            if stream:
                callback(f"\n{GREEN}✅ APPROVED{RESET}\n")
        elif is_permission_error(feedback):
            result.status = TaskStatus.PERMISSION_DENIED
            result.error = feedback
            if stream:
                callback(f"\n{YELLOW}⚠ PERMISSION DENIED: {feedback}{RESET}\n")
        else:
            result.status = TaskStatus.DECLINED
            result.error = feedback
            if stream:
                callback(f"\n{YELLOW}❌ DECLINED: {feedback}{RESET}\n")
    except Exception as e:
        result.status = TaskStatus.FAILED
        result.error = str(e)
        logger.exception(f"Task {task.id} failed with exception")

    return result


class Orchestrator:
    """Task orchestrator class for programmatic use."""

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        history_manager: HistoryManager | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.history_manager = history_manager

    async def run(
        self,
        task_definition: TaskDefinition,
        yaml_path: str | None = None,
        resume_point: ResumePoint | None = None,
        resume_history: ExecutionHistory | None = None,
    ) -> OrchestratorResult:
        """Run the orchestrator."""
        return await run_orchestrator(
            task_definition=task_definition,
            config=self.config,
            yaml_path=yaml_path,
            history_manager=self.history_manager,
            resume_point=resume_point,
            resume_history=resume_history,
        )


async def run_orchestrator(
    task_definition: TaskDefinition,
    config: OrchestratorConfig | None = None,
    yaml_path: str | None = None,
    history_manager: HistoryManager | None = None,
    resume_point: ResumePoint | None = None,
    resume_history: ExecutionHistory | None = None,
) -> OrchestratorResult:
    """Run the full task orchestrator with history tracking and resume support.

    Flow:
    - For each task in order (respecting dependencies):
      - Execute instruction
      - Run validation
      - If approved: proceed to next task
      - If declined: retry same task with feedback (up to max_retries)

    Args:
        task_definition: Tasks to execute
        config: Orchestrator configuration
        yaml_path: Path to YAML file (for history tracking)
        history_manager: History manager instance (enables history tracking)
        resume_point: Where to resume from (e.g., ResumePoint("task3", ExecutionPhase.VALIDATION))
        resume_history: Previous execution history to resume from
    """
    if config is None:
        config = OrchestratorConfig()

    # Initialize history tracking
    history: ExecutionHistory | None = None
    if history_manager and yaml_path:
        history = resume_history or history_manager.create_history(yaml_path, config)

    task_results: list[TaskResult] = []
    total_attempts = 0
    completed_tasks: set[str] = set()

    # Restore state from history if resuming
    if resume_history:
        for tr in resume_history.task_results:
            if tr.status == TaskStatus.APPROVED:
                completed_tasks.add(tr.task_id)
                task_results.append(tr)
        total_attempts = resume_history.total_attempts

    # Build task index map
    task_map = {t.id: (i, t) for i, t in enumerate(task_definition.tasks, start=1)}

    # Determine starting point
    start_task_index = 0
    skip_to_validation = False
    existing_instruction_output: str | None = None

    if resume_point:
        if resume_point.task_id not in task_map:
            raise ValueError(f"Resume task not found: {resume_point.task_id}")
        start_task_index = task_map[resume_point.task_id][0] - 1
        if resume_point.phase == ExecutionPhase.VALIDATION:
            skip_to_validation = True
            # Find existing instruction output from history
            if resume_history:
                for tr in resume_history.task_results:
                    if tr.task_id == resume_point.task_id and tr.instruction_output:
                        existing_instruction_output = tr.instruction_output
                        break
            if not existing_instruction_output:
                raise ValueError(
                    f"Cannot resume from validation: no instruction output found for {resume_point.task_id}"
                )

    total_tasks = len(task_definition.tasks)

    for task_index, task in enumerate(task_definition.tasks, start=1):
        # Skip tasks before resume point
        if task_index - 1 < start_task_index:
            continue

        # Skip already completed tasks
        if task.id in completed_tasks:
            continue

        # Check dependencies
        for dep in task.depends_on:
            if dep not in completed_tasks:
                logger.error(f"Dependency {dep} not completed for task {task.id}")
                error_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=f"Dependency {dep} not completed",
                )
                task_results.append(error_result)
                if history and history_manager:
                    history.task_results = task_results
                    history.completed = False
                    history.error = f"Dependency {dep} not completed"
                    history_manager.save_history(history)
                return OrchestratorResult(
                    task_results=task_results,
                    success=False,
                    total_attempts=total_attempts,
                    summary=f"Failed due to unmet dependency: {dep}",
                )

        # Update history: current task
        if history and history_manager:
            history.current_task_id = task.id
            history.current_phase = ExecutionPhase.INSTRUCTION.value
            history_manager.save_history(history)

        # Execute task with retry logic
        attempts = 0
        feedback: str | None = None
        result: TaskResult | None = None
        first_attempt_of_task = True

        while attempts < config.max_retries_per_task:
            attempts += 1
            total_attempts += 1

            if total_attempts > config.max_total_retries:
                logger.error("Max total retries exceeded")
                error_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    attempts=attempts,
                    error="Max total retries exceeded",
                )
                task_results.append(error_result)
                if history and history_manager:
                    history.task_results = task_results
                    history.total_attempts = total_attempts
                    history.error = "Max total retries exceeded"
                    history_manager.save_history(history)
                return OrchestratorResult(
                    task_results=task_results,
                    success=False,
                    total_attempts=total_attempts,
                    summary="Max total retries exceeded",
                )

            # Determine if we should skip instruction (resume from validation)
            should_skip_instruction = (
                first_attempt_of_task
                and skip_to_validation
                and task.id == resume_point.task_id
                if resume_point
                else False
            )
            first_attempt_of_task = False

            # Update history: phase tracking
            if history and history_manager:
                if not should_skip_instruction:
                    history.current_phase = ExecutionPhase.INSTRUCTION.value
                else:
                    history.current_phase = ExecutionPhase.VALIDATION.value
                history_manager.save_history(history)

            logger.info(f"Task {task.id}: attempt {attempts}/{config.max_retries_per_task}")
            result = await execute_task(
                task,
                config,
                task_number=task_index,
                total_tasks=total_tasks,
                attempt=attempts,
                previous_feedback=feedback,
                skip_instruction=should_skip_instruction,
                existing_instruction_output=existing_instruction_output if should_skip_instruction else None,
            )
            result.attempts = attempts

            # Update history: task result
            if history and history_manager:
                # Update or add result
                existing_idx = next(
                    (i for i, tr in enumerate(history.task_results) if tr.task_id == task.id),
                    None,
                )
                if existing_idx is not None:
                    history.task_results[existing_idx] = result
                else:
                    history.task_results.append(result)
                history.total_attempts = total_attempts
                history_manager.save_history(history)

            if result.status == TaskStatus.APPROVED:
                logger.info(f"Task {task.id} approved on attempt {attempts}")
                completed_tasks.add(task.id)
                if history and history_manager:
                    history.completed_task_ids = list(completed_tasks)
                    history_manager.save_history(history)
                break
            if result.status == TaskStatus.PERMISSION_DENIED:
                logger.error(f"Task {task.id} failed: permission denied (no retry)")
                break
            if result.status == TaskStatus.DECLINED:
                feedback = result.error
                logger.warning(f"Task {task.id} declined: {feedback}")
                # Clear skip flag for retry
                skip_to_validation = False
                existing_instruction_output = None
            else:
                logger.error(f"Task {task.id} failed: {result.error}")
                break

        if result:
            # Update task_results (avoid duplicates)
            existing_idx = next(
                (i for i, tr in enumerate(task_results) if tr.task_id == task.id),
                None,
            )
            if existing_idx is not None:
                task_results[existing_idx] = result
            else:
                task_results.append(result)

            if result.status != TaskStatus.APPROVED:
                if result.status == TaskStatus.PERMISSION_DENIED:
                    summary = f"Task {task.id} failed: permission denied"
                else:
                    summary = f"Task {task.id} failed after {attempts} attempts"
                if history and history_manager:
                    history.task_results = task_results
                    history.error = summary
                    history_manager.save_history(history)
                return OrchestratorResult(
                    task_results=task_results,
                    success=False,
                    total_attempts=total_attempts,
                    summary=summary,
                )

        # Reset skip flag after first task
        skip_to_validation = False
        existing_instruction_output = None

    # Mark history as completed
    if history and history_manager:
        history.task_results = task_results
        history.completed_task_ids = list(completed_tasks)
        history.completed = True
        history.current_task_id = None
        history.current_phase = None
        history_manager.save_history(history)

    return OrchestratorResult(
        task_results=task_results,
        success=True,
        total_attempts=total_attempts,
        summary=f"All {len(task_definition.tasks)} tasks completed successfully",
    )
