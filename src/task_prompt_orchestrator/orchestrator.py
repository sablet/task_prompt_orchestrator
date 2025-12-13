"""Task orchestrator using Claude Agent SDK."""

import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)

from .models import MODEL_DEFAULT
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
from .templates import render_template

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
            "Task",
            "WebFetch",
            "WebSearch",
        ]
    )
    cwd: str | None = None
    model: str | None = MODEL_DEFAULT
    permission_mode: str = "acceptEdits"
    stream_output: bool = True
    stream_callback: Callable[[str], None] | None = None
    step_mode: bool = False
    _step_executed: bool = False  # Internal flag to track if a step was executed

    def to_dict(self) -> dict[str, Any]:
        """Serialize config for history storage."""
        return {
            "max_retries_per_task": self.max_retries_per_task,
            "max_total_retries": self.max_total_retries,
            "allowed_tools": self.allowed_tools,
            "cwd": self.cwd,
            "model": self.model,
            "permission_mode": self.permission_mode,
            "step_mode": self.step_mode,
        }


HISTORY_DIR = ".task-orchestrator-history"


def save_history_to_file(
    history_dir: Path, history_id: str, data: dict[str, Any]
) -> None:
    """Save history data to a JSON file."""
    history_dir.mkdir(parents=True, exist_ok=True)
    file_path = history_dir / f"{history_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_history_from_file(history_dir: Path, history_id: str) -> dict[str, Any]:
    """Load history data from a JSON file."""
    file_path = history_dir / f"{history_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"History not found: {history_id}")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def list_history_files(history_dir: Path) -> list[dict[str, Any]]:
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

    @staticmethod
    def _generate_history_id(yaml_path: str) -> str:
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

    def find_incomplete_by_path(self, yaml_path: str) -> ExecutionHistory | None:
        """Find incomplete history for a given YAML file path."""
        resolved_path = str(Path(yaml_path).resolve())
        for h in self.list_incomplete_histories():
            if h.yaml_path == resolved_path:
                return h
        return None

    def find_by_path(
        self, yaml_path: str, include_completed: bool = True
    ) -> list[ExecutionHistory]:
        """Find all histories for a given YAML file path."""
        resolved_path = str(Path(yaml_path).resolve())
        histories = (
            self.list_histories()
            if include_completed
            else self.list_incomplete_histories()
        )
        return [h for h in histories if h.yaml_path == resolved_path]

    def delete_history(self, history_id: str) -> bool:
        """Delete a history file."""
        file_path = self._history_file_path(history_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False


def build_instruction_prompt(
    task: Task,
    cwd: str | None = None,
    exploration_path: str | None = None,
) -> str:
    """Build prompt for task instruction execution."""
    return render_template(
        "task_instruction.j2",
        task_name=task.name,
        cwd=cwd,
        instruction=task.instruction,
        exploration_path=exploration_path,
    )


def build_validation_prompt(
    task: Task,
    instruction_output: str,
    common_validation: list[str] | None = None,
    exploration_path: str | None = None,
) -> str:
    """Build prompt for task validation."""
    return render_template(
        "task_validation.j2",
        task_name=task.name,
        instruction=task.instruction,
        instruction_output=instruction_output,
        validation_criteria=task.validation,
        common_validation=common_validation or [],
        exploration_path=exploration_path,
    )


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
    if "all criteria" in output_lower and (
        "met" in output_lower or "pass" in output_lower
    ):
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


def _format_tool_info(block: ToolUseBlock) -> str:
    """Format tool use block info for display."""
    tool_info = f"{CYAN}▶ {block.name}{RESET}"
    inp = block.input
    if not isinstance(inp, dict):
        return tool_info + "\n"
    # Extract relevant info based on tool type
    detail = None
    if "command" in inp:  # Bash
        detail = inp["command"][:80]
    elif "file_path" in inp:  # Read, Write, Edit
        detail = inp["file_path"]
    elif "pattern" in inp:  # Glob, Grep
        detail = inp["pattern"][:60]
        if "path" in inp:
            detail += f" in {inp['path']}"
    elif "prompt" in inp:  # Task
        detail = inp["prompt"]
    if detail:
        tool_info += f" {DIM}{detail}{RESET}"
    return tool_info + "\n"


def _process_text_block(
    block: TextBlock,
    output_parts: list[str],
    callback: Callable[[str], None],
    stream: bool,
) -> None:
    """Process a text block from assistant message."""
    output_parts.append(block.text)
    if stream:
        text = block.text
        # テキストとコマンドの間に空行を入れる
        if text and not text.endswith("\n\n"):
            text = text.rstrip("\n") + "\n\n"
        callback(text)


def _process_tool_result(
    message: ToolResultBlock, callback: Callable[[str], None]
) -> None:
    """Process a tool result block."""
    if not message.content:
        return
    content_str = str(message.content)
    if len(content_str) > 200:
        content_str = content_str[:200] + "..."
    callback(f"{DIM}  └─ {content_str}{RESET}\n")


class StepModeInterrupt(Exception):
    """Raised when step mode requires stopping after a claude query."""

    def __init__(self, output: str, phase: str = ""):
        self.output = output
        self.phase = phase
        super().__init__(f"Step mode: stopped after {phase}")


async def run_claude_query(
    prompt: str,
    config: OrchestratorConfig,
    phase: str = "",
    model_override: str | None = None,
) -> str:
    """Run a single Claude Code query and return text output with streaming.

    Args:
        prompt: The prompt to send to Claude.
        config: Orchestrator configuration.
        phase: Optional phase name for logging/debugging.
        model_override: Optional model to use instead of config.model.

    Raises:
        StepModeInterrupt: If step_mode is True, raised after query completes.
    """
    options = ClaudeAgentOptions(
        allowed_tools=config.allowed_tools,
        permission_mode=config.permission_mode,
    )
    if config.cwd:
        options.cwd = config.cwd
    model_to_use = model_override or config.model
    if model_to_use:
        options.model = model_to_use

    output_parts: list[str] = []
    callback = config.stream_callback or default_stream_callback
    stream = config.stream_output

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    _process_text_block(block, output_parts, callback, stream)
                elif isinstance(block, ToolUseBlock) and stream:
                    callback(_format_tool_info(block))
        elif isinstance(message, ToolResultBlock) and stream:
            _process_tool_result(message, callback)
        elif isinstance(message, ResultMessage) and stream:
            cost = getattr(message, "cost_usd", None)
            if cost:
                callback(f"{DIM}[cost: ${cost:.4f}]{RESET}\n")

    if stream:
        callback("\n")

    output = "\n".join(output_parts)

    # Step mode: mark that a step was executed
    if config.step_mode:
        config._step_executed = True

    return output


async def execute_instruction_phase(
    task: Task,
    config: OrchestratorConfig,
    task_number: int,
    total_tasks: int,
    attempt: int,
    previous_feedback: str | None = None,
    exploration_path: str | None = None,
) -> str:
    """Execute the instruction phase of a task. Returns instruction output."""
    callback = config.stream_callback or default_stream_callback
    stream = config.stream_output

    instruction_prompt = build_instruction_prompt(
        task, cwd=config.cwd, exploration_path=exploration_path
    )
    if previous_feedback:
        instruction_prompt += f"\n\n### Previous Attempt Feedback:\n{previous_feedback}\n\nPlease address the issues noted above."

    logger.info(f"Executing instruction for task: {task.id}")

    if stream:
        callback(f"\n{BOLD}{'=' * 60}{RESET}\n")
        callback(
            f"{BOLD}📋 INSTRUCTION [{task_number}/{total_tasks}] (attempt {attempt}): {task.name}{RESET}\n"
        )
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
    common_validation: list[str] | None = None,
    exploration_path: str | None = None,
) -> tuple[str, bool, str]:
    """Execute the validation phase of a task. Returns (output, approved, feedback)."""
    callback = config.stream_callback or default_stream_callback
    stream = config.stream_output

    validation_prompt = build_validation_prompt(
        task, instruction_output, common_validation, exploration_path=exploration_path
    )
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


@dataclass
class StepResult:
    """Result when step mode stops execution."""

    task_result: TaskResult
    stopped_after: str  # "instruction" or "validation"


async def execute_task(
    task: Task,
    config: OrchestratorConfig,
    task_number: int = 1,
    total_tasks: int = 1,
    attempt: int = 1,
    previous_feedback: str | None = None,
    skip_instruction: bool = False,
    existing_instruction_output: str | None = None,
    common_validation: list[str] | None = None,
    exploration_path: str | None = None,
) -> TaskResult | StepResult:
    """Execute a single task with instruction and validation.

    Args:
        skip_instruction: If True, skip instruction phase and use existing_instruction_output
        existing_instruction_output: Pre-existing instruction output for resume from validation
        exploration_path: Path to pre-collected exploration context file (optional)

    Returns:
        TaskResult on normal completion, StepResult if step mode stopped execution.
    """
    result = TaskResult(task_id=task.id, status=TaskStatus.IN_PROGRESS)
    callback = config.stream_callback or default_stream_callback
    stream = config.stream_output

    try:
        if skip_instruction and existing_instruction_output:
            result.instruction_output = existing_instruction_output
            if stream:
                callback(f"\n{BOLD}{'=' * 60}{RESET}\n")
                callback(
                    f"{BOLD}📋 RESUMING from VALIDATION [{task_number}/{total_tasks}]: {task.name}{RESET}\n"
                )
                callback(f"{BOLD}{'=' * 60}{RESET}\n")
        else:
            result.instruction_output = await execute_instruction_phase(
                task,
                config,
                task_number,
                total_tasks,
                attempt,
                previous_feedback,
                exploration_path,
            )
            # Step mode: stop after instruction
            if config.step_mode and config._step_executed:
                logger.info(f"Step mode: stopping after instruction for task {task.id}")
                if stream:
                    callback(
                        f"\n{YELLOW}⏸ STEP MODE: stopped after instruction{RESET}\n"
                    )
                return StepResult(task_result=result, stopped_after="instruction")

        validation_output, approved, feedback = await execute_validation_phase(
            task,
            config,
            task_number,
            total_tasks,
            result.instruction_output,
            common_validation,
            exploration_path,
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

        # Step mode: stop after validation (even if approved)
        if config.step_mode and config._step_executed:
            logger.info(f"Step mode: stopping after validation for task {task.id}")
            if stream:
                callback(f"\n{YELLOW}⏸ STEP MODE: stopped after validation{RESET}\n")
            return StepResult(task_result=result, stopped_after="validation")

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


@dataclass
class _OrchestratorState:
    """Mutable state for orchestrator execution."""

    task_results: list[TaskResult] = field(default_factory=list)
    total_attempts: int = 0
    completed_tasks: set[str] = field(default_factory=set)
    start_task_index: int = 0
    skip_to_validation: bool = False
    existing_instruction_output: str | None = None


def _restore_state_from_history(
    state: _OrchestratorState, resume_history: ExecutionHistory
) -> None:
    """Restore orchestrator state from previous execution history."""
    for tr in resume_history.task_results:
        if tr.status == TaskStatus.APPROVED:
            state.completed_tasks.add(tr.task_id)
            state.task_results.append(tr)
    state.total_attempts = resume_history.total_attempts


def _determine_start_point(
    state: _OrchestratorState,
    resume_point: ResumePoint,
    task_map: dict[str, tuple[int, Task]],
    resume_history: ExecutionHistory | None,
) -> None:
    """Determine starting point for orchestration from resume point."""
    if resume_point.task_id not in task_map:
        raise ValueError(f"Resume task not found: {resume_point.task_id}")
    state.start_task_index = task_map[resume_point.task_id][0] - 1
    if resume_point.phase != ExecutionPhase.VALIDATION:
        return
    state.skip_to_validation = True
    if resume_history:
        for tr in resume_history.task_results:
            if tr.task_id == resume_point.task_id and tr.instruction_output:
                state.existing_instruction_output = tr.instruction_output
                break
    if not state.existing_instruction_output:
        raise ValueError(
            f"Cannot resume from validation: no instruction output for {resume_point.task_id}"
        )


def _check_dependency(task: Task, completed_tasks: set[str]) -> str | None:
    """Check if task dependencies are met. Returns failed dep name or None."""
    for dep in task.depends_on:
        if dep not in completed_tasks:
            return dep
    return None


def _save_history(
    history: ExecutionHistory | None,
    history_manager: HistoryManager | None,
    state: _OrchestratorState,
    error: str | None = None,
    completed: bool = False,
) -> None:
    """Save history if history tracking is enabled."""
    if not history or not history_manager:
        return
    history.task_results = state.task_results
    history.total_attempts = state.total_attempts
    if error:
        history.error = error
        history.completed = False
    if completed:
        history.completed_task_ids = list(state.completed_tasks)
        history.completed = True
        history.current_task_id = None
        history.current_phase = None
    history_manager.save_history(history)


def _update_history_result(
    history: ExecutionHistory | None,
    history_manager: HistoryManager | None,
    result: TaskResult,
    total_attempts: int,
) -> None:
    """Update or add task result in history."""
    if not history or not history_manager:
        return
    existing_idx = next(
        (
            i
            for i, tr in enumerate(history.task_results)
            if tr.task_id == result.task_id
        ),
        None,
    )
    if existing_idx is not None:
        history.task_results[existing_idx] = result
    else:
        history.task_results.append(result)
    history.total_attempts = total_attempts
    history_manager.save_history(history)


def _update_task_results(task_results: list[TaskResult], result: TaskResult) -> None:
    """Update task results list, avoiding duplicates."""
    existing_idx = next(
        (i for i, tr in enumerate(task_results) if tr.task_id == result.task_id),
        None,
    )
    if existing_idx is not None:
        task_results[existing_idx] = result
    else:
        task_results.append(result)


async def _execute_task_with_retries(
    task: Task,
    config: OrchestratorConfig,
    state: _OrchestratorState,
    task_index: int,
    total_tasks: int,
    history: ExecutionHistory | None,
    history_manager: HistoryManager | None,
    resume_point: ResumePoint | None,
    common_validation: list[str] | None = None,
    exploration_path: str | None = None,
) -> tuple[TaskResult | StepResult | None, str | None]:
    """Execute a single task with retry logic.

    Returns (result, error_summary) - error_summary is set if max retries exceeded.
    StepResult is returned if step mode stopped execution.
    """
    attempts = 0
    feedback: str | None = None
    result: TaskResult | StepResult | None = None
    first_attempt = True
    skip_instruction = state.skip_to_validation
    instruction_output = state.existing_instruction_output

    while attempts < config.max_retries_per_task:
        attempts += 1
        state.total_attempts += 1

        if state.total_attempts > config.max_total_retries:
            logger.error("Max total retries exceeded")
            error_result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                attempts=attempts,
                error="Max total retries exceeded",
            )
            state.task_results.append(error_result)
            return None, "Max total retries exceeded"

        should_skip = bool(
            first_attempt
            and skip_instruction
            and resume_point
            and task.id == resume_point.task_id
        )
        first_attempt = False

        if history and history_manager:
            phase = (
                ExecutionPhase.VALIDATION if should_skip else ExecutionPhase.INSTRUCTION
            )
            history.current_phase = phase.value
            history_manager.save_history(history)

        logger.info(f"Task {task.id}: attempt {attempts}/{config.max_retries_per_task}")
        exec_result = await execute_task(
            task,
            config,
            task_number=task_index,
            total_tasks=total_tasks,
            attempt=attempts,
            previous_feedback=feedback,
            skip_instruction=should_skip,
            existing_instruction_output=instruction_output if should_skip else None,
            common_validation=common_validation,
            exploration_path=exploration_path,
        )

        # Handle StepResult (step mode stopped execution)
        if isinstance(exec_result, StepResult):
            exec_result.task_result.attempts = attempts
            _update_history_result(
                history, history_manager, exec_result.task_result, state.total_attempts
            )
            # Update history phase based on where we stopped
            if history and history_manager:
                if exec_result.stopped_after == "instruction":
                    history.current_phase = ExecutionPhase.VALIDATION.value
                else:
                    history.current_phase = ExecutionPhase.INSTRUCTION.value
                history_manager.save_history(history)
            return exec_result, None

        result = exec_result
        result.attempts = attempts

        _update_history_result(history, history_manager, result, state.total_attempts)

        if result.status == TaskStatus.APPROVED:
            logger.info(f"Task {task.id} approved on attempt {attempts}")
            state.completed_tasks.add(task.id)
            if history and history_manager:
                history.completed_task_ids = list(state.completed_tasks)
                history_manager.save_history(history)
            break
        if result.status == TaskStatus.PERMISSION_DENIED:
            logger.error(f"Task {task.id} failed: permission denied (no retry)")
            break
        if result.status == TaskStatus.DECLINED:
            feedback = result.error
            logger.warning(f"Task {task.id} declined: {feedback}")
            skip_instruction = False
            instruction_output = None
        else:
            logger.error(f"Task {task.id} failed: {result.error}")
            break

    return result, None


async def run_orchestrator(
    task_definition: TaskDefinition,
    config: OrchestratorConfig | None = None,
    yaml_path: str | None = None,
    history_manager: HistoryManager | None = None,
    resume_point: ResumePoint | None = None,
    resume_history: ExecutionHistory | None = None,
    exploration_path: str | None = None,
) -> OrchestratorResult:
    """Run the full task orchestrator with history tracking and resume support."""
    if config is None:
        config = OrchestratorConfig()

    history: ExecutionHistory | None = None
    if history_manager and yaml_path:
        history = resume_history or history_manager.create_history(yaml_path, config)

    state = _OrchestratorState()
    if resume_history:
        _restore_state_from_history(state, resume_history)

    task_map = {t.id: (i, t) for i, t in enumerate(task_definition.tasks, start=1)}

    if resume_point:
        _determine_start_point(state, resume_point, task_map, resume_history)

    total_tasks = len(task_definition.tasks)

    for task_index, task in enumerate(task_definition.tasks, start=1):
        if task_index - 1 < state.start_task_index or task.id in state.completed_tasks:
            continue

        failed_dep = _check_dependency(task, state.completed_tasks)
        if failed_dep:
            logger.error(f"Dependency {failed_dep} not completed for task {task.id}")
            error_result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=f"Dependency {failed_dep} not completed",
            )
            state.task_results.append(error_result)
            _save_history(history, history_manager, state, error=error_result.error)
            return OrchestratorResult(
                task_results=state.task_results,
                success=False,
                total_attempts=state.total_attempts,
                summary=f"Failed due to unmet dependency: {failed_dep}",
            )

        if history and history_manager:
            history.current_task_id = task.id
            history.current_phase = ExecutionPhase.INSTRUCTION.value
            history_manager.save_history(history)

        result, error_summary = await _execute_task_with_retries(
            task,
            config,
            state,
            task_index,
            total_tasks,
            history,
            history_manager,
            resume_point,
            common_validation=task_definition.common_validation,
            exploration_path=exploration_path,
        )

        if error_summary:
            _save_history(history, history_manager, state, error=error_summary)
            return OrchestratorResult(
                task_results=state.task_results,
                success=False,
                total_attempts=state.total_attempts,
                summary=error_summary,
            )

        # Handle StepResult (step mode stopped execution)
        if isinstance(result, StepResult):
            _update_task_results(state.task_results, result.task_result)
            logger.info(
                f"Step mode: stopping after {result.stopped_after} for task {task.id}"
            )
            return OrchestratorResult(
                task_results=state.task_results,
                success=True,
                total_attempts=state.total_attempts,
                summary=f"Step mode: stopped after {result.stopped_after} for task {task.id}",
                step_stopped=True,
            )

        if result:
            _update_task_results(state.task_results, result)
            if result.status != TaskStatus.APPROVED:
                summary = (
                    f"Task {task.id} failed: permission denied"
                    if result.status == TaskStatus.PERMISSION_DENIED
                    else f"Task {task.id} failed after {result.attempts} attempts"
                )
                _save_history(history, history_manager, state, error=summary)
                return OrchestratorResult(
                    task_results=state.task_results,
                    success=False,
                    total_attempts=state.total_attempts,
                    summary=summary,
                )

        state.skip_to_validation = False
        state.existing_instruction_output = None

    _save_history(history, history_manager, state, completed=True)

    return OrchestratorResult(
        task_results=state.task_results,
        success=True,
        total_attempts=state.total_attempts,
        summary=f"All {len(task_definition.tasks)} tasks completed successfully",
    )
