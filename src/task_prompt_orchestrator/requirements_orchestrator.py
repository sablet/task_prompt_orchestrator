"""Requirements orchestrator - Loop B implementation."""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .orchestrator import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RESET,
    YELLOW,
    HistoryManager,
    OrchestratorConfig,
    default_stream_callback,
    delete_history_file,
    list_history_files,
    load_history_from_file,
    run_claude_query,
    run_orchestrator,
    save_history_to_file,
)
from .prompts import (
    build_single_requirement_verification_prompt,
    build_task_generation_prompt,
    check_coverage,
    get_unmet_requirement_ids,
    parse_generated_tasks,
    parse_verification_result,
)
from .schema import (
    LoopBExecutionHistory,
    LoopBIteration,
    LoopBStatus,
    OrchestratorResult,
    Requirement,
    RequirementDefinition,
    Task,
    TaskDefinition,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)

LOOPB_HISTORY_DIR = ".task-orchestrator-history/loopb"


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file content."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass
class RequirementsOrchestratorConfig:
    """Configuration for Loop B orchestrator."""

    max_iterations: int = 3
    tasks_output_dir: str | None = None
    orchestrator_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config for history storage."""
        return {
            "max_iterations": self.max_iterations,
            "tasks_output_dir": self.tasks_output_dir,
            "orchestrator_config": self.orchestrator_config.to_dict(),
        }


class StepModeStop(Exception):
    """Raised when step mode stops Loop B execution."""

    def __init__(self, phase: str):
        self.phase = phase
        super().__init__(f"Step mode: stopped after {phase}")


class LoopBHistoryManager:
    """Manages Loop B execution history persistence."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.history_dir = self.base_dir / LOOPB_HISTORY_DIR

    @staticmethod
    def _generate_history_id(requirements_path: str) -> str:
        req_name = Path(requirements_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{req_name}_{timestamp}"

    def create_history(
        self,
        requirements_path: str,
        config: RequirementsOrchestratorConfig,
    ) -> LoopBExecutionHistory:
        """Create a new Loop B execution history."""
        now = datetime.now().isoformat()

        tasks_output_dir = config.tasks_output_dir
        if not tasks_output_dir:
            tasks_output_dir = str(self.history_dir / "tasks")

        resolved_path = str(Path(requirements_path).resolve())
        requirements_hash = compute_file_hash(resolved_path)

        history = LoopBExecutionHistory(
            history_id=self._generate_history_id(requirements_path),
            requirements_path=resolved_path,
            tasks_output_dir=tasks_output_dir,
            started_at=now,
            updated_at=now,
            status=LoopBStatus.GENERATING_TASKS,
            max_iterations=config.max_iterations,
            current_iteration=0,
            iterations=[],
            completed_task_ids=[],
            loop_c_history_ids=[],
            requirements_hash=requirements_hash,
        )
        self.save_history(history)
        return history

    def save_history(self, history: LoopBExecutionHistory) -> None:
        """Save execution history to file."""
        history.updated_at = datetime.now().isoformat()
        save_history_to_file(self.history_dir, history.history_id, history.to_dict())

    def load_history(self, history_id: str) -> LoopBExecutionHistory:
        """Load execution history from file."""
        data = load_history_from_file(self.history_dir, history_id)
        return LoopBExecutionHistory.from_dict(data)

    def list_histories(self) -> list[LoopBExecutionHistory]:
        """List all Loop B execution histories, sorted by date descending."""
        data_list = list_history_files(self.history_dir)
        histories = [LoopBExecutionHistory.from_dict(d) for d in data_list]
        histories.sort(key=lambda h: h.started_at, reverse=True)
        return histories

    def list_incomplete_histories(self) -> list[LoopBExecutionHistory]:
        """List only incomplete (resumable) Loop B histories."""
        return [
            h
            for h in self.list_histories()
            if h.status not in {LoopBStatus.COMPLETED, LoopBStatus.FAILED}
        ]

    def find_incomplete_by_path(
        self, requirements_path: str
    ) -> LoopBExecutionHistory | None:
        """Find incomplete history for a given requirements file path."""
        resolved_path = str(Path(requirements_path).resolve())
        for h in self.list_incomplete_histories():
            if h.requirements_path == resolved_path:
                return h
        return None

    def find_history_by_path(
        self, requirements_path: str
    ) -> LoopBExecutionHistory | None:
        """Find any history (including completed) for a given requirements file path.

        Returns the most recent history matching the path, regardless of status.
        """
        resolved_path = str(Path(requirements_path).resolve())
        for h in self.list_histories():  # Already sorted by date descending
            if h.requirements_path == resolved_path:
                return h
        return None

    def delete_history(self, history_id: str) -> bool:
        """Delete a history file."""
        return delete_history_file(self.history_dir, history_id)


class RequirementsOrchestrator:
    """Orchestrates Loop B (requirements fulfillment loop)."""

    def __init__(
        self,
        requirements: RequirementDefinition,
        config: RequirementsOrchestratorConfig | None = None,
        history_manager: LoopBHistoryManager | None = None,
        requirements_path: str | None = None,
    ):
        self.requirements = requirements
        self.config = config or RequirementsOrchestratorConfig()
        self.history_manager = history_manager
        self.requirements_path = requirements_path or ""
        self.history: LoopBExecutionHistory | None = None
        self.all_completed_tasks: list[Task] = []
        self.all_task_results: list[TaskResult] = []
        self.loop_c_history_manager: HistoryManager | None = None

    async def run(self) -> LoopBExecutionHistory:
        """Execute the Loop B orchestration from the beginning."""
        self._initialize_new()
        return await self._run_loop()

    async def resume(
        self, existing_history: LoopBExecutionHistory
    ) -> LoopBExecutionHistory:
        """Resume Loop B orchestration from existing history."""
        self._initialize_from_history(existing_history)
        return await self._run_loop()

    def _get_callback(self):
        """Get stream callback from config."""
        return (
            self.config.orchestrator_config.stream_callback or default_stream_callback
        )

    def _print_loop_header(self, iteration: int) -> None:
        """Print Loop B iteration header."""
        callback = self._get_callback()
        if not self.config.orchestrator_config.stream_output:
            return
        max_iter = self.config.max_iterations
        num_reqs = len(self.requirements.requirements)
        callback(f"\n{BOLD}{'━' * 60}{RESET}\n")
        callback(
            f"{BOLD}{CYAN}🔄 LOOP B{RESET} - Iteration [{iteration + 1}/{max_iter}]\n"
        )
        callback(
            f"{DIM}   Requirements: {num_reqs} | Max iterations: {max_iter}{RESET}\n"
        )
        callback(f"{BOLD}{'━' * 60}{RESET}\n")

    def _print_step(self, step_name: str, description: str = "") -> None:
        """Print current step in Loop B."""
        callback = self._get_callback()
        if not self.config.orchestrator_config.stream_output:
            return
        desc_str = f" - {description}" if description else ""
        callback(f"{YELLOW}▶ [LOOP B] {step_name}{desc_str}{RESET}\n")

    def _print_loop_c_start(self, iteration: int, num_tasks: int) -> None:
        """Print Loop C execution start."""
        callback = self._get_callback()
        if not self.config.orchestrator_config.stream_output:
            return
        callback(f"\n{BOLD}{'-' * 60}{RESET}\n")
        callback(f"{BOLD}{GREEN}📋 LOOP C{RESET} - Starting task execution\n")
        callback(f"{DIM}   Iteration: {iteration + 1} | Tasks: {num_tasks}{RESET}\n")
        callback(f"{BOLD}{'-' * 60}{RESET}\n")

    async def _run_loop(self) -> LoopBExecutionHistory:
        """Execute the main Loop B iteration loop."""
        assert self.history is not None

        # Determine where to resume from based on status and current_iteration
        resume_info = self._get_resume_info()
        start_iteration = resume_info["start_iteration"]
        skip_task_generation = resume_info["skip_task_generation"]
        resume_loop_c = resume_info["resume_loop_c"]
        skip_initial_verification = resume_info.get("skip_initial_verification", False)
        resume_verification_index = resume_info.get("resume_verification_index", 0)
        partial_verification_results = resume_info.get("partial_verification_results")

        try:
            for iteration in range(start_iteration, self.config.max_iterations):
                self._print_loop_header(iteration)
                # Only skip on the first iteration of resume
                do_skip_task_gen = skip_task_generation and iteration == start_iteration
                do_skip_initial_verify = (
                    skip_initial_verification and iteration == start_iteration
                )
                # Resume verification only on first iteration
                do_resume_verification = (
                    resume_verification_index > 0 and iteration == start_iteration
                )

                # Step 1: Pre-verification - check which requirements need work
                if not do_skip_initial_verify:
                    if do_resume_verification:
                        unmet_requirements = await self._pre_verify_requirements(
                            iteration,
                            resume_index=resume_verification_index,
                            partial_results=partial_verification_results,
                        )
                    else:
                        unmet_requirements = await self._pre_verify_requirements(
                            iteration
                        )
                    if not unmet_requirements:
                        # All requirements already met
                        logger.info("Loop B: All requirements already met")
                        self._complete({"all_requirements_met": True})
                        break
                else:
                    # Use previous iteration's unmet requirements
                    unmet_requirements = self._get_previous_unmet_requirements()

                # Reset verification resume state after first iteration
                resume_verification_index = 0
                partial_verification_results = None

                # Step 2: Get or generate tasks (for unmet requirements only)
                tasks, tasks_yaml_path = await self._get_or_generate_tasks(
                    iteration, do_skip_task_gen, unmet_requirements
                )
                if tasks is None:
                    break  # _fail already called

                # Step 3: Execute Loop C
                should_resume_loop_c = resume_loop_c and iteration == start_iteration
                loop_c_result, loop_c_history_id = await self._run_loop_c_step(
                    iteration,
                    tasks_yaml_path,
                    should_resume_loop_c,
                    num_tasks=len(tasks),
                )

                # Update completed tasks
                self._update_completed_tasks(tasks, loop_c_result)

                # Check if Loop C stopped due to step mode
                if loop_c_result.step_stopped:
                    logger.info("Step mode: Loop C stopped, pausing Loop B")
                    self._save_history()
                    return self.history

                # Update current_iteration after Loop C completes (if we skipped earlier)
                if do_skip_task_gen:
                    self.history.current_iteration = iteration + 1
                    self._save_history()

                # Step 4: Record iteration (verification done at start of next iteration)
                self._record_iteration(
                    iteration,
                    tasks_yaml_path,
                    loop_c_history_id,
                    verification_result=None,
                    unmet_requirements=[r.id for r in unmet_requirements],
                )

                # Check max iterations
                if iteration + 1 >= self.config.max_iterations:
                    self._fail(f"Max iterations ({self.config.max_iterations}) reached")
                    break

        except StepModeStop as e:
            logger.info(f"Step mode: stopped after {e.phase}")
            self._save_history()
            return self.history

        return self.history

    async def _get_or_generate_tasks(
        self,
        iteration: int,
        skip_generation: bool,
        unmet_requirements: list[Requirement],
    ) -> tuple[list[Task] | None, str]:
        """Get existing tasks or generate new ones for the iteration.

        Args:
            iteration: Current iteration number (0-indexed)
            skip_generation: Whether to skip generation and load existing tasks
            unmet_requirements: Requirements that are not yet met

        Returns:
            Tuple of (tasks, tasks_yaml_path). tasks is None if generation failed.
        """
        assert self.history is not None

        if not skip_generation:
            return await self._generate_new_tasks(iteration, unmet_requirements)

        # Resume: try to find existing tasks
        return await self._load_or_regenerate_tasks(iteration, unmet_requirements)

    async def _generate_new_tasks(
        self, iteration: int, unmet_requirements: list[Requirement]
    ) -> tuple[list[Task] | None, str]:
        """Generate new tasks for the iteration.

        Args:
            iteration: Current iteration number (0-indexed)
            unmet_requirements: Requirements that are not yet met
        """
        assert self.history is not None

        self.history.current_iteration = iteration + 1
        self._save_history()

        logger.info(
            f"Loop B iteration {iteration + 1}: Generating tasks for "
            f"{len(unmet_requirements)} unmet requirements"
        )
        self._print_step(
            "TASK GENERATION",
            f"Generating tasks for {len(unmet_requirements)} unmet requirements",
        )
        self.history.status = LoopBStatus.GENERATING_TASKS
        self._save_history()

        tasks = await self._generate_tasks_with_coverage_warning(
            iteration, unmet_requirements
        )
        if not tasks:
            self._fail("Failed to generate tasks")
            return None, ""

        tasks_yaml_path = self._write_tasks_yaml(tasks, iteration)
        return tasks, tasks_yaml_path

    async def _load_or_regenerate_tasks(
        self, iteration: int, unmet_requirements: list[Requirement]
    ) -> tuple[list[Task] | None, str]:
        """Load existing tasks or regenerate if not found.

        Args:
            iteration: Current iteration number (0-indexed)
            unmet_requirements: Requirements that are not yet met
        """
        assert self.history is not None

        # First check if there's an iteration record for this iteration
        existing_iter = next(
            (
                it
                for it in self.history.iterations
                if it.iteration_number == iteration + 1
            ),
            None,
        )
        if existing_iter and Path(existing_iter.tasks_yaml_path).exists():
            tasks_yaml_path = existing_iter.tasks_yaml_path
            task_def = TaskDefinition.from_yaml(tasks_yaml_path)
            logger.info(
                f"Loop B iteration {iteration + 1}: Resuming with existing tasks from {tasks_yaml_path}"
            )
            return task_def.tasks, tasks_yaml_path

        # Try to find tasks file by expected name (may exist even without iteration record)
        expected_path = self._get_expected_tasks_path(iteration)
        if expected_path and Path(expected_path).exists():
            task_def = TaskDefinition.from_yaml(expected_path)
            logger.info(
                f"Loop B iteration {iteration + 1}: Resuming with tasks from {expected_path}"
            )
            return task_def.tasks, expected_path

        # Fallback: regenerate tasks if file missing
        logger.warning("Cannot resume: tasks file missing, regenerating")
        return await self._generate_new_tasks(iteration, unmet_requirements)

    async def _run_loop_c_step(
        self, iteration: int, tasks_yaml_path: str, resume: bool, num_tasks: int = 0
    ) -> tuple[OrchestratorResult, str | None]:
        """Execute Loop C step."""
        assert self.history is not None

        logger.info(f"Loop B iteration {iteration + 1}: Executing Loop C")
        self._print_loop_c_start(iteration, num_tasks)
        self.history.status = LoopBStatus.EXECUTING_TASKS
        self._save_history()

        return await self._execute_loop_c(tasks_yaml_path, resume=resume)

    async def _pre_verify_requirements(
        self,
        iteration: int,
        resume_index: int = 0,
        partial_results: list[dict[str, Any]] | None = None,
    ) -> list[Requirement]:
        """Verify all requirements and return unmet ones.

        This is called at the start of each iteration to determine which
        requirements need tasks generated.

        Args:
            iteration: Current iteration number (0-indexed)
            resume_index: Index of requirement to resume from (for step mode resume)
            partial_results: Previously completed verification results (for resume)

        Returns:
            List of Requirement objects that are not yet met

        Raises:
            StepModeStop: If step mode is enabled and a step was executed.
        """
        assert self.history is not None

        if resume_index > 0:
            logger.info(
                f"Loop B iteration {iteration + 1}: Resuming verification from requirement {resume_index + 1}"
            )
            self._print_step(
                "PRE-VERIFICATION",
                f"Resuming from requirement {resume_index + 1}/{len(self.requirements.requirements)}",
            )
        else:
            logger.info(f"Loop B iteration {iteration + 1}: Pre-verifying requirements")
            self._print_step(
                "PRE-VERIFICATION", "Checking which requirements need work"
            )

        self.history.status = LoopBStatus.VERIFYING_REQUIREMENTS
        self._save_history()

        verification = await self._verify_requirements(resume_index, partial_results)

        # Store verification result for reference
        if self.history.iterations:
            # Update the last iteration's verification result
            self.history.iterations[-1].verification_result = verification
            self._save_history()

        # Get unmet requirement IDs and map to Requirement objects
        unmet_ids = get_unmet_requirement_ids(verification)
        unmet_requirements = [
            req for req in self.requirements.requirements if req.id in unmet_ids
        ]

        callback = self._get_callback()
        config = self.config.orchestrator_config
        if config.stream_output:
            met_count = len(self.requirements.requirements) - len(unmet_requirements)
            callback(
                f"\n{DIM}Pre-verification: {met_count} met, "
                f"{len(unmet_requirements)} need work{RESET}\n"
            )

        return unmet_requirements

    def _get_previous_unmet_requirements(self) -> list[Requirement]:
        """Get unmet requirements from the previous iteration.

        Used when resuming and skipping initial verification.
        """
        assert self.history is not None

        if not self.history.iterations:
            # No previous iteration, return all requirements
            return list(self.requirements.requirements)

        last_iter = self.history.iterations[-1]
        unmet_ids = set(last_iter.unmet_requirements)

        return [req for req in self.requirements.requirements if req.id in unmet_ids]

    def _record_iteration(
        self,
        iteration: int,
        tasks_yaml_path: str,
        loop_c_history_id: str | None,
        verification_result: dict[str, Any] | None,
        unmet_requirements: list[str],
    ) -> None:
        """Record or update iteration in history."""
        assert self.history is not None

        existing_iter_idx = next(
            (
                i
                for i, it in enumerate(self.history.iterations)
                if it.iteration_number == iteration + 1
            ),
            None,
        )
        iteration_record = LoopBIteration(
            iteration_number=iteration + 1,
            tasks_yaml_path=tasks_yaml_path,
            loop_c_history_id=loop_c_history_id,
            verification_result=verification_result,
            unmet_requirements=unmet_requirements,
        )
        if existing_iter_idx is not None:
            self.history.iterations[existing_iter_idx] = iteration_record
        else:
            self.history.iterations.append(iteration_record)
        if (
            loop_c_history_id
            and loop_c_history_id not in self.history.loop_c_history_ids
        ):
            self.history.loop_c_history_ids.append(loop_c_history_id)
        self._save_history()

    def _get_resume_info(self) -> dict[str, Any]:
        """Determine resume point based on status and history state."""
        assert self.history is not None

        status = self.history.status
        current_iter = self.history.current_iteration

        # Default: start fresh from iteration 0
        if current_iter == 0:
            return {
                "start_iteration": 0,
                "skip_task_generation": False,
                "skip_initial_verification": False,
                "resume_loop_c": False,
            }

        # If we were executing tasks (Loop C), resume that iteration
        if status == LoopBStatus.EXECUTING_TASKS:
            return {
                "start_iteration": current_iter - 1,  # 0-indexed
                "skip_task_generation": True,  # Tasks already generated
                "skip_initial_verification": True,  # Verification already done
                "resume_loop_c": True,  # Resume Loop C from where it stopped
            }

        # If we were generating tasks, restart task generation for that iteration
        # (verification was done, but task generation failed/interrupted)
        if status == LoopBStatus.GENERATING_TASKS:
            return {
                "start_iteration": current_iter - 1,
                "skip_task_generation": False,
                "skip_initial_verification": True,  # Verification already done
                "resume_loop_c": False,
            }

        # If verifying, resume verification from where it stopped
        if status == LoopBStatus.VERIFYING_REQUIREMENTS:
            return {
                "start_iteration": current_iter - 1,
                "skip_task_generation": False,
                "skip_initial_verification": False,
                "resume_loop_c": False,
                "resume_verification_index": self.history.current_verification_index,
                "partial_verification_results": self.history.partial_verification_results,
            }

        # Completed or failed - shouldn't normally reach here
        return {
            "start_iteration": current_iter,
            "skip_task_generation": False,
            "skip_initial_verification": False,
            "resume_loop_c": False,
        }

    def _initialize_new(self) -> None:
        """Initialize new history and state for fresh run."""
        if self.history_manager and self.requirements_path:
            self.history = self.history_manager.create_history(
                self.requirements_path, self.config
            )
            self.loop_c_history_manager = HistoryManager(
                str(self.history_manager.base_dir)
            )
        else:
            now = datetime.now().isoformat()
            tasks_output_dir = self.config.tasks_output_dir or "generated"
            self.history = LoopBExecutionHistory(
                history_id=f"inmemory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                requirements_path=self.requirements_path,
                tasks_output_dir=tasks_output_dir,
                started_at=now,
                updated_at=now,
                status=LoopBStatus.GENERATING_TASKS,
                max_iterations=self.config.max_iterations,
                current_iteration=0,
                iterations=[],
                completed_task_ids=[],
                loop_c_history_ids=[],
            )

    def _initialize_from_history(self, existing_history: LoopBExecutionHistory) -> None:
        """Initialize state from existing history for resume."""
        self.history = existing_history

        # Clear error if resuming from failed state
        if self.history.status == LoopBStatus.FAILED:
            self.history.error = None

        # Initialize Loop C history manager
        if self.history_manager:
            self.loop_c_history_manager = HistoryManager(
                str(self.history_manager.base_dir)
            )

        # Restore completed tasks from previous iterations
        self._restore_completed_tasks_from_history()

        logger.info(
            f"Resumed from iteration {self.history.current_iteration}, "
            f"{len(self.all_completed_tasks)} completed tasks restored"
        )

    def _restore_completed_tasks_from_history(self) -> None:
        """Restore completed tasks and results from history iterations."""
        assert self.history is not None

        for iteration in self.history.iterations:
            tasks_yaml_path = iteration.tasks_yaml_path
            if not Path(tasks_yaml_path).exists():
                logger.warning(f"Tasks YAML not found: {tasks_yaml_path}")
                continue

            task_def = TaskDefinition.from_yaml(tasks_yaml_path)
            task_map = {t.id: t for t in task_def.tasks}

            # Restore completed tasks
            for task_id in self.history.completed_task_ids:
                if task_id in task_map:
                    task = task_map[task_id]
                    if task not in self.all_completed_tasks:
                        self.all_completed_tasks.append(task)

            # Create placeholder TaskResults for approved tasks
            for task_id in self.history.completed_task_ids:
                if not any(r.task_id == task_id for r in self.all_task_results):
                    self.all_task_results.append(
                        TaskResult(
                            task_id=task_id,
                            status=TaskStatus.APPROVED,
                            validation_approved=True,
                        )
                    )

    def _save_history(self) -> None:
        """Save history if manager is available."""
        if self.history_manager and self.history:
            self.history_manager.save_history(self.history)

    def _fail(self, error: str) -> None:
        """Mark execution as failed."""
        if self.history:
            self.history.status = LoopBStatus.FAILED
            self.history.error = error
            self._save_history()

    def _complete(self, final_result: dict[str, Any]) -> None:
        """Mark execution as completed."""
        if self.history:
            self.history.status = LoopBStatus.COMPLETED
            self.history.final_result = final_result
            self._save_history()

    async def _generate_tasks_with_coverage_warning(
        self, iteration: int, unmet_requirements: list[Requirement]
    ) -> list[Task]:
        """Generate tasks and warn if coverage is incomplete.

        Coverage check is informational only - requirement verification loop
        will catch any missing requirements in subsequent iterations.
        """
        tasks = await self._generate_tasks(iteration, unmet_requirements)
        if not tasks:
            return []

        # Create a temporary RequirementDefinition for coverage check
        unmet_req_def = RequirementDefinition(
            requirements=unmet_requirements,
            common_validation=self.requirements.common_validation,
        )
        all_covered, uncovered = check_coverage(unmet_req_def, tasks)
        if not all_covered:
            logger.warning(
                f"Coverage incomplete: {len(uncovered)} criteria not explicitly covered "
                f"by task validations: {uncovered}. "
                "These may be addressed in subsequent iterations via requirement verification."
            )

        return tasks

    async def _generate_tasks(
        self, iteration: int, unmet_requirements: list[Requirement]
    ) -> list[Task]:
        """Generate tasks using LLM.

        Args:
            iteration: Current iteration number (0-indexed)
            unmet_requirements: Requirements that need tasks generated

        Raises:
            StepModeStop: If step mode is enabled and a step was executed.
        """
        assert self.history is not None

        # Get previous feedback if available
        previous_feedback = None
        if self.history.iterations:
            previous_verification = self.history.iterations[-1].verification_result
            if previous_verification:
                previous_feedback = previous_verification.get(
                    "feedback_for_additional_tasks", ""
                )

        prompt = build_task_generation_prompt(
            unmet_requirements=unmet_requirements,
            all_requirements=list(self.requirements.requirements),
            common_validation=self.requirements.common_validation,
            completed_tasks=self.all_completed_tasks if iteration > 0 else None,
            previous_feedback=previous_feedback,
        )

        config = self.config.orchestrator_config
        callback = self._get_callback()
        if config.stream_output:
            callback(f"\n{BOLD}{'=' * 60}{RESET}\n")
            callback(f"{DIM}>>> PROMPT >>>{RESET}\n")
            callback(f"{DIM}{prompt}{RESET}\n")
            callback(f"{DIM}<<< END PROMPT <<<{RESET}\n\n")
        output = await run_claude_query(prompt, config, phase="task_generation")
        tasks = parse_generated_tasks(output)

        # Step mode check after task generation
        if config.step_mode and config._step_executed:
            # Write tasks before stopping so they can be resumed
            if tasks:
                self._write_tasks_yaml(tasks, iteration)
            raise StepModeStop("task_generation")

        return tasks

    def _write_tasks_yaml(self, tasks: list[Task], iteration: int) -> str:
        """Write generated tasks to YAML file."""
        assert self.history is not None
        output_dir = Path(self.history.tasks_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        req_name = (
            Path(self.requirements_path).stem if self.requirements_path else "tasks"
        )
        filename = f"{req_name}-tasks-iter{iteration + 1}.yaml"
        filepath = output_dir / filename

        tasks_data: dict[str, Any] = {
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "instruction": t.instruction,
                    "validation": t.validation,
                    "depends_on": t.depends_on,
                }
                for t in tasks
            ]
        }
        # Include common_validation from requirements
        if self.requirements.common_validation:
            tasks_data["common_validation"] = self.requirements.common_validation

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                tasks_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(f"Generated tasks written to: {filepath}")
        return str(filepath)

    def _get_expected_tasks_path(self, iteration: int) -> str | None:
        """Get the expected tasks YAML path for a given iteration.

        This is used when resuming to find the tasks file even if the
        iteration record hasn't been created yet.
        """
        if not self.history:
            return None

        output_dir = Path(self.history.tasks_output_dir)
        req_name = (
            Path(self.requirements_path).stem if self.requirements_path else "tasks"
        )
        filename = f"{req_name}-tasks-iter{iteration + 1}.yaml"
        return str(output_dir / filename)

    async def _execute_loop_c(
        self, tasks_yaml_path: str, resume: bool = False
    ) -> tuple[OrchestratorResult, str | None]:
        """Execute Loop C with the generated tasks.

        Args:
            tasks_yaml_path: Path to the tasks YAML file
            resume: If True, try to resume from incomplete Loop C history

        Returns:
            Tuple of (OrchestratorResult, loop_c_history_id)
        """
        task_definition = TaskDefinition.from_yaml(tasks_yaml_path)
        resume_history = None
        resume_point = None

        # Try to find incomplete Loop C history for this tasks file
        if resume and self.loop_c_history_manager:
            resolved_path = str(Path(tasks_yaml_path).resolve())
            for history in self.loop_c_history_manager.list_incomplete_histories():
                if history.yaml_path == resolved_path:
                    resume_history = history
                    # Determine resume point from history
                    if history.current_task_id and history.current_phase:
                        from .schema import ResumePoint, ExecutionPhase

                        resume_point = ResumePoint(
                            task_id=history.current_task_id,
                            phase=ExecutionPhase(history.current_phase),
                        )
                    logger.info(
                        f"Resuming Loop C from history: {history.history_id}, "
                        f"task: {history.current_task_id}, phase: {history.current_phase}"
                    )
                    break

        result = await run_orchestrator(
            task_definition,
            config=self.config.orchestrator_config,
            yaml_path=tasks_yaml_path,
            history_manager=self.loop_c_history_manager,
            resume_point=resume_point,
            resume_history=resume_history,
        )

        # Get the history ID from the most recent history or resumed history
        loop_c_history_id = None
        if resume_history:
            loop_c_history_id = resume_history.history_id
        elif self.loop_c_history_manager:
            histories = self.loop_c_history_manager.list_histories()
            if histories:
                loop_c_history_id = histories[0].history_id

        return result, loop_c_history_id

    def _update_completed_tasks(
        self, tasks: list[Task], result: OrchestratorResult
    ) -> None:
        """Update the list of completed tasks and results."""
        assert self.history is not None
        task_map = {t.id: t for t in tasks}

        for task_result in result.task_results:
            if task_result.status == TaskStatus.APPROVED:
                task_id = task_result.task_id
                if (
                    task_id in task_map
                    and task_id not in self.history.completed_task_ids
                ):
                    self.all_completed_tasks.append(task_map[task_id])
                    self.history.completed_task_ids.append(task_id)
            self.all_task_results.append(task_result)

    async def _verify_single_requirement(
        self, req: "Requirement", req_index: int, total_reqs: int
    ) -> dict[str, Any]:
        """Verify a single requirement by actually executing verification steps.

        Args:
            req: The requirement to verify
            req_index: 1-indexed position of this requirement
            total_reqs: Total number of requirements

        Returns:
            Verification result for this requirement
        """

        prompt = build_single_requirement_verification_prompt(
            req, list(self.requirements.requirements)
        )
        config = self.config.orchestrator_config
        callback = self._get_callback()

        if config.stream_output:
            callback(f"\n{BOLD}{'=' * 60}{RESET}\n")
            callback(
                f"{BOLD}🔍 VERIFICATION [{req_index}/{total_reqs}]: {req.id}{RESET}\n"
            )
            callback(f"{DIM}   {req.name}{RESET}\n")
            callback(f"{BOLD}{'=' * 60}{RESET}\n")
            callback(f"{DIM}>>> PROMPT >>>{RESET}\n")
            callback(f"{DIM}{prompt}{RESET}\n")
            callback(f"{DIM}<<< END PROMPT <<<{RESET}\n\n")

        output = await run_claude_query(prompt, config, phase="verification")
        result = parse_verification_result(output)

        # Ensure requirement_id is set
        if "requirement_id" not in result:
            result["requirement_id"] = req.id

        return result

    async def _verify_requirements(
        self, resume_index: int = 0, partial_results: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Verify if requirements are met by checking each requirement individually.

        Args:
            resume_index: Index of requirement to resume from (for step mode resume)
            partial_results: Previously completed verification results (for resume)

        Raises:
            StepModeStop: If step mode is enabled and a step was executed.
        """
        assert self.history is not None

        config = self.config.orchestrator_config
        callback = self._get_callback()
        total_reqs = len(self.requirements.requirements)

        if config.stream_output:
            callback(f"\n{BOLD}{'-' * 60}{RESET}\n")
            callback(f"{BOLD}{CYAN}🔎 REQUIREMENT VERIFICATION{RESET}\n")
            if resume_index > 0:
                callback(
                    f"{DIM}   Resuming from requirement {resume_index + 1}/{total_reqs}{RESET}\n"
                )
            else:
                callback(f"{DIM}   Total requirements: {total_reqs}{RESET}\n")
            callback(f"{BOLD}{'-' * 60}{RESET}\n")

        # Initialize or restore partial results
        requirement_statuses: list[dict[str, Any]] = list(partial_results or [])
        all_met = all(s.get("met", False) for s in requirement_statuses)

        # Verify each requirement individually (starting from resume_index)
        for idx, req in enumerate(self.requirements.requirements):
            if idx < resume_index:
                continue  # Skip already verified requirements

            # Update history with current verification progress
            self.history.current_verification_index = idx
            self.history.partial_verification_results = requirement_statuses
            self._save_history()

            result = await self._verify_single_requirement(req, idx + 1, total_reqs)
            met = result.get("met", False)
            if not met:
                all_met = False

            status_record = {
                "requirement_id": req.id,
                "met": met,
                "evidence": result.get("summary", ""),
                "criteria_results": result.get("criteria_results", []),
                "issues": result.get("issues", []),
            }
            requirement_statuses.append(status_record)

            if config.stream_output:
                status_icon = f"{GREEN}✓{RESET}" if met else f"{YELLOW}✗{RESET}"
                callback(f"\n  {status_icon} {req.id}: {'met' if met else 'not met'}\n")

            # Step mode check after each verification
            if config.step_mode and config._step_executed:
                # Save progress before stopping
                self.history.current_verification_index = idx + 1
                self.history.partial_verification_results = requirement_statuses
                self._save_history()
                raise StepModeStop("verification")

        # Reset verification progress after completion
        self.history.current_verification_index = 0
        self.history.partial_verification_results = None
        self._save_history()

        # Aggregate results
        unmet_count = sum(1 for s in requirement_statuses if not s["met"])
        summary = (
            "All requirements met"
            if all_met
            else f"{unmet_count} of {len(requirement_statuses)} requirements not met"
        )

        # Build feedback for additional tasks if needed
        feedback_parts = []
        for status in requirement_statuses:
            if not status["met"] and status.get("issues"):
                feedback_parts.append(
                    f"{status['requirement_id']}: {', '.join(status['issues'])}"
                )

        if config.stream_output:
            callback(f"\n{BOLD}{'-' * 60}{RESET}\n")
            callback(f"{CYAN}Verification Summary: {summary}{RESET}\n")
            callback(f"{BOLD}{'-' * 60}{RESET}\n")

        return {
            "all_requirements_met": all_met,
            "requirement_status": requirement_statuses,
            "summary": summary,
            "feedback_for_additional_tasks": "\n".join(feedback_parts),
        }


async def run_requirements_orchestrator(
    requirements_path: str,
    config: RequirementsOrchestratorConfig | None = None,
    base_dir: str | None = None,
) -> LoopBExecutionHistory:
    """Convenience function to run Loop B orchestration.

    Args:
        requirements_path: Path to requirements YAML file
        config: Orchestrator configuration
        base_dir: Base directory for history storage

    Returns:
        Loop B execution history
    """
    requirements = RequirementDefinition.from_yaml(requirements_path)
    config = config or RequirementsOrchestratorConfig()

    history_manager = LoopBHistoryManager(base_dir)

    orchestrator = RequirementsOrchestrator(
        requirements=requirements,
        config=config,
        history_manager=history_manager,
        requirements_path=requirements_path,
    )

    return await orchestrator.run()
