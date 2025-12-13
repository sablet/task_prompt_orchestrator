"""Requirements orchestrator - Loop B implementation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .loopb_history import (
    LoopBHistoryManager,
    RequirementsOrchestratorConfig,
    StepModeStop,
)
from .loopb_tasks import TaskGenerationMixin
from .loopb_verification import VerificationMixin
from .orchestrator import (
    BOLD,
    CYAN,
    DIM,
    RESET,
    YELLOW,
    HistoryManager,
    default_stream_callback,
)
from .schema import (
    LoopBExecutionHistory,
    LoopBIteration,
    LoopBStatus,
    Requirement,
    RequirementDefinition,
    Task,
    TaskDefinition,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class RequirementsOrchestrator(VerificationMixin, TaskGenerationMixin):
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
        if not self.config.orchestrator_config.stream_output:
            return
        callback = self._get_callback()
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
        if not self.config.orchestrator_config.stream_output:
            return
        callback = self._get_callback()
        desc_str = f" - {description}" if description else ""
        callback(f"{YELLOW}▶ [LOOP B] {step_name}{desc_str}{RESET}\n")

    async def _run_loop(self) -> LoopBExecutionHistory:
        """Execute the main Loop B iteration loop."""
        assert self.history is not None

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
                do_skip_task_gen = skip_task_generation and iteration == start_iteration
                do_skip_initial_verify = (
                    skip_initial_verification and iteration == start_iteration
                )
                do_resume_verification = (
                    resume_verification_index > 0 and iteration == start_iteration
                )

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
                        logger.info("Loop B: All requirements already met")
                        self._complete({"all_requirements_met": True})
                        break
                else:
                    unmet_requirements = self._get_previous_unmet_requirements()

                resume_verification_index = 0
                partial_verification_results = None

                tasks, tasks_yaml_path = await self._get_or_generate_tasks(
                    iteration, do_skip_task_gen, unmet_requirements
                )
                if tasks is None:
                    break

                should_resume_loop_c = resume_loop_c and iteration == start_iteration
                loop_c_result, loop_c_history_id = await self._run_loop_c_step(
                    iteration,
                    tasks_yaml_path,
                    should_resume_loop_c,
                    num_tasks=len(tasks),
                )

                self._update_completed_tasks(tasks, loop_c_result)

                if loop_c_result.step_stopped:
                    logger.info("Step mode: Loop C stopped, pausing Loop B")
                    self._save_history()
                    return self.history

                if do_skip_task_gen:
                    self.history.current_iteration = iteration + 1
                    self._save_history()

                self._record_iteration(
                    iteration,
                    tasks_yaml_path,
                    loop_c_history_id,
                    verification_result=None,
                    unmet_requirements=[r.id for r in unmet_requirements],
                )

                if iteration + 1 >= self.config.max_iterations:
                    self._fail(f"Max iterations ({self.config.max_iterations}) reached")
                    break

        except StepModeStop as e:
            logger.info(f"Step mode: stopped after {e.phase}")
            self._save_history()
            return self.history

        return self.history

    def _get_previous_unmet_requirements(self) -> list[Requirement]:
        """Get unmet requirements from the previous iteration."""
        assert self.history is not None
        if not self.history.iterations:
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

        if current_iter == 0:
            return {
                "start_iteration": 0,
                "skip_task_generation": False,
                "skip_initial_verification": False,
                "resume_loop_c": False,
            }

        if status == LoopBStatus.EXECUTING_TASKS:
            return {
                "start_iteration": current_iter - 1,
                "skip_task_generation": True,
                "skip_initial_verification": True,
                "resume_loop_c": True,
            }

        if status == LoopBStatus.GENERATING_TASKS:
            return {
                "start_iteration": current_iter - 1,
                "skip_task_generation": False,
                "skip_initial_verification": True,
                "resume_loop_c": False,
            }

        if status == LoopBStatus.VERIFYING_REQUIREMENTS:
            return {
                "start_iteration": current_iter - 1,
                "skip_task_generation": False,
                "skip_initial_verification": False,
                "resume_loop_c": False,
                "resume_verification_index": self.history.current_verification_index,
                "partial_verification_results": self.history.partial_verification_results,
            }

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
        if self.history.status == LoopBStatus.FAILED:
            self.history.error = None
        if self.history_manager:
            self.loop_c_history_manager = HistoryManager(
                str(self.history_manager.base_dir)
            )
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
            for task_id in self.history.completed_task_ids:
                if task_id in task_map:
                    task = task_map[task_id]
                    if task not in self.all_completed_tasks:
                        self.all_completed_tasks.append(task)
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


async def run_requirements_orchestrator(
    requirements_path: str,
    config: RequirementsOrchestratorConfig | None = None,
    base_dir: str | None = None,
) -> LoopBExecutionHistory:
    """Convenience function to run Loop B orchestration."""
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
