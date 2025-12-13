"""Loop B task generation logic - task generation and Loop C execution mixin."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .loopb_history import StepModeStop
from .orchestrator import BOLD, DIM, GREEN, RESET, run_claude_query, run_orchestrator
from .prompts import build_task_generation_prompt, check_coverage, parse_generated_tasks
from .schema import (
    ExecutionPhase,
    LoopBStatus,
    OrchestratorResult,
    RequirementDefinition,
    ResumePoint,
    Task,
    TaskDefinition,
    TaskResult,
    TaskStatus,
)

if TYPE_CHECKING:
    from .loopb_history import RequirementsOrchestratorConfig
    from .orchestrator import HistoryManager
    from .schema import LoopBExecutionHistory, Requirement

logger = logging.getLogger(__name__)


class TaskGenerationMixin:
    """Mixin providing task generation and Loop C execution functionality."""

    # Type hints for attributes from RequirementsOrchestrator
    history: "LoopBExecutionHistory | None"
    requirements: "RequirementDefinition"
    config: "RequirementsOrchestratorConfig"
    requirements_path: str
    all_completed_tasks: list[Task]
    all_task_results: list[TaskResult]
    loop_c_history_manager: "HistoryManager | None"

    def _get_callback(self) -> Any:
        """Get stream callback."""
        ...

    def _save_history(self) -> None:
        """Save history."""
        ...

    def _fail(self, error: str) -> None:
        """Mark execution as failed."""
        ...

    def _print_step(self, step_name: str, description: str = "") -> None:
        """Print step."""
        ...

    def _print_loop_c_start(self, iteration: int, num_tasks: int) -> None:
        """Print Loop C start."""
        callback = self._get_callback()
        if not self.config.orchestrator_config.stream_output:
            return
        callback(f"\n{BOLD}{'-' * 60}{RESET}\n")
        callback(f"{BOLD}{GREEN}📋 LOOP C{RESET} - Starting task execution\n")
        callback(f"{DIM}   Iteration: {iteration + 1} | Tasks: {num_tasks}{RESET}\n")
        callback(f"{BOLD}{'-' * 60}{RESET}\n")

    async def _get_or_generate_tasks(
        self,
        iteration: int,
        skip_generation: bool,
        unmet_requirements: list["Requirement"],
    ) -> tuple[list[Task] | None, str]:
        """Get existing tasks or generate new ones for the iteration."""
        if not skip_generation:
            return await self._generate_new_tasks(iteration, unmet_requirements)
        return await self._load_or_regenerate_tasks(iteration, unmet_requirements)

    async def _generate_new_tasks(
        self, iteration: int, unmet_requirements: list["Requirement"]
    ) -> tuple[list[Task] | None, str]:
        """Generate new tasks for the iteration."""
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
        self, iteration: int, unmet_requirements: list["Requirement"]
    ) -> tuple[list[Task] | None, str]:
        """Load existing tasks or regenerate if not found."""
        assert self.history is not None

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

        expected_path = self._get_expected_tasks_path(iteration)
        if expected_path and Path(expected_path).exists():
            task_def = TaskDefinition.from_yaml(expected_path)
            logger.info(
                f"Loop B iteration {iteration + 1}: Resuming with tasks from {expected_path}"
            )
            return task_def.tasks, expected_path

        logger.warning("Cannot resume: tasks file missing, regenerating")
        return await self._generate_new_tasks(iteration, unmet_requirements)

    async def _generate_tasks_with_coverage_warning(
        self, iteration: int, unmet_requirements: list["Requirement"]
    ) -> list[Task]:
        """Generate tasks and warn if coverage is incomplete."""
        tasks = await self._generate_tasks(iteration, unmet_requirements)
        if not tasks:
            return []

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
        self, iteration: int, unmet_requirements: list["Requirement"]
    ) -> list[Task]:
        """Generate tasks using LLM."""
        assert self.history is not None

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

        if config.step_mode and config._step_executed:
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
        """Get the expected tasks YAML path for a given iteration."""
        if not self.history:
            return None

        output_dir = Path(self.history.tasks_output_dir)
        req_name = (
            Path(self.requirements_path).stem if self.requirements_path else "tasks"
        )
        filename = f"{req_name}-tasks-iter{iteration + 1}.yaml"
        return str(output_dir / filename)

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

    async def _execute_loop_c(
        self, tasks_yaml_path: str, resume: bool = False
    ) -> tuple[OrchestratorResult, str | None]:
        """Execute Loop C with the generated tasks."""
        task_definition = TaskDefinition.from_yaml(tasks_yaml_path)
        resume_history = None
        resume_point = None

        if resume and self.loop_c_history_manager:
            resolved_path = str(Path(tasks_yaml_path).resolve())
            for history in self.loop_c_history_manager.list_incomplete_histories():
                if history.yaml_path == resolved_path:
                    resume_history = history
                    if history.current_task_id and history.current_phase:
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
