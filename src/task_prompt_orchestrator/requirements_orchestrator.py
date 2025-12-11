"""Requirements orchestrator - Loop B implementation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .orchestrator import (
    HistoryManager,
    OrchestratorConfig,
    delete_history_file,
    list_history_files,
    load_history_from_file,
    run_claude_query,
    run_orchestrator,
    save_history_to_file,
)
from .prompts import (
    build_requirement_verification_prompt,
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
    RequirementDefinition,
    Task,
    TaskDefinition,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)

LOOPB_HISTORY_DIR = ".task-orchestrator-history/loopb"


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

        history = LoopBExecutionHistory(
            history_id=self._generate_history_id(requirements_path),
            requirements_path=str(Path(requirements_path).resolve()),
            tasks_output_dir=tasks_output_dir,
            started_at=now,
            updated_at=now,
            status=LoopBStatus.GENERATING_TASKS,
            max_iterations=config.max_iterations,
            current_iteration=0,
            iterations=[],
            completed_task_ids=[],
            loop_c_history_ids=[],
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

    async def _run_loop(self) -> LoopBExecutionHistory:
        """Execute the main Loop B iteration loop."""
        assert self.history is not None

        start_iteration = self.history.current_iteration
        for iteration in range(start_iteration, self.config.max_iterations):
            self.history.current_iteration = iteration + 1
            self._save_history()

            # Step 1: Generate tasks
            logger.info(f"Loop B iteration {iteration + 1}: Generating tasks")
            self.history.status = LoopBStatus.GENERATING_TASKS
            self._save_history()

            tasks = await self._generate_tasks_with_coverage_check(iteration)
            if not tasks:
                self._fail("Failed to generate tasks with full coverage")
                break

            # Step 2: Write tasks to YAML file
            tasks_yaml_path = self._write_tasks_yaml(tasks, iteration)

            # Step 3: Execute Loop C
            logger.info(f"Loop B iteration {iteration + 1}: Executing Loop C")
            self.history.status = LoopBStatus.EXECUTING_TASKS
            self._save_history()

            loop_c_result, loop_c_history_id = await self._execute_loop_c(
                tasks_yaml_path
            )

            # Update completed tasks
            self._update_completed_tasks(tasks, loop_c_result)

            # Step 4: Verify requirements
            logger.info(f"Loop B iteration {iteration + 1}: Verifying requirements")
            self.history.status = LoopBStatus.VERIFYING_REQUIREMENTS
            self._save_history()

            verification = await self._verify_requirements()
            unmet_requirements = get_unmet_requirement_ids(verification)

            # Record iteration
            iteration_record = LoopBIteration(
                iteration_number=iteration + 1,
                tasks_yaml_path=tasks_yaml_path,
                loop_c_history_id=loop_c_history_id,
                verification_result=verification,
                unmet_requirements=unmet_requirements,
            )
            self.history.iterations.append(iteration_record)
            if loop_c_history_id:
                self.history.loop_c_history_ids.append(loop_c_history_id)
            self._save_history()

            # Step 5: Check completion
            if verification.get("all_requirements_met", False):
                logger.info("Loop B: All requirements met")
                self._complete(verification)
                break

            if iteration + 1 >= self.config.max_iterations:
                self._fail(f"Max iterations ({self.config.max_iterations}) reached")
                break

            # Prepare for next iteration
            logger.info(
                f"Loop B: {len(unmet_requirements)} requirements not met, "
                "generating additional tasks"
            )
            self.history.status = LoopBStatus.GENERATING_ADDITIONAL_TASKS
            self._save_history()

        return self.history

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

    async def _generate_tasks_with_coverage_check(
        self, iteration: int, max_retries: int = 3
    ) -> list[Task]:
        """Generate tasks and retry if coverage is incomplete."""
        coverage_feedback: str | None = None
        tasks: list[Task] = []
        uncovered: list[str] = []

        for attempt in range(max_retries):
            tasks = await self._generate_tasks(iteration, coverage_feedback)
            if not tasks:
                return []

            all_covered, uncovered = check_coverage(self.requirements, tasks)
            if all_covered:
                return tasks

            logger.info(
                f"Coverage check failed (attempt {attempt + 1}/{max_retries}): "
                f"uncovered criteria: {uncovered}"
            )
            coverage_feedback = (
                f"以下のacceptance criteriaがvalidationでカバーされていません: {uncovered}. "
                "各validationのcoversフィールドで全criteriaを網羅してください。"
            )

        logger.warning(f"Max coverage retries reached. Uncovered: {uncovered}")
        return tasks  # Return last attempt even if incomplete

    async def _generate_tasks(
        self, iteration: int, coverage_feedback: str | None = None
    ) -> list[Task]:
        """Generate tasks using LLM."""
        assert self.history is not None
        if iteration == 0 and not coverage_feedback:
            prompt = build_task_generation_prompt(self.requirements)
        else:
            previous_verification = (
                self.history.iterations[-1].verification_result
                if self.history.iterations
                else None
            )
            feedback_parts = []
            if previous_verification:
                feedback_parts.append(
                    previous_verification.get("feedback_for_additional_tasks", "")
                )
            if coverage_feedback:
                feedback_parts.append(coverage_feedback)
            feedback = "\n".join(filter(None, feedback_parts)) or None

            prompt = build_task_generation_prompt(
                self.requirements,
                completed_tasks=self.all_completed_tasks if iteration > 0 else None,
                previous_feedback=feedback,
            )

        output = await run_claude_query(
            prompt, self.config.orchestrator_config, phase="task_generation"
        )
        return parse_generated_tasks(output)

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

        tasks_data = {
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

    async def _execute_loop_c(
        self, tasks_yaml_path: str
    ) -> tuple[OrchestratorResult, str | None]:
        """Execute Loop C with the generated tasks."""
        task_definition = TaskDefinition.from_yaml(tasks_yaml_path)

        result = await run_orchestrator(
            task_definition,
            config=self.config.orchestrator_config,
            yaml_path=tasks_yaml_path,
            history_manager=self.loop_c_history_manager,
        )

        # Get the history ID from the most recent history
        loop_c_history_id = None
        if self.loop_c_history_manager:
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

    async def _verify_requirements(self) -> dict[str, Any]:
        """Verify if requirements are met."""
        # Filter to only approved task results
        approved_results = [
            r for r in self.all_task_results if r.status == TaskStatus.APPROVED
        ]

        prompt = build_requirement_verification_prompt(
            self.requirements, approved_results
        )
        output = await run_claude_query(
            prompt, self.config.orchestrator_config, phase="verification"
        )
        return parse_verification_result(output)


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
