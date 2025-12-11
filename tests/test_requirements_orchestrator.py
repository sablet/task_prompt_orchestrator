"""Tests for requirements_orchestrator.py - Loop B implementation."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from task_prompt_orchestrator.orchestrator import OrchestratorConfig
from task_prompt_orchestrator.requirements_orchestrator import (
    LoopBHistoryManager,
    RequirementsOrchestrator,
    RequirementsOrchestratorConfig,
)
from task_prompt_orchestrator.schema import (
    LoopBStatus,
    OrchestratorResult,
    Requirement,
    RequirementDefinition,
    TaskResult,
    TaskStatus,
)


@pytest.fixture
def sample_requirements() -> RequirementDefinition:
    """Create sample requirements for testing."""
    return RequirementDefinition(
        requirements=[
            Requirement(id="req_1", name="Requirement 1", acceptance_criteria=["Done"]),
            Requirement(id="req_2", name="Requirement 2", acceptance_criteria=["Done"]),
        ],
    )


def make_task_response(task_id: str = "task_1", covers: list[str] | None = None) -> str:
    """Create mock task generation response with coverage mapping."""
    if covers is None:
        covers = ["req_1.1", "req_2.1"]  # Default covers both requirements
    covers_str = ", ".join(covers)
    return f"""```yaml
tasks:
  - id: {task_id}
    name: Task
    instruction: Do
    validation:
      - criterion: "OK"
        covers: [{covers_str}]
```"""


def make_verify_response(all_met: bool, unmet: list[str] | None = None) -> str:
    """Create mock verification response."""
    statuses = ", ".join(
        f'{{"requirement_id": "{r}", "met": {str(r not in (unmet or [])).lower()}}}'
        for r in ["req_1", "req_2"]
    )
    return f'{{"all_requirements_met": {str(all_met).lower()}, "requirement_status": [{statuses}], "feedback_for_additional_tasks": ""}}'


def make_mock_result(
    task_ids: list[str], success: bool = True, failed_ids: list[str] | None = None
) -> OrchestratorResult:
    """Create mock OrchestratorResult."""
    failed_ids = failed_ids or []
    results = []
    for tid in task_ids:
        status = TaskStatus.FAILED if tid in failed_ids else TaskStatus.APPROVED
        results.append(TaskResult(task_id=tid, status=status))
    return OrchestratorResult(
        success=success,
        task_results=results,
        total_attempts=1,
        summary="Done" if success else "Failed",
    )


class TestLoopBHistoryManager:
    """Tests for LoopBHistoryManager."""

    def test_history_lifecycle(self) -> None:
        """Test create, load, list, and delete history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoopBHistoryManager(temp_dir)
            config = RequirementsOrchestratorConfig(max_iterations=3)

            # Create
            h1 = manager.create_history("/path/req1.yaml", config)
            h2 = manager.create_history("/path/req2.yaml", config)
            assert h1.status == LoopBStatus.GENERATING_TASKS
            assert h1.max_iterations == 3

            # Load
            loaded = manager.load_history(h1.history_id)
            assert loaded.history_id == h1.history_id

            # List all
            assert len(manager.list_histories()) == 2

            # List incomplete
            h1.status = LoopBStatus.COMPLETED
            manager.save_history(h1)
            incomplete = manager.list_incomplete_histories()
            assert len(incomplete) == 1
            assert incomplete[0].history_id == h2.history_id

            # Delete
            assert manager.delete_history(h1.history_id) is True
            assert manager.delete_history("nonexistent") is False


class TestRequirementsOrchestrator:
    """Tests for RequirementsOrchestrator - Loop B orchestration."""

    @pytest.mark.asyncio
    async def test_iteration_scenarios(self, sample_requirements: RequirementDefinition) -> None:
        """Test single iteration success, multiple iterations, and max iterations failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Scenario 1: Single iteration success
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir)
            with self._mock_claude_and_loopc(
                [make_task_response(), make_verify_response(True)],
                [make_mock_result(["task_1"])],
            ):
                result = await orchestrator.run()
            assert result.status == LoopBStatus.COMPLETED
            assert result.current_iteration == 1
            assert "task_1" in result.completed_task_ids

            # Scenario 2: Two iterations until success
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir)
            with self._mock_claude_and_loopc(
                [
                    make_task_response("t1"), make_verify_response(False, ["req_2"]),
                    make_task_response("t2"), make_verify_response(True),
                ],
                [make_mock_result(["t1"]), make_mock_result(["t2"])],
            ):
                result = await orchestrator.run()
            assert result.status == LoopBStatus.COMPLETED
            assert result.current_iteration == 2
            assert set(result.completed_task_ids) == {"t1", "t2"}

            # Scenario 3: Max iterations reached
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir, max_iter=2)
            with self._mock_claude_and_loopc(
                [
                    make_task_response(), make_verify_response(False, ["req_2"]),
                    make_task_response(), make_verify_response(False, ["req_2"]),
                ],
                [make_mock_result(["x"]), make_mock_result(["y"])],
            ):
                result = await orchestrator.run()
            assert result.status == LoopBStatus.FAILED
            assert "Max iterations" in (result.error or "")

    @pytest.mark.asyncio
    async def test_task_generation_failure(self, sample_requirements: RequirementDefinition) -> None:
        """Test failure when task generation returns no tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir)
            with patch(
                "task_prompt_orchestrator.requirements_orchestrator.run_claude_query",
                new_callable=AsyncMock,
                return_value="invalid yaml",
            ):
                result = await orchestrator.run()
            assert result.status == LoopBStatus.FAILED
            assert "Failed to generate tasks" in (result.error or "")

    @pytest.mark.asyncio
    async def test_in_memory_execution(self, sample_requirements: RequirementDefinition) -> None:
        """Test execution without history manager."""
        config = RequirementsOrchestratorConfig(
            max_iterations=1,
            tasks_output_dir="/tmp/test_tasks",
            orchestrator_config=OrchestratorConfig(stream_output=False),
        )
        orchestrator = RequirementsOrchestrator(
            requirements=sample_requirements, config=config, history_manager=None, requirements_path=""
        )
        with self._mock_claude_and_loopc(
            [make_task_response(), make_verify_response(True)],
            [make_mock_result(["task_1"])],
        ):
            result = await orchestrator.run()
        assert result.status == LoopBStatus.COMPLETED
        assert result.history_id.startswith("inmemory_")

    @pytest.mark.asyncio
    async def test_loop_c_failure_scenarios(self, sample_requirements: RequirementDefinition) -> None:
        """Test Loop B behavior when Loop C fails or has partial failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Scenario 1: Loop C returns success=False (max retries reached)
            # Use max_iter=1 to simplify
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir, max_iter=1)
            with self._mock_claude_and_loopc(
                [make_task_response(), make_verify_response(False, ["req_1", "req_2"])],
                [make_mock_result(["task_1"], success=False, failed_ids=["task_1"])],
            ):
                result = await orchestrator.run()
            # Loop B continues to verification even if Loop C failed
            # completed_task_ids should be empty since task_1 failed
            assert "task_1" not in result.completed_task_ids
            # Loop B reaches max iterations and fails
            assert result.status == LoopBStatus.FAILED

            # Scenario 2: Loop C raises exception - should propagate
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir, max_iter=1)
            with patch(
                "task_prompt_orchestrator.requirements_orchestrator.run_claude_query",
                new_callable=AsyncMock,
                side_effect=[make_task_response()],
            ):
                with patch(
                    "task_prompt_orchestrator.requirements_orchestrator.run_orchestrator",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("Loop C crashed"),
                ):
                    with pytest.raises(RuntimeError, match="Loop C crashed"):
                        await orchestrator.run()

            # Scenario 3: Partial success - some tasks approved, some failed
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir, max_iter=1)
            with self._mock_claude_and_loopc(
                [make_task_response("t1"), make_verify_response(False, ["req_2"])],
                [make_mock_result(["t1", "t2"], success=False, failed_ids=["t2"])],
            ):
                result = await orchestrator.run()
            # t1 should be in completed, t2 should not
            assert "t1" in result.completed_task_ids
            assert "t2" not in result.completed_task_ids

    def _create_orchestrator(
        self, requirements: RequirementDefinition, temp_dir: str, max_iter: int = 3
    ) -> RequirementsOrchestrator:
        """Helper to create orchestrator with standard config."""
        return RequirementsOrchestrator(
            requirements=requirements,
            config=RequirementsOrchestratorConfig(
                max_iterations=max_iter,
                tasks_output_dir=str(Path(temp_dir) / "tasks"),
                orchestrator_config=OrchestratorConfig(stream_output=False),
            ),
            history_manager=LoopBHistoryManager(temp_dir),
            requirements_path="/path/to/req.yaml",
        )

    @staticmethod
    def _mock_claude_and_loopc(claude_responses: list[str], loopc_results: list[OrchestratorResult]):
        """Context manager to mock both run_claude_query and run_orchestrator."""
        claude_mock = patch(
            "task_prompt_orchestrator.requirements_orchestrator.run_claude_query",
            new_callable=AsyncMock,
            side_effect=claude_responses,
        )
        loopc_mock = patch(
            "task_prompt_orchestrator.requirements_orchestrator.run_orchestrator",
            new_callable=AsyncMock,
            side_effect=loopc_results,
        )

        class CombinedContext:
            def __enter__(self):
                self.c = claude_mock.__enter__()
                self.l = loopc_mock.__enter__()
                return self

            def __exit__(self, *args):
                loopc_mock.__exit__(*args)
                claude_mock.__exit__(*args)

        return CombinedContext()
