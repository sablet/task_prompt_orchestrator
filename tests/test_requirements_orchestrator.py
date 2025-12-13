"""Tests for requirements_orchestrator.py - Loop B implementation."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from task_prompt_orchestrator.orchestrator import (
    OrchestratorConfig,
    StepResult,
    execute_task,
    run_orchestrator,
)
from task_prompt_orchestrator.requirements_orchestrator import (
    LoopBHistoryManager,
    RequirementsOrchestrator,
    RequirementsOrchestratorConfig,
)
from task_prompt_orchestrator.schema import (
    AcceptanceCriterion,
    LoopBStatus,
    OrchestratorResult,
    Requirement,
    RequirementDefinition,
    Task,
    TaskDefinition,
    TaskResult,
    TaskStatus,
    YamlType,
    detect_yaml_type,
)


@pytest.fixture
def sample_requirements() -> RequirementDefinition:
    """Create sample requirements for testing."""
    return RequirementDefinition(
        requirements=[
            Requirement(
                id="req_1",
                name="Requirement 1",
                acceptance_criteria=[
                    AcceptanceCriterion(criterion="Done", verify="Check")
                ],
            ),
            Requirement(
                id="req_2",
                name="Requirement 2",
                acceptance_criteria=[
                    AcceptanceCriterion(criterion="Done", verify="Check")
                ],
            ),
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
    """Create mock verification response (legacy, for all requirements at once)."""
    statuses = ", ".join(
        f'{{"requirement_id": "{r}", "met": {str(r not in (unmet or [])).lower()}}}'
        for r in ["req_1", "req_2"]
    )
    return f'{{"all_requirements_met": {str(all_met).lower()}, "requirement_status": [{statuses}], "feedback_for_additional_tasks": ""}}'


def make_single_verify_response(req_id: str, met: bool) -> str:
    """Create mock verification response for a single requirement."""
    return f'{{"requirement_id": "{req_id}", "met": {str(met).lower()}, "summary": "Verified", "criteria_results": [], "issues": []}}'


def make_exploration_response() -> str:
    """Create mock shared exploration response."""
    return """### コードベース構造
- CLI: scripts/main.py
- Tests: tests/
- Docs: doc/

### 実行結果
#### pytest tests/
$ pytest tests/ -v
All tests passed.

### 生成されたアーティファクト
None generated."""


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

            # Create actual files
            req1_path = Path(temp_dir) / "req1.yaml"
            req2_path = Path(temp_dir) / "req2.yaml"
            req1_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: []\n"
            )
            req2_path.write_text(
                "requirements:\n  - id: req_2\n    name: Test\n    acceptance_criteria: []\n"
            )

            # Create
            h1 = manager.create_history(str(req1_path), config)
            h2 = manager.create_history(str(req2_path), config)
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
    async def test_iteration_scenarios(
        self, sample_requirements: RequirementDefinition
    ) -> None:
        """Test single iteration success, multiple iterations, and max iterations failure.

        New flow (pre-verification first):
        - Iter start: Pre-verify all reqs -> Get unmet list
        - If all met: Complete
        - Else: Task gen (for unmet only) -> LoopC -> Next iter
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Scenario 1: All requirements already met at start
            # Flow: exploration -> pre-verify req_1 (met) -> pre-verify req_2 (met) -> Complete
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir)
            with self._mock_claude_and_loopc(
                [
                    make_exploration_response(),  # Shared exploration
                    make_single_verify_response("req_1", True),
                    make_single_verify_response("req_2", True),
                ],
                [],  # No LoopC execution
            ):
                result = await orchestrator.run()
            assert result.status == LoopBStatus.COMPLETED
            # No tasks were executed since requirements were already met
            assert result.completed_task_ids == []

            # Scenario 2: Some requirements unmet, then met after one iteration
            # Flow: exploration -> pre-verify (req_2 unmet) -> task_gen -> LoopC
            #    -> iter2: exploration (new iter) -> pre-verify (all met) -> Complete
            # Note: exploration is re-run each iteration
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir)
            with self._mock_claude_and_loopc(
                [
                    make_exploration_response(),  # Iter 1: exploration
                    # Iter 1: pre-verification
                    make_single_verify_response("req_1", True),
                    make_single_verify_response("req_2", False),  # Unmet
                    make_task_response("t1"),  # Task gen for req_2
                    make_exploration_response(),  # Iter 2: exploration (new iteration)
                    # Iter 2: pre-verification
                    make_single_verify_response("req_1", True),
                    make_single_verify_response("req_2", True),  # Now met
                ],
                [make_mock_result(["t1"])],
            ):
                result = await orchestrator.run()
            assert result.status == LoopBStatus.COMPLETED
            assert "t1" in result.completed_task_ids

            # Scenario 3: Max iterations reached
            # Flow: exploration -> pre-verify (unmet) -> task -> LoopC
            #    -> exploration (new iter) -> pre-verify (still unmet) -> task -> LoopC -> fail
            orchestrator = self._create_orchestrator(
                sample_requirements, temp_dir, max_iter=2
            )
            with self._mock_claude_and_loopc(
                [
                    make_exploration_response(),  # Iter 1: exploration
                    # Iter 1
                    make_single_verify_response("req_1", True),
                    make_single_verify_response("req_2", False),
                    make_task_response("t1"),
                    make_exploration_response(),  # Iter 2: exploration (new iteration)
                    # Iter 2
                    make_single_verify_response("req_1", True),
                    make_single_verify_response("req_2", False),  # Still unmet
                    make_task_response("t2"),
                    # Would be Iter 3 but max reached
                ],
                [make_mock_result(["t1"]), make_mock_result(["t2"])],
            ):
                result = await orchestrator.run()
            assert result.status == LoopBStatus.FAILED
            assert "Max iterations" in (result.error or "")

    @pytest.mark.asyncio
    async def test_task_generation_failure(
        self, sample_requirements: RequirementDefinition
    ) -> None:
        """Test failure when task generation returns no tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = self._create_orchestrator(sample_requirements, temp_dir)
            call_count = 0

            async def mock_query(prompt, config, phase=None, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Shared exploration
                    return make_exploration_response()
                if call_count <= 3:
                    # Pre-verification: return unmet
                    req_id = "req_1" if call_count == 2 else "req_2"
                    return make_single_verify_response(req_id, False)
                # Task generation: invalid yaml
                return "invalid yaml"

            with patch(
                "task_prompt_orchestrator.loopb_verification.run_claude_query",
                new_callable=AsyncMock,
                side_effect=mock_query,
            ):
                with patch(
                    "task_prompt_orchestrator.loopb_tasks.run_claude_query",
                    new_callable=AsyncMock,
                    side_effect=mock_query,
                ):
                    result = await orchestrator.run()
            assert result.status == LoopBStatus.FAILED
            assert "Failed to generate tasks" in (result.error or "")

    @pytest.mark.asyncio
    async def test_in_memory_execution(
        self, sample_requirements: RequirementDefinition
    ) -> None:
        """Test execution without history manager."""
        config = RequirementsOrchestratorConfig(
            max_iterations=2,
            tasks_output_dir="/tmp/test_tasks",
            orchestrator_config=OrchestratorConfig(stream_output=False),
        )
        orchestrator = RequirementsOrchestrator(
            requirements=sample_requirements,
            config=config,
            history_manager=None,
            requirements_path="",
        )
        with self._mock_claude_and_loopc(
            [
                make_exploration_response(),  # Iter 1: exploration
                # Iter 1: pre-verify (unmet) -> task_gen
                make_single_verify_response("req_1", False),
                make_single_verify_response("req_2", False),
                make_task_response(),
                make_exploration_response(),  # Iter 2: exploration (new iteration)
                # Iter 2: pre-verify (all met) -> complete
                make_single_verify_response("req_1", True),
                make_single_verify_response("req_2", True),
            ],
            [make_mock_result(["task_1"])],
        ):
            result = await orchestrator.run()
        assert result.status == LoopBStatus.COMPLETED
        assert result.history_id.startswith("inmemory_")

    @pytest.mark.asyncio
    async def test_loop_c_failure_scenarios(
        self, sample_requirements: RequirementDefinition
    ) -> None:
        """Test Loop B behavior when Loop C fails or has partial failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Scenario 1: Loop C returns success=False (max retries reached)
            # Flow: exploration -> pre-verify (unmet) -> task_gen -> LoopC (fail) -> max iter
            orchestrator = self._create_orchestrator(
                sample_requirements, temp_dir, max_iter=1
            )
            with self._mock_claude_and_loopc(
                [
                    make_exploration_response(),  # Shared exploration
                    # Pre-verification
                    make_single_verify_response("req_1", False),
                    make_single_verify_response("req_2", False),
                    make_task_response(),
                ],
                [make_mock_result(["task_1"], success=False, failed_ids=["task_1"])],
            ):
                result = await orchestrator.run()
            # completed_task_ids should be empty since task_1 failed
            assert "task_1" not in result.completed_task_ids
            # Loop B reaches max iterations and fails
            assert result.status == LoopBStatus.FAILED

            # Scenario 2: Loop C raises exception - should propagate
            orchestrator = self._create_orchestrator(
                sample_requirements, temp_dir, max_iter=1
            )
            with patch(
                "task_prompt_orchestrator.loopb_verification.run_claude_query",
                new_callable=AsyncMock,
                side_effect=[
                    make_exploration_response(),  # Shared exploration
                    # Pre-verification
                    make_single_verify_response("req_1", False),
                    make_single_verify_response("req_2", False),
                ],
            ):
                with patch(
                    "task_prompt_orchestrator.loopb_tasks.run_claude_query",
                    new_callable=AsyncMock,
                    side_effect=[make_task_response()],
                ):
                    with patch(
                        "task_prompt_orchestrator.loopb_tasks.run_orchestrator",
                        new_callable=AsyncMock,
                        side_effect=RuntimeError("Loop C crashed"),
                    ):
                        with pytest.raises(RuntimeError, match="Loop C crashed"):
                            await orchestrator.run()

            # Scenario 3: Partial success - some tasks approved, some failed
            orchestrator = self._create_orchestrator(
                sample_requirements, temp_dir, max_iter=1
            )
            with self._mock_claude_and_loopc(
                [
                    make_exploration_response(),  # Shared exploration
                    # Pre-verification
                    make_single_verify_response("req_1", False),
                    make_single_verify_response("req_2", False),
                    make_task_response("t1"),
                ],
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
        # Create actual requirements file
        req_path = Path(temp_dir) / "req.yaml"
        req_path.write_text(
            "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n  - id: req_2\n    name: Test2\n    acceptance_criteria: [Done]\n"
        )
        return RequirementsOrchestrator(
            requirements=requirements,
            config=RequirementsOrchestratorConfig(
                max_iterations=max_iter,
                tasks_output_dir=str(Path(temp_dir) / "tasks"),
                orchestrator_config=OrchestratorConfig(stream_output=False),
            ),
            history_manager=LoopBHistoryManager(temp_dir),
            requirements_path=str(req_path),
        )

    @staticmethod
    def _mock_claude_and_loopc(
        claude_responses: list[str], loopc_results: list[OrchestratorResult]
    ):
        """Context manager to mock both run_claude_query and run_orchestrator."""
        # Use a shared iterator so both mocks consume from the same sequence
        response_iter = iter(claude_responses)

        async def shared_side_effect(*args, **kwargs):
            return next(response_iter)

        claude_mock_verification = patch(
            "task_prompt_orchestrator.loopb_verification.run_claude_query",
            new_callable=AsyncMock,
            side_effect=shared_side_effect,
        )
        claude_mock_tasks = patch(
            "task_prompt_orchestrator.loopb_tasks.run_claude_query",
            new_callable=AsyncMock,
            side_effect=shared_side_effect,
        )
        loopc_mock = patch(
            "task_prompt_orchestrator.loopb_tasks.run_orchestrator",
            new_callable=AsyncMock,
            side_effect=loopc_results,
        )

        class CombinedContext:
            def __enter__(self):
                self.cv = claude_mock_verification.__enter__()
                self.ct = claude_mock_tasks.__enter__()
                self.l = loopc_mock.__enter__()
                return self

            def __exit__(self, *args):
                loopc_mock.__exit__(*args)
                claude_mock_tasks.__exit__(*args)
                claude_mock_verification.__exit__(*args)

        return CombinedContext()


class TestYamlFormatValidation:
    """Test Case A: Invalid YAML format detection and error exit."""

    def test_detect_yaml_type_loopb(self) -> None:
        """Loop B (requirements) YAML is correctly detected."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: []\n"
            )
            f.flush()
            assert detect_yaml_type(f.name) == YamlType.LOOP_B

    def test_detect_yaml_type_loopc(self) -> None:
        """Loop C (tasks) YAML is correctly detected."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "tasks:\n  - id: task_1\n    name: Test\n    instruction: Do\n    validation: []\n"
            )
            f.flush()
            assert detect_yaml_type(f.name) == YamlType.LOOP_C

    def test_detect_yaml_type_unknown(self) -> None:
        """Invalid YAML format (neither Loop B nor Loop C) returns UNKNOWN."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid:\n  key: value\n")
            f.flush()
            assert detect_yaml_type(f.name) == YamlType.UNKNOWN

    def test_detect_yaml_type_empty(self) -> None:
        """Empty YAML file returns UNKNOWN."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            assert detect_yaml_type(f.name) == YamlType.UNKNOWN


class TestCLIInvalidYamlFormat:
    """Test Case A: CLI behavior with invalid YAML format."""

    def test_invalid_yaml_format_error(self) -> None:
        """CLI returns error code 1 for invalid YAML format."""
        from task_prompt_orchestrator.cli import (
            InvalidYamlFormatError,
            _detect_loop_type,
        )
        import argparse

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid:\n  key: value\n")
            f.flush()

            args = argparse.Namespace(input_file=f.name, loopb=False)
            with pytest.raises(InvalidYamlFormatError) as exc_info:
                _detect_loop_type(args)

            assert "Invalid YAML format" in str(exc_info.value)
            assert (
                "Expected 'tasks' key (Loop C) or 'requirements' key (Loop B)"
                in str(exc_info.value)
            )


class TestLoopBResumeFromInterruption:
    """Test Case B: Resume Loop B orchestrator after interruption during Loop C."""

    @pytest.mark.asyncio
    async def test_resume_from_executing_tasks_state(self) -> None:
        """Resume from EXECUTING_TASKS status should resume Loop C, not regenerate tasks.

        Flow: Resume -> LoopC -> Iter2 pre-verify (all met) -> Complete
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoopBHistoryManager(temp_dir)
            requirements = RequirementDefinition(
                requirements=[
                    Requirement(
                        id="req_1",
                        name="Requirement 1",
                        acceptance_criteria=[
                            AcceptanceCriterion(criterion="Done", verify="Check")
                        ],
                    ),
                ]
            )

            # Create requirements file
            req_path = Path(temp_dir) / "requirements.yaml"
            req_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n"
            )

            # Create tasks file (simulating that tasks were already generated)
            tasks_dir = Path(temp_dir) / ".task-orchestrator-history/loopb/tasks"
            tasks_dir.mkdir(parents=True, exist_ok=True)
            tasks_yaml_path = tasks_dir / "requirements-tasks-iter1.yaml"
            tasks_yaml_path.write_text("""tasks:
  - id: task_1
    name: Task 1
    instruction: Do task 1
    validation:
      - criterion: "OK"
        covers: [req_1.1]
    depends_on: []
""")

            config = RequirementsOrchestratorConfig(
                max_iterations=3,
                tasks_output_dir=str(tasks_dir),
                orchestrator_config=OrchestratorConfig(stream_output=False),
            )

            # Create history in EXECUTING_TASKS state (simulating interruption during Loop C)
            history = manager.create_history(str(req_path), config)
            history.status = LoopBStatus.EXECUTING_TASKS
            history.current_iteration = 1
            history.iterations = []  # No iteration record yet (interrupted before completion)
            manager.save_history(history)

            # Create orchestrator for resume
            orchestrator = RequirementsOrchestrator(
                requirements=requirements,
                config=config,
                history_manager=manager,
                requirements_path=str(req_path),
            )

            # Mock the LLM calls and Loop C execution
            task_gen_call_count = 0
            loop_c_call_count = 0
            verification_call_count = 0

            async def mock_claude_query(
                prompt: str, config: OrchestratorConfig, phase: str = "", **kwargs
            ) -> str:
                nonlocal task_gen_call_count, verification_call_count
                if phase == "task_generation":
                    task_gen_call_count += 1
                elif phase == "verification":
                    verification_call_count += 1
                # Return single requirement verification result (all met for iter2 pre-verify)
                return '{"requirement_id": "req_1", "met": true, "summary": "Verified", "criteria_results": [], "issues": []}'

            async def mock_run_orchestrator(*args, **kwargs) -> OrchestratorResult:
                nonlocal loop_c_call_count
                loop_c_call_count += 1
                return OrchestratorResult(
                    success=True,
                    task_results=[
                        TaskResult(task_id="task_1", status=TaskStatus.APPROVED)
                    ],
                    total_attempts=1,
                    summary="Done",
                )

            with patch(
                "task_prompt_orchestrator.loopb_verification.run_claude_query",
                new_callable=AsyncMock,
                side_effect=mock_claude_query,
            ):
                with patch(
                    "task_prompt_orchestrator.loopb_tasks.run_claude_query",
                    new_callable=AsyncMock,
                    side_effect=mock_claude_query,
                ):
                    with patch(
                        "task_prompt_orchestrator.loopb_tasks.run_orchestrator",
                        new_callable=AsyncMock,
                        side_effect=mock_run_orchestrator,
                    ):
                        result = await orchestrator.resume(history)

            # Verify: task generation should NOT have been called (tasks already exist)
            assert (
                task_gen_call_count == 0
            ), f"Task generation called {task_gen_call_count} times, expected 0"

            # Verify: Loop C should have been called
            assert (
                loop_c_call_count == 1
            ), f"Loop C called {loop_c_call_count} times, expected 1"

            # Verify: pre-verification should have been called (for iter 2)
            assert (
                verification_call_count == 1
            ), f"Verification called {verification_call_count} times, expected 1"

            # Verify: completed successfully
            assert result.status == LoopBStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_resume_from_generating_tasks_state(self) -> None:
        """Resume from GENERATING_TASKS status should regenerate tasks.

        Flow: Resume (skip pre-verify) -> TaskGen -> LoopC -> Iter2 pre-verify (met) -> Complete
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoopBHistoryManager(temp_dir)
            requirements = RequirementDefinition(
                requirements=[
                    Requirement(
                        id="req_1",
                        name="Requirement 1",
                        acceptance_criteria=[
                            AcceptanceCriterion(criterion="Done", verify="Check")
                        ],
                    ),
                ]
            )

            # Create requirements file
            req_path = Path(temp_dir) / "requirements.yaml"
            req_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n"
            )

            tasks_dir = Path(temp_dir) / ".task-orchestrator-history/loopb/tasks"
            tasks_dir.mkdir(parents=True, exist_ok=True)

            config = RequirementsOrchestratorConfig(
                max_iterations=3,
                tasks_output_dir=str(tasks_dir),
                orchestrator_config=OrchestratorConfig(stream_output=False),
            )

            # Create history in GENERATING_TASKS state (simulating interruption during task generation)
            history = manager.create_history(str(req_path), config)
            history.status = LoopBStatus.GENERATING_TASKS
            history.current_iteration = 1
            manager.save_history(history)

            # Create orchestrator for resume
            orchestrator = RequirementsOrchestrator(
                requirements=requirements,
                config=config,
                history_manager=manager,
                requirements_path=str(req_path),
            )

            # Track calls
            task_gen_call_count = 0
            loop_c_call_count = 0
            verification_call_count = 0

            async def mock_claude_query(
                prompt: str, config: OrchestratorConfig, phase: str = "", **kwargs
            ) -> str:
                nonlocal task_gen_call_count, verification_call_count
                if phase == "task_generation":
                    task_gen_call_count += 1
                    return """```yaml
tasks:
  - id: task_1
    name: Task 1
    instruction: Do task 1
    validation:
      - criterion: "OK"
        covers: [req_1.1]
```"""
                elif phase == "verification":
                    verification_call_count += 1
                # Return single requirement verification result (all met for iter2 pre-verify)
                return '{"requirement_id": "req_1", "met": true, "summary": "Verified", "criteria_results": [], "issues": []}'

            async def mock_run_orchestrator(*args, **kwargs) -> OrchestratorResult:
                nonlocal loop_c_call_count
                loop_c_call_count += 1
                return OrchestratorResult(
                    success=True,
                    task_results=[
                        TaskResult(task_id="task_1", status=TaskStatus.APPROVED)
                    ],
                    total_attempts=1,
                    summary="Done",
                )

            with patch(
                "task_prompt_orchestrator.loopb_verification.run_claude_query",
                new_callable=AsyncMock,
                side_effect=mock_claude_query,
            ):
                with patch(
                    "task_prompt_orchestrator.loopb_tasks.run_claude_query",
                    new_callable=AsyncMock,
                    side_effect=mock_claude_query,
                ):
                    with patch(
                        "task_prompt_orchestrator.loopb_tasks.run_orchestrator",
                        new_callable=AsyncMock,
                        side_effect=mock_run_orchestrator,
                    ):
                        result = await orchestrator.resume(history)

            # Verify: task generation SHOULD have been called
            assert (
                task_gen_call_count == 1
            ), f"Task generation called {task_gen_call_count} times, expected 1"

            # Verify: Loop C should have been called
            assert (
                loop_c_call_count == 1
            ), f"Loop C called {loop_c_call_count} times, expected 1"

            # Verify: pre-verification should have been called (for iter 2)
            assert (
                verification_call_count == 1
            ), f"Verification called {verification_call_count} times, expected 1"

            # Verify: completed successfully
            assert result.status == LoopBStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_resume_info_for_executing_tasks(self) -> None:
        """_get_resume_info returns correct values for EXECUTING_TASKS status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoopBHistoryManager(temp_dir)
            requirements = RequirementDefinition(
                requirements=[
                    Requirement(
                        id="req_1",
                        name="Req",
                        acceptance_criteria=[
                            AcceptanceCriterion(criterion="Done", verify="Check")
                        ],
                    )
                ]
            )

            req_path = Path(temp_dir) / "requirements.yaml"
            req_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n"
            )

            config = RequirementsOrchestratorConfig(max_iterations=3)

            orchestrator = RequirementsOrchestrator(
                requirements=requirements,
                config=config,
                history_manager=manager,
                requirements_path=str(req_path),
            )

            # Simulate history in EXECUTING_TASKS state at iteration 1
            history = manager.create_history(str(req_path), config)
            history.status = LoopBStatus.EXECUTING_TASKS
            history.current_iteration = 1
            orchestrator.history = history

            resume_info = orchestrator._get_resume_info()

            assert resume_info["start_iteration"] == 0  # 0-indexed for iteration 1
            assert resume_info["skip_task_generation"] is True
            assert resume_info["skip_initial_verification"] is True
            assert resume_info["resume_loop_c"] is True

    @pytest.mark.asyncio
    async def test_resume_info_for_generating_tasks(self) -> None:
        """_get_resume_info returns correct values for GENERATING_TASKS status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoopBHistoryManager(temp_dir)
            requirements = RequirementDefinition(
                requirements=[
                    Requirement(
                        id="req_1",
                        name="Req",
                        acceptance_criteria=[
                            AcceptanceCriterion(criterion="Done", verify="Check")
                        ],
                    )
                ]
            )

            req_path = Path(temp_dir) / "requirements.yaml"
            req_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n"
            )

            config = RequirementsOrchestratorConfig(max_iterations=3)

            orchestrator = RequirementsOrchestrator(
                requirements=requirements,
                config=config,
                history_manager=manager,
                requirements_path=str(req_path),
            )

            # Simulate history in GENERATING_TASKS state at iteration 1
            history = manager.create_history(str(req_path), config)
            history.status = LoopBStatus.GENERATING_TASKS
            history.current_iteration = 1
            orchestrator.history = history

            resume_info = orchestrator._get_resume_info()

            assert resume_info["start_iteration"] == 0  # 0-indexed for iteration 1
            assert resume_info["skip_task_generation"] is False  # Should regenerate
            assert (
                resume_info["skip_initial_verification"] is True
            )  # Pre-verify already done
            assert resume_info["resume_loop_c"] is False

    @pytest.mark.asyncio
    async def test_resume_info_for_verifying_requirements(self) -> None:
        """_get_resume_info returns correct values for VERIFYING_REQUIREMENTS status.

        VERIFYING_REQUIREMENTS means pre-verification was interrupted.
        Should re-run verification from the start of the same iteration.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoopBHistoryManager(temp_dir)
            requirements = RequirementDefinition(
                requirements=[
                    Requirement(
                        id="req_1",
                        name="Req",
                        acceptance_criteria=[
                            AcceptanceCriterion(criterion="Done", verify="Check")
                        ],
                    )
                ]
            )

            req_path = Path(temp_dir) / "requirements.yaml"
            req_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n"
            )

            config = RequirementsOrchestratorConfig(max_iterations=3)

            orchestrator = RequirementsOrchestrator(
                requirements=requirements,
                config=config,
                history_manager=manager,
                requirements_path=str(req_path),
            )

            # Simulate history in VERIFYING_REQUIREMENTS state at iteration 1
            history = manager.create_history(str(req_path), config)
            history.status = LoopBStatus.VERIFYING_REQUIREMENTS
            history.current_iteration = 1
            orchestrator.history = history

            resume_info = orchestrator._get_resume_info()

            # VERIFYING_REQUIREMENTS means pre-verification was interrupted
            # Re-run from same iteration with verification
            assert resume_info["start_iteration"] == 0  # 0-indexed for iteration 1
            assert resume_info["skip_task_generation"] is False
            assert (
                resume_info["skip_initial_verification"] is False
            )  # Re-run verification
            assert resume_info["resume_loop_c"] is False


class TestStepModeLoopC:
    """Test --step mode for Loop C: stops after each claude code call."""

    @staticmethod
    def _make_mock_claude_query(return_value: str):
        """Create a mock that sets _step_executed flag like the real function."""

        async def mock_query(prompt, config, phase=""):
            config._step_executed = True
            return return_value

        return mock_query

    @pytest.mark.asyncio
    async def test_step_mode_stops_after_instruction(self) -> None:
        """Step mode should stop after instruction phase (first claude call)."""
        task = Task(
            id="task_1",
            name="Test Task",
            instruction="Do something",
            validation=["Check result"],
        )
        config = OrchestratorConfig(
            stream_output=False,
            step_mode=True,
        )

        # Mock run_claude_query to simulate instruction execution
        with patch(
            "task_prompt_orchestrator.orchestrator.run_claude_query",
            new_callable=AsyncMock,
            side_effect=self._make_mock_claude_query("Instruction output"),
        ) as mock_query:
            result = await execute_task(task, config)

        # Should return StepResult after instruction
        assert isinstance(result, StepResult)
        assert result.stopped_after == "instruction"
        assert result.task_result.instruction_output == "Instruction output"
        assert result.task_result.validation_output is None
        # Only instruction query should be called
        assert mock_query.call_count == 1

    @pytest.mark.asyncio
    async def test_step_mode_stops_after_validation_on_resume(self) -> None:
        """Step mode should stop after validation phase when resuming from validation."""
        task = Task(
            id="task_1",
            name="Test Task",
            instruction="Do something",
            validation=["Check result"],
        )
        config = OrchestratorConfig(
            stream_output=False,
            step_mode=True,
        )
        # Reset step_executed flag (simulating fresh start for this phase)
        config._step_executed = False

        # Mock run_claude_query for validation
        with patch(
            "task_prompt_orchestrator.orchestrator.run_claude_query",
            new_callable=AsyncMock,
            side_effect=self._make_mock_claude_query(
                '{"approved": true, "feedback": ""}'
            ),
        ) as mock_query:
            result = await execute_task(
                task,
                config,
                skip_instruction=True,
                existing_instruction_output="Previous instruction output",
            )

        # Should return StepResult after validation
        assert isinstance(result, StepResult)
        assert result.stopped_after == "validation"
        assert result.task_result.instruction_output == "Previous instruction output"
        assert result.task_result.validation_output is not None
        assert result.task_result.validation_approved is True
        # Only validation query should be called
        assert mock_query.call_count == 1

    @pytest.mark.asyncio
    async def test_run_orchestrator_step_mode(self) -> None:
        """run_orchestrator should return step_stopped=True when step mode stops execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tasks YAML
            tasks_path = Path(temp_dir) / "tasks.yaml"
            tasks_path.write_text("""tasks:
  - id: task_1
    name: Task 1
    instruction: Do task 1
    validation:
      - Check it worked
  - id: task_2
    name: Task 2
    instruction: Do task 2
    validation:
      - Check it worked
""")

            task_def = TaskDefinition.from_yaml(str(tasks_path))
            config = OrchestratorConfig(
                stream_output=False,
                step_mode=True,
            )

            # Mock run_claude_query with step_executed flag setting
            with patch(
                "task_prompt_orchestrator.orchestrator.run_claude_query",
                new_callable=AsyncMock,
                side_effect=self._make_mock_claude_query("Output"),
            ):
                result = await run_orchestrator(task_def, config)

            # Should stop after first claude call
            assert result.step_stopped is True
            assert result.success is True
            assert "Step mode" in result.summary
            # Only task_1's instruction should have been attempted
            assert len(result.task_results) == 1
            assert result.task_results[0].task_id == "task_1"


class TestStepModeLoopB:
    """Test --step mode for Loop B: stops after each claude code call.

    New flow: Pre-verify -> Task gen -> Loop C -> (next iter pre-verify) -> ...
    """

    @pytest.mark.asyncio
    async def test_step_mode_stops_after_pre_verification(
        self, sample_requirements: RequirementDefinition
    ) -> None:
        """Step mode should stop after pre-verification phase (first claude calls)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create requirements file
            req_path = Path(temp_dir) / "requirements.yaml"
            req_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n  - id: req_2\n    name: Test2\n    acceptance_criteria: [Done]\n"
            )

            tasks_dir = Path(temp_dir) / "tasks"
            config = RequirementsOrchestratorConfig(
                max_iterations=3,
                tasks_output_dir=str(tasks_dir),
                orchestrator_config=OrchestratorConfig(
                    stream_output=False,
                    step_mode=True,
                ),
            )

            manager = LoopBHistoryManager(temp_dir)
            orchestrator = RequirementsOrchestrator(
                requirements=sample_requirements,
                config=config,
                history_manager=manager,
                requirements_path=str(req_path),
            )

            call_count = 0

            async def mock_query(prompt, cfg, phase="", **kwargs):
                nonlocal call_count
                call_count += 1
                # Set _step_executed like the real function does
                cfg._step_executed = True
                return make_single_verify_response("req_1", False)

            # Mock run_claude_query for pre-verification
            with patch(
                "task_prompt_orchestrator.loopb_verification.run_claude_query",
                new_callable=AsyncMock,
                side_effect=mock_query,
            ):
                with patch(
                    "task_prompt_orchestrator.loopb_tasks.run_claude_query",
                    new_callable=AsyncMock,
                    side_effect=mock_query,
                ):
                    result = await orchestrator.run()

            # Should stop after first pre-verification call (step mode)
            assert result.status not in {LoopBStatus.COMPLETED, LoopBStatus.FAILED}
            # Pre-verification should have been called once before StepModeStop
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_step_mode_stops_after_task_generation(
        self, sample_requirements: RequirementDefinition
    ) -> None:
        """Step mode should stop after task generation phase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create requirements file
            req_path = Path(temp_dir) / "requirements.yaml"
            req_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n  - id: req_2\n    name: Test2\n    acceptance_criteria: [Done]\n"
            )

            tasks_dir = Path(temp_dir) / "tasks"
            config = RequirementsOrchestratorConfig(
                max_iterations=3,
                tasks_output_dir=str(tasks_dir),
                orchestrator_config=OrchestratorConfig(
                    stream_output=False,
                    step_mode=False,  # Enable only for task generation
                ),
            )

            manager = LoopBHistoryManager(temp_dir)
            orchestrator = RequirementsOrchestrator(
                requirements=sample_requirements,
                config=config,
                history_manager=manager,
                requirements_path=str(req_path),
            )

            call_count = {"exploration": 0, "verification": 0, "task_gen": 0}

            async def mock_claude_query(prompt, cfg, phase="", **kwargs):
                if phase == "exploration":
                    call_count["exploration"] += 1
                    return make_exploration_response()
                elif phase == "verification":
                    call_count["verification"] += 1
                    req_id = "req_1" if call_count["verification"] == 1 else "req_2"
                    return make_single_verify_response(req_id, False)
                else:
                    call_count["task_gen"] += 1
                    # Enable step mode for task generation
                    cfg.step_mode = True
                    cfg._step_executed = True
                    return make_task_response()

            with patch(
                "task_prompt_orchestrator.loopb_verification.run_claude_query",
                new_callable=AsyncMock,
                side_effect=mock_claude_query,
            ):
                with patch(
                    "task_prompt_orchestrator.loopb_tasks.run_claude_query",
                    new_callable=AsyncMock,
                    side_effect=mock_claude_query,
                ):
                    result = await orchestrator.run()

            # Should stop after task generation (StepModeStop raised)
            assert result.status not in {LoopBStatus.COMPLETED, LoopBStatus.FAILED}
            # Exploration once, pre-verification called for both reqs, task gen called once
            assert call_count["exploration"] == 1
            assert call_count["verification"] == 2
            assert call_count["task_gen"] == 1

    @pytest.mark.asyncio
    async def test_step_mode_stops_after_loop_c_step(
        self, sample_requirements: RequirementDefinition
    ) -> None:
        """Step mode should stop when Loop C stops due to step mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create requirements file
            req_path = Path(temp_dir) / "requirements.yaml"
            req_path.write_text(
                "requirements:\n  - id: req_1\n    name: Test\n    acceptance_criteria: [Done]\n  - id: req_2\n    name: Test2\n    acceptance_criteria: [Done]\n"
            )

            tasks_dir = Path(temp_dir) / "tasks"
            config = RequirementsOrchestratorConfig(
                max_iterations=3,
                tasks_output_dir=str(tasks_dir),
                orchestrator_config=OrchestratorConfig(
                    stream_output=False,
                    step_mode=False,
                ),
            )

            manager = LoopBHistoryManager(temp_dir)
            orchestrator = RequirementsOrchestrator(
                requirements=sample_requirements,
                config=config,
                history_manager=manager,
                requirements_path=str(req_path),
            )

            call_count = {
                "exploration": 0,
                "verification": 0,
                "task_gen": 0,
                "loop_c": 0,
            }

            async def mock_claude_query(prompt, cfg, phase="", **kwargs):
                if phase == "exploration":
                    call_count["exploration"] += 1
                    return make_exploration_response()
                elif phase == "verification":
                    call_count["verification"] += 1
                    req_id = "req_1" if call_count["verification"] == 1 else "req_2"
                    return make_single_verify_response(req_id, False)
                else:
                    call_count["task_gen"] += 1
                    return make_task_response()

            async def mock_run_orchestrator(*args, **kwargs):
                call_count["loop_c"] += 1
                # Simulate Loop C stopping due to step mode
                return OrchestratorResult(
                    success=True,
                    task_results=[
                        TaskResult(task_id="task_1", status=TaskStatus.IN_PROGRESS)
                    ],
                    total_attempts=1,
                    summary="Step mode: stopped",
                    step_stopped=True,
                )

            with patch(
                "task_prompt_orchestrator.loopb_verification.run_claude_query",
                new_callable=AsyncMock,
                side_effect=mock_claude_query,
            ):
                with patch(
                    "task_prompt_orchestrator.loopb_tasks.run_claude_query",
                    new_callable=AsyncMock,
                    side_effect=mock_claude_query,
                ):
                    with patch(
                        "task_prompt_orchestrator.loopb_tasks.run_orchestrator",
                        new_callable=AsyncMock,
                        side_effect=mock_run_orchestrator,
                    ):
                        result = await orchestrator.run()

            # Should stop after Loop C returns step_stopped
            assert result.status not in {LoopBStatus.COMPLETED, LoopBStatus.FAILED}
            # Exploration once, pre-verification for both reqs, task gen, and Loop C should have been called
            assert call_count["exploration"] == 1
            assert call_count["verification"] == 2
            assert call_count["task_gen"] == 1
            assert call_count["loop_c"] == 1
