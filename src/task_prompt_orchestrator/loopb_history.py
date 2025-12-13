"""Loop B history management - history persistence and configuration."""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .orchestrator import (
    OrchestratorConfig,
    delete_history_file,
    list_history_files,
    load_history_from_file,
    save_history_to_file,
)
from .schema import (
    LoopBExecutionHistory,
    LoopBStatus,
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
