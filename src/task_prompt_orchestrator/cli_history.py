"""CLI functions for history command (Loop C)."""

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .orchestrator import HistoryManager
from .schema import ExecutionHistory

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


# =============================================================================
# History Formatting
# =============================================================================


def format_history_summary(history: ExecutionHistory) -> str:
    """Format a history entry as one-line summary for list view."""
    status_icon = "✅" if history.completed else "🔄"
    completed = len(history.completed_task_ids)
    total = len(history.task_results) + (
        1
        if history.current_task_id
        and history.current_task_id not in history.completed_task_ids
        else 0
    )
    total = max(total, completed)

    current = ""
    if history.current_task_id and not history.completed:
        current = f" @ {history.current_task_id}_{history.current_phase or '?'}"

    return f"{status_icon} {history.history_id}  [{completed}/{total}]{current}"


def _format_datetime_loopc(dt_str: str) -> str:
    """Format ISO datetime string to readable format."""
    try:
        return dt_str.replace("T", " ").split(".")[0]
    except (ValueError, IndexError):
        return dt_str


def format_history_detail(history: ExecutionHistory) -> str:
    """Format a history entry with timeline visualization."""
    status_icon = "✅" if history.completed else "🔄"
    status = "COMPLETED" if history.completed else "IN_PROGRESS"

    # Shorten file path for display
    yaml_path = history.yaml_path
    with contextlib.suppress(ValueError):
        yaml_path = str(Path(yaml_path).relative_to(Path.cwd()))

    separator = "━" * 60

    lines = [
        separator,
        f"{status_icon} LOOP C: {history.history_id}",
        separator,
        f"  File:     {yaml_path}",
        f"  Status:   {status}",
        f"  Started:  {_format_datetime_loopc(history.started_at)}",
        f"  Updated:  {_format_datetime_loopc(history.updated_at)}",
    ]

    if history.error:
        lines.append(f"  Error:    {history.error}")

    # Execution Timeline
    lines.append("")
    lines.append("━━ Execution Timeline " + "━" * 38)
    lines.append("")

    completed_ids = set(history.completed_task_ids)
    total_tasks = len(history.task_results) + (
        1
        if history.current_task_id and history.current_task_id not in completed_ids
        else 0
    )
    total_tasks = max(total_tasks, len(completed_ids))

    # Show completed tasks
    for i, tr in enumerate(history.task_results, 1):
        is_approved = tr.status.value == "approved"
        status_mark = "✅" if is_approved else "❌"
        phase_icon = "📋" if is_approved else "🔍"
        lines.append(f"{phase_icon} [{i}/{total_tasks}] {tr.task_id}")
        lines.append(
            f"  └─ {status_mark} {tr.status.value.upper()} (attempts: {tr.attempts})"
        )
        if tr.error:
            error_short = tr.error[:60] + "..." if len(tr.error) > 60 else tr.error
            lines.append(f"       └─ Error: {error_short}")

    # Show current task if in progress
    if history.current_task_id and not history.completed:
        task_num = len(history.task_results) + 1
        phase = history.current_phase or "unknown"
        phase_icon = "📋" if phase == "instruction" else "🔍"
        lines.append(
            f"{phase_icon} [{task_num}/{total_tasks}] {history.current_task_id} ← current"
        )
        lines.append(f"  └─ 🔄 {phase.upper()} in progress")

    if not history.task_results and not history.current_task_id:
        lines.append("  (not started)")

    # Summary
    lines.append("")
    lines.append(f"Progress: {len(completed_ids)}/{total_tasks} tasks completed")
    lines.append(f"Attempts: {history.total_attempts} total")

    # Resume info
    if not history.completed:
        resume_points = history.get_resume_points()
        if resume_points:
            lines.append("")
            lines.append("Resume points:")
            for point in resume_points[:3]:
                lines.append(f"  - {point}")

    lines.append(separator)

    return "\n".join(lines)


# =============================================================================
# History Command Handlers
# =============================================================================


def _history_show(cwd: str, history_id: str, as_json: bool) -> int:
    """Handle history show subcommand - searches both Loop B and Loop C."""
    from .cli_loopb import format_loopb_history_detail
    from .loopb_history import LoopBHistoryManager
    from .schema import LoopBStatus

    # Try Loop C first
    loopc_manager = HistoryManager(cwd)
    try:
        loopc_history = loopc_manager.load_history(history_id)
        if as_json:
            print(json.dumps(loopc_history.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_history_detail(loopc_history))
            if not loopc_history.completed:
                print()
                print("To resume, use:")
                print(f"  task-orchestrator resume {loopc_history.history_id}")
        return 0
    except FileNotFoundError:
        pass

    # Try Loop B
    loopb_manager = LoopBHistoryManager(cwd)
    try:
        loopb_history = loopb_manager.load_history(history_id)
        if as_json:
            print(json.dumps(loopb_history.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_loopb_history_detail(loopb_history))
            if loopb_history.status not in {LoopBStatus.COMPLETED, LoopBStatus.FAILED}:
                print()
                print("To resume, use:")
                print(f"  task-orchestrator resume {loopb_history.history_id} --loopb")
        return 0
    except FileNotFoundError:
        pass

    logger.error(f"History not found: {history_id}")
    return 1


def _history_list_combined(cwd: str, show_all: bool, as_json: bool) -> int:
    """Handle history list showing both Loop B and Loop C."""
    from .cli_loopb import format_loopb_history_summary
    from .loopb_history import LoopBHistoryManager
    from .schema import LoopBStatus

    loopc_manager = HistoryManager(cwd)
    loopb_manager = LoopBHistoryManager(cwd)

    loopc_histories = (
        loopc_manager.list_histories()
        if show_all
        else loopc_manager.list_incomplete_histories()
    )
    loopb_histories = (
        loopb_manager.list_histories()
        if show_all
        else [
            h
            for h in loopb_manager.list_histories()
            if h.status not in {LoopBStatus.COMPLETED, LoopBStatus.FAILED}
        ]
    )

    if not loopc_histories and not loopb_histories:
        msg = (
            "No execution history found."
            if show_all
            else "No incomplete executions found. Use --all to show completed ones."
        )
        print(msg)
        return 0

    if as_json:
        combined = {
            "loopb": [h.to_dict() for h in loopb_histories],
            "loopc": [h.to_dict() for h in loopc_histories],
        }
        print(json.dumps(combined, indent=2, ensure_ascii=False))
    else:
        title = "All execution history:" if show_all else "Incomplete executions:"
        print(title)
        if loopb_histories:
            print("\n[Loop B - Requirements]")
            for loopb_h in loopb_histories:
                print(format_loopb_history_summary(loopb_h))
        if loopc_histories:
            print("\n[Loop C - Tasks]")
            for loopc_h in loopc_histories:
                print(format_history_summary(loopc_h))
    return 0


def _history_list(
    history_manager: HistoryManager, show_all: bool, as_json: bool
) -> int:
    """Handle history list subcommand (Loop C only)."""
    histories = (
        history_manager.list_histories()
        if show_all
        else history_manager.list_incomplete_histories()
    )
    if not histories:
        msg = (
            "No Loop C execution history found."
            if show_all
            else "No incomplete Loop C executions found. Use --all to show completed ones."
        )
        print(msg)
        return 0
    if as_json:
        print(
            json.dumps([h.to_dict() for h in histories], indent=2, ensure_ascii=False)
        )
    else:
        title = (
            "All execution history:"
            if show_all
            else "Incomplete executions (resumable):"
        )
        print(title)
        for history in histories:
            print(format_history_summary(history))
    return 0


def _history_by_file(
    history_manager: HistoryManager, file_path: str, as_json: bool, show_all: bool
) -> int:
    """Show history for a specific file path."""
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return 1

    histories = history_manager.find_by_path(file_path, include_completed=show_all)
    if not histories:
        msg = (
            f"No history found for: {file_path}"
            if show_all
            else f"No incomplete history found for: {file_path}. Use --all to include completed."
        )
        print(msg)
        return 0

    if len(histories) == 1:
        # Single match: show details
        if as_json:
            print(json.dumps(histories[0].to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_history_detail(histories[0]))
    elif as_json:
        # Multiple matches: show list as JSON
        print(
            json.dumps([h.to_dict() for h in histories], indent=2, ensure_ascii=False)
        )
    else:
        # Multiple matches: show list
        print(f"Found {len(histories)} histories for: {file_path}")
        for history in histories:
            print(format_history_summary(history))
    return 0


def history_command(args: "argparse.Namespace") -> int:
    """Execute the history command."""
    from .cli_loopb import handle_loopb_history

    cwd = args.cwd or str(Path.cwd())

    # --loopb-children requires explicit Loop B mode
    if args.loopb_children:
        return handle_loopb_history(args, cwd, format_history_summary)

    # --loopb forces Loop B only mode
    if args.loopb:
        return handle_loopb_history(args, cwd, format_history_summary)

    # Positional argument: could be file path or history ID
    if args.file:
        if Path(args.file).exists():
            # It's a file path - search by path
            history_manager = HistoryManager(cwd)
            return _history_by_file(history_manager, args.file, args.json, args.all)
        # Treat as history ID
        return _history_show(cwd, args.file, args.json)

    if args.show:
        return _history_show(cwd, args.show, args.json)

    # Default: show both Loop B and Loop C
    return _history_list_combined(cwd, args.all, args.json)
