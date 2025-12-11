"""CLI functions for Loop B (requirements orchestrator)."""

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .orchestrator import HistoryManager, OrchestratorConfig
from .requirements_orchestrator import (
    LoopBHistoryManager,
    RequirementsOrchestrator,
    RequirementsOrchestratorConfig,
)
from .schema import (
    LoopBExecutionHistory,
    LoopBStatus,
    RequirementDefinition,
)

logger = logging.getLogger(__name__)


def format_loopb_history_summary(history: LoopBExecutionHistory) -> str:
    """Format a Loop B history entry as one-line summary."""
    status_icon = {
        LoopBStatus.COMPLETED: "✅",
        LoopBStatus.FAILED: "❌",
    }.get(history.status, "🔄")

    iterations = len(history.iterations)
    completed = len(history.completed_task_ids)

    return (
        f"{status_icon} {history.history_id}  "
        f"[iter:{iterations}/{history.max_iterations}] "
        f"[tasks:{completed}] "
        f"{history.status.value}"
    )


def format_loopb_history_detail(history: LoopBExecutionHistory) -> str:
    """Format a Loop B history entry with full details."""
    status_icon = {
        LoopBStatus.COMPLETED: "✅",
        LoopBStatus.FAILED: "❌",
    }.get(history.status, "🔄")

    lines = [
        f"{status_icon} {history.history_id}",
        "",
        "Configuration:",
        f"  Requirements: {history.requirements_path}",
        f"  Tasks output: {history.tasks_output_dir}",
        f"  Status:       {history.status.value}",
        f"  Iterations:   {history.current_iteration}/{history.max_iterations}",
        f"  Started:      {history.started_at}",
        f"  Updated:      {history.updated_at}",
    ]

    if history.error:
        lines.append(f"  Error:        {history.error}")

    # Iterations
    lines.append("")
    lines.append("Iterations:")
    if history.iterations:
        for it in history.iterations:
            met = (
                "✓"
                if it.verification_result
                and it.verification_result.get("all_requirements_met")
                else "✗"
            )
            lines.append(f"  {it.iteration_number}. {met} tasks: {it.tasks_yaml_path}")
            if it.loop_c_history_id:
                lines.append(f"     Loop C: {it.loop_c_history_id}")
            if it.unmet_requirements:
                lines.append(f"     Unmet: {', '.join(it.unmet_requirements)}")
    else:
        lines.append("  (none)")

    # Loop C histories
    if history.loop_c_history_ids:
        lines.append("")
        lines.append("Loop C histories:")
        for hid in history.loop_c_history_ids:
            lines.append(f"  - {hid}")

    # Completed tasks
    if history.completed_task_ids:
        lines.append("")
        lines.append(f"Completed tasks ({len(history.completed_task_ids)}):")
        for tid in history.completed_task_ids[:10]:
            lines.append(f"  - {tid}")
        if len(history.completed_task_ids) > 10:
            lines.append(f"  ... and {len(history.completed_task_ids) - 10} more")

    return "\n".join(lines)


def handle_loopb_history_delete(
    args: argparse.Namespace, history_manager: LoopBHistoryManager
) -> int:
    """Handle 'history --loopb --delete' subcommand."""
    if history_manager.delete_history(args.delete):
        print(f"Deleted Loop B history: {args.delete}")
        return 0
    logger.error(f"Loop B history not found: {args.delete}")
    return 1


def handle_loopb_history_show(
    args: argparse.Namespace, history_manager: LoopBHistoryManager
) -> int:
    """Handle 'history --loopb --show' subcommand."""
    try:
        history = history_manager.load_history(args.show)
    except FileNotFoundError:
        logger.error(f"Loop B history not found: {args.show}")
        return 1

    if args.json:
        print(json.dumps(history.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_loopb_history_detail(history))
    return 0


def handle_loopb_history_list(
    args: argparse.Namespace, history_manager: LoopBHistoryManager
) -> int:
    """Handle 'history --loopb' list subcommand."""
    histories = (
        history_manager.list_histories()
        if args.all
        else history_manager.list_incomplete_histories()
    )

    if not histories:
        msg = (
            "No Loop B execution history found."
            if args.all
            else "No incomplete Loop B executions found. Use --all to show completed."
        )
        print(msg)
        return 0

    if args.json:
        print(
            json.dumps([h.to_dict() for h in histories], indent=2, ensure_ascii=False)
        )
    else:
        title = (
            "All Loop B execution history:"
            if args.all
            else "Incomplete Loop B executions:"
        )
        print(title)
        for history in histories:
            print(format_loopb_history_summary(history))

    return 0


def handle_loopb_children(
    args: argparse.Namespace,
    loopb_manager: LoopBHistoryManager,
    loopc_manager: HistoryManager,
    format_loopc_summary: Callable[[Any], str],
) -> int:
    """Handle 'history --loopb-children' subcommand."""
    try:
        loopb_history = loopb_manager.load_history(args.loopb_children)
    except FileNotFoundError:
        logger.error(f"Loop B history not found: {args.loopb_children}")
        return 1

    if not loopb_history.loop_c_history_ids:
        print(f"No Loop C histories found for {args.loopb_children}")
        return 0

    print(f"Loop C histories for {args.loopb_children}:")
    for hid in loopb_history.loop_c_history_ids:
        try:
            loopc_history = loopc_manager.load_history(hid)
            print(format_loopc_summary(loopc_history))
        except FileNotFoundError:
            print(f"  ⚠ {hid} (not found)")

    return 0


def handle_loopb_history(
    args: argparse.Namespace, cwd: str, format_loopc_summary: Callable[[Any], str]
) -> int:
    """Handle Loop B history subcommands."""
    loopb_manager = LoopBHistoryManager(cwd)

    if args.loopb_children:
        loopc_manager = HistoryManager(cwd)
        return handle_loopb_children(
            args, loopb_manager, loopc_manager, format_loopc_summary
        )

    if args.delete:
        return handle_loopb_history_delete(args, loopb_manager)

    if args.show:
        return handle_loopb_history_show(args, loopb_manager)

    return handle_loopb_history_list(args, loopb_manager)


def build_loopb_result_dict(history: LoopBExecutionHistory) -> dict[str, object]:
    """Build the result dictionary for Loop B output."""
    return {
        "success": history.status == LoopBStatus.COMPLETED,
        "status": history.status.value,
        "history_id": history.history_id,
        "iterations": history.current_iteration,
        "max_iterations": history.max_iterations,
        "completed_task_ids": history.completed_task_ids,
        "loop_c_history_ids": history.loop_c_history_ids,
        "final_result": history.final_result,
        "error": history.error,
    }


def output_loopb_result(
    history: LoopBExecutionHistory, args: argparse.Namespace
) -> int:
    """Output the Loop B result and return exit code."""
    # Import here to avoid circular import
    from .cli import write_json_result

    result_dict = build_loopb_result_dict(history)
    output_path = Path(args.output) if args.output else None
    write_json_result(result_dict, output_path)

    if history.status == LoopBStatus.COMPLETED:
        logger.info("All requirements fulfilled!")
        return 0

    logger.error(f"Loop B failed: {history.error or history.status.value}")
    return 1


async def run_requirements_command(args: argparse.Namespace) -> int:
    """Execute the run-requirements command (Loop B)."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    requirements_path = Path(args.requirements_file)
    if not requirements_path.exists():
        logger.error(f"Requirements file not found: {requirements_path}")
        return 1

    logger.info(f"Loading requirements from: {requirements_path}")
    requirements = RequirementDefinition.from_yaml(str(requirements_path))
    logger.info(f"Loaded {len(requirements.requirements)} requirements")

    cwd = args.cwd or str(Path.cwd())

    allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"]
    if not args.no_web:
        allowed_tools.extend(["WebFetch", "WebSearch"])

    permission_mode = "bypassPermissions" if args.bypass_permissions else "acceptEdits"

    orchestrator_config = OrchestratorConfig(
        max_retries_per_task=args.max_retries,
        max_total_retries=args.max_total_retries,
        cwd=cwd,
        model=args.model,
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
    )

    config = RequirementsOrchestratorConfig(
        max_iterations=args.max_iterations,
        tasks_output_dir=args.tasks_output_dir,
        orchestrator_config=orchestrator_config,
    )

    history_manager = LoopBHistoryManager(cwd)

    orchestrator = RequirementsOrchestrator(
        requirements=requirements,
        config=config,
        history_manager=history_manager,
        requirements_path=str(requirements_path),
    )

    logger.info("Starting Loop B orchestrator...")
    result = await orchestrator.run()

    return output_loopb_result(result, args)
