"""CLI entry point for task prompt orchestrator."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import anyio

from .cli_loopb import (
    handle_loopb_history,
    output_loopb_result,
    run_loopb_orchestrator,
)
from .orchestrator import (
    HistoryManager,
    OrchestratorConfig,
    build_instruction_prompt,
    build_validation_prompt,
    run_orchestrator,
)
from .schema import (
    ExecutionHistory,
    ExecutionPhase,
    OrchestratorResult,
    ResumePoint,
    TaskDefinition,
    YamlType,
    create_sample_task_yaml,
    detect_yaml_type,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parser Helpers
# =============================================================================


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by run/resume commands."""
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for Claude Code execution",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (e.g., claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for results JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable WebFetch and WebSearch tools",
    )
    parser.add_argument(
        "--bypass-permissions",
        action="store_true",
        help="Use bypassPermissions mode (use with caution)",
    )


def _add_retry_args(parser: argparse.ArgumentParser) -> None:
    """Add retry-related arguments."""
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per task (default: 3)",
    )
    parser.add_argument(
        "--max-total-retries",
        type=int,
        default=10,
        help="Max total retries across all tasks (default: 10)",
    )


def _add_loopb_flag(parser: argparse.ArgumentParser) -> None:
    """Add --loopb flag to switch between Loop C and Loop B."""
    parser.add_argument(
        "--loopb",
        action="store_true",
        help="Force Loop B mode (auto-detected from file content if omitted)",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Task Prompt Orchestrator - Automate multi-step Claude Code tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run tasks from YAML file (Loop C)
  task-orchestrator run tasks.yaml

  # Run requirements-driven execution (Loop B)
  task-orchestrator run requirements.yaml --loopb

  # Run with custom settings
  task-orchestrator run tasks.yaml --max-retries 5 --cwd /path/to/project

  # Resume from a specific point (Loop C)
  task-orchestrator resume <history_id> --from task2_validation

  # Resume Loop B execution
  task-orchestrator resume <history_id> --loopb

  # List execution history
  task-orchestrator history
  task-orchestrator history --loopb

  # Generate sample YAML
  task-orchestrator sample > sample_tasks.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command (unified for Loop C and Loop B)
    run_parser = subparsers.add_parser(
        "run", help="Run tasks/requirements from YAML file"
    )
    run_parser.add_argument(
        "input_file", type=str, help="Path to task or requirements YAML"
    )
    _add_loopb_flag(run_parser)
    _add_common_args(run_parser)
    _add_retry_args(run_parser)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution flow and prompts without executing",
    )

    # Resume command (unified for Loop C and Loop B)
    resume_parser = subparsers.add_parser(
        "resume", help="Resume execution from history"
    )
    resume_parser.add_argument("history_id", type=str, help="History ID to resume")
    _add_loopb_flag(resume_parser)
    _add_common_args(resume_parser)
    # Loop C specific
    resume_parser.add_argument(
        "--from",
        dest="resume_from",
        type=str,
        default=None,
        help="Resume point (e.g., task2_instruction, task3_validation). Loop C only.",
    )

    # History command
    history_parser = subparsers.add_parser("history", help="List execution history")
    _add_loopb_flag(history_parser)
    history_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all history (including completed)",
    )
    history_parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory to look for history",
    )
    history_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    history_parser.add_argument(
        "--show",
        type=str,
        default=None,
        metavar="HISTORY_ID",
        help="Show details of a specific history entry",
    )
    history_parser.add_argument(
        "--loopb-children",
        type=str,
        default=None,
        metavar="LOOPB_HISTORY_ID",
        help="Show Loop C histories for a specific Loop B execution",
    )

    # Sample command
    subparsers.add_parser("sample", help="Print sample YAML to stdout")

    return parser.parse_args()


# =============================================================================
# Config Builder
# =============================================================================


def build_orchestrator_config(args: argparse.Namespace) -> OrchestratorConfig:
    """Build OrchestratorConfig from parsed arguments."""
    cwd = args.cwd or str(Path.cwd())

    allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"]
    if not args.no_web:
        allowed_tools.extend(["WebFetch", "WebSearch"])

    permission_mode = "bypassPermissions" if args.bypass_permissions else "acceptEdits"

    return OrchestratorConfig(
        max_retries_per_task=args.max_retries,
        max_total_retries=args.max_total_retries,
        cwd=cwd,
        model=args.model,
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
    )


# =============================================================================
# Output and Formatting
# =============================================================================


def print_dry_run(task_def: TaskDefinition, config: OrchestratorConfig) -> None:
    """Print dry run information."""
    print("=" * 70)
    print("DRY RUN - Task Prompt Orchestrator")
    print("=" * 70)
    print()

    print("## Configuration")
    print(f"  Working directory:    {config.cwd}")
    print(f"  Model:                {config.model or '(default)'}")
    print(f"  Max retries per task: {config.max_retries_per_task}")
    print(f"  Max total retries:    {config.max_total_retries}")
    print(f"  Permission mode:      {config.permission_mode}")
    print(f"  Allowed tools:        {', '.join(config.allowed_tools)}")
    if config.permission_mode == "bypassPermissions":
        print("  WARNING: bypassPermissions allows all tools without prompts")
    print()

    print(f"## Tasks ({len(task_def.tasks)} total)")
    print()

    for i, task in enumerate(task_def.tasks, start=1):
        print("-" * 70)
        print(f"### Task [{i}/{len(task_def.tasks)}]: {task.name}")
        print(f"    ID: {task.id}")
        if task.depends_on:
            print(f"    Depends on: {', '.join(task.depends_on)}")
        print()

        print("#### INSTRUCTION PROMPT:")
        print("```")
        instruction = build_instruction_prompt(task, cwd=config.cwd)
        print(instruction)
        print("```")
        print()

        print("#### VALIDATION PROMPT (template):")
        print("```")
        validation = build_validation_prompt(task, "<instruction output will be here>")
        print(validation)
        print("```")
        print()

    print("=" * 70)
    print("END DRY RUN")
    print("=" * 70)


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


def format_history_detail(history: ExecutionHistory) -> str:
    """Format a history entry with full details."""
    status = "completed" if history.completed else "incomplete"
    status_icon = "✅" if history.completed else "🔄"

    lines = [
        f"{status_icon} {history.history_id}",
        "",
        "Configuration:",
        f"  YAML:       {history.yaml_path}",
        f"  Status:     {status}",
        f"  Started:    {history.started_at}",
        f"  Updated:    {history.updated_at}",
    ]

    if history.current_task_id:
        lines.append(
            f"  Current:    {history.current_task_id}_{history.current_phase or 'unknown'}"
        )

    if history.error:
        lines.append(f"  Error:      {history.error}")

    cfg = history.config_snapshot
    if cfg:
        lines.append("")
        lines.append("Saved config:")
        lines.append(f"  cwd:        {cfg.get('cwd', '(none)')}")
        lines.append(f"  model:      {cfg.get('model', '(default)')}")
        lines.append(
            f"  retries:    {cfg.get('max_retries_per_task', 3)} per task, "
            f"{cfg.get('max_total_retries', 10)} total"
        )

    lines.append("")
    lines.append("Task results:")
    if history.task_results:
        for tr in history.task_results:
            status_mark = "✓" if tr.status.value == "approved" else "✗"
            error_info = (
                f" - {tr.error[:50]}..."
                if tr.error and len(tr.error) > 50
                else (f" - {tr.error}" if tr.error else "")
            )
            lines.append(
                f"  {status_mark} {tr.task_id}: {tr.status.value} "
                f"(attempts: {tr.attempts}){error_info}"
            )
    else:
        lines.append("  (none)")

    resume_points = history.get_resume_points()
    if resume_points and not history.completed:
        lines.append("")
        lines.append("Resume points:")
        for point in resume_points:
            lines.append(f"  - {point}")

    return "\n".join(lines)


def output_results(
    result: OrchestratorResult,
    output_path: str | None,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Output orchestration results to file or stdout."""
    result_dict: dict[str, Any] = {
        "success": result.success,
        "summary": result.summary,
        "total_attempts": result.total_attempts,
        "tasks": [
            {
                "task_id": tr.task_id,
                "status": tr.status.value,
                "attempts": tr.attempts,
                "approved": tr.validation_approved,
                "error": tr.error,
            }
            for tr in result.task_results
        ],
    }
    if extra_fields:
        result_dict.update(extra_fields)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Results written to: {path}")
    else:
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))


# =============================================================================
# History Command
# =============================================================================


def _history_show(
    history_manager: HistoryManager, history_id: str, as_json: bool
) -> int:
    """Handle history show subcommand."""
    try:
        history = history_manager.load_history(history_id)
    except FileNotFoundError:
        logger.error(f"History not found: {history_id}")
        return 1
    if as_json:
        print(json.dumps(history.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_history_detail(history))
        if not history.completed:
            print()
            print("To resume, use:")
            print(f"  task-orchestrator resume {history.history_id}")
    return 0


def _history_list(
    history_manager: HistoryManager, show_all: bool, as_json: bool
) -> int:
    """Handle history list subcommand."""
    histories = (
        history_manager.list_histories()
        if show_all
        else history_manager.list_incomplete_histories()
    )
    if not histories:
        msg = (
            "No execution history found."
            if show_all
            else "No incomplete executions found. Use --all to show completed ones."
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


def history_command(args: argparse.Namespace) -> int:
    """Execute the history command."""
    cwd = args.cwd or str(Path.cwd())

    if args.loopb or args.loopb_children:
        return handle_loopb_history(args, cwd, format_history_summary)

    history_manager = HistoryManager(cwd)

    if args.show:
        return _history_show(history_manager, args.show, args.json)
    return _history_list(history_manager, args.all, args.json)


# =============================================================================
# Resume Command (Loop C)
# =============================================================================


def _determine_resume_point(
    args_resume_from: str | None,
    history: ExecutionHistory,
    task_def: TaskDefinition,
) -> ResumePoint | None:
    """Determine resume point from args, history, or task definition."""
    if args_resume_from:
        return ResumePoint.parse(args_resume_from)
    if history.current_task_id and history.current_phase:
        return ResumePoint(
            task_id=history.current_task_id,
            phase=ExecutionPhase(history.current_phase),
        )
    approved_ids = set(history.completed_task_ids)
    for task in task_def.tasks:
        if task.id not in approved_ids:
            return ResumePoint(task_id=task.id, phase=ExecutionPhase.INSTRUCTION)
    return None


def _load_resumable_history(
    history_manager: HistoryManager, history_id: str
) -> ExecutionHistory | None:
    """Load history and validate it can be resumed. Returns None on error."""
    try:
        history = history_manager.load_history(history_id)
    except FileNotFoundError:
        logger.error(f"History not found: {history_id}")
        return None
    if history.completed:
        logger.error(f"History {history_id} is already completed.")
        return None
    if not Path(history.yaml_path).exists():
        logger.error(f"YAML file not found: {history.yaml_path}")
        return None
    return history


async def resume_loopc(args: argparse.Namespace) -> int:
    """Execute the resume command for Loop C."""
    cwd = args.cwd or str(Path.cwd())
    history_manager = HistoryManager(cwd)

    history = _load_resumable_history(history_manager, args.history_id)
    if not history:
        return 1

    task_def = TaskDefinition.from_yaml(history.yaml_path)

    try:
        resume_point = _determine_resume_point(args.resume_from, history, task_def)
    except ValueError as e:
        logger.error(str(e))
        return 1
    if not resume_point:
        logger.error("Could not determine resume point.")
        return 1

    logger.info(f"Resuming from: {resume_point}")

    saved_config = history.config_snapshot
    config = OrchestratorConfig(
        max_retries_per_task=saved_config.get("max_retries_per_task", 3),
        max_total_retries=saved_config.get("max_total_retries", 10),
        cwd=args.cwd or saved_config.get("cwd") or cwd,
        model=saved_config.get("model"),
        allowed_tools=saved_config.get("allowed_tools", []),
        permission_mode=saved_config.get("permission_mode", "acceptEdits"),
    )

    result = await run_orchestrator(
        task_def,
        config,
        yaml_path=history.yaml_path,
        history_manager=history_manager,
        resume_point=resume_point,
        resume_history=history,
    )

    output_results(
        result,
        args.output,
        {"resumed_from": str(resume_point), "history_id": history.history_id},
    )

    if result.success:
        logger.info("All tasks completed successfully!")
    else:
        logger.error(f"Orchestration failed: {result.summary}")
    return 0 if result.success else 1


# =============================================================================
# Run Command (Loop C)
# =============================================================================


async def run_loopc(args: argparse.Namespace) -> int:
    """Execute the run command for Loop C."""
    yaml_path = Path(args.input_file)
    if not yaml_path.exists():
        logger.error(f"YAML file not found: {yaml_path}")
        return 1

    config = build_orchestrator_config(args)

    # Check for incomplete history to auto-resume
    history_manager = HistoryManager(config.cwd)
    existing = history_manager.find_incomplete_by_path(str(yaml_path))
    if existing:
        logger.info(f"Found incomplete execution: {existing.history_id}")
        logger.info("Auto-resuming from previous state...")
        args.history_id = existing.history_id
        args.resume_from = None
        return await resume_loopc(args)

    logger.info(f"Loading task definition from: {yaml_path}")
    task_def = TaskDefinition.from_yaml(str(yaml_path))
    logger.info(f"Loaded {len(task_def.tasks)} tasks")

    if args.dry_run:
        print_dry_run(task_def, config)
        return 0

    logger.info("Starting orchestrator...")
    result = await run_orchestrator(
        task_def,
        config,
        yaml_path=str(yaml_path),
        history_manager=history_manager,
    )

    output_results(result, args.output)

    if result.success:
        logger.info("All tasks completed successfully!")
        return 0
    logger.error(f"Orchestration failed: {result.summary}")
    return 1


# =============================================================================
# Unified Run/Resume Commands
# =============================================================================


def _detect_loop_type(args: argparse.Namespace) -> bool:
    """Detect if this should run as Loop B. Returns True for Loop B."""
    if args.loopb:
        return True

    yaml_path = Path(args.input_file)
    if not yaml_path.exists():
        return False

    yaml_type = detect_yaml_type(str(yaml_path))
    if yaml_type == YamlType.LOOP_B:
        logger.info("Detected requirements file (Loop B mode)")
        return True
    if yaml_type == YamlType.LOOP_C:
        logger.info("Detected task file (Loop C mode)")
        return False

    logger.warning("Could not detect YAML type, defaulting to Loop C")
    return False


async def run_command(args: argparse.Namespace) -> int:
    """Execute the run command (dispatches to Loop B or Loop C)."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    is_loopb = _detect_loop_type(args)

    if is_loopb:
        from .cli_loopb import print_loopb_dry_run

        if args.dry_run:
            return print_loopb_dry_run(args)
        result = await run_loopb_orchestrator(args)
        return output_loopb_result(result, args)

    return await run_loopc(args)


async def resume_command(args: argparse.Namespace) -> int:
    """Execute the resume command (dispatches to Loop B or Loop C)."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.loopb:
        from .cli_loopb import resume_loopb

        return await resume_loopb(args)

    return await resume_loopc(args)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.command == "sample":
        print(create_sample_task_yaml())
        return 0
    if args.command == "run":
        return anyio.run(run_command, args)
    if args.command == "history":
        return history_command(args)
    if args.command == "resume":
        return anyio.run(resume_command, args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
