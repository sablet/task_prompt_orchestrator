"""CLI entry point for task prompt orchestrator."""

import argparse
import json
import logging
import sys
from pathlib import Path

import anyio

from .orchestrator import (
    HistoryManager,
    OrchestratorConfig,
    build_instruction_prompt,
    build_validation_prompt,
    run_orchestrator,
)
from .schema import ExecutionHistory, ExecutionPhase, ResumePoint, TaskDefinition, create_sample_task_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Task Prompt Orchestrator - Automate multi-step Claude Code tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run tasks from YAML file
  task-orchestrator run tasks.yaml

  # Run with custom settings
  task-orchestrator run tasks.yaml --max-retries 5 --cwd /path/to/project

  # Resume from a specific point
  task-orchestrator resume <history_id> --from task2_validation

  # List execution history
  task-orchestrator history

  # Generate sample YAML
  task-orchestrator sample > sample_tasks.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run tasks from YAML file")
    run_parser.add_argument("yaml_file", type=str, help="Path to task definition YAML")
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per task (default: 3)",
    )
    run_parser.add_argument(
        "--max-total-retries",
        type=int,
        default=10,
        help="Max total retries across all tasks (default: 10)",
    )
    run_parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for Claude Code execution",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (e.g., claude-sonnet-4-20250514)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for results JSON",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    run_parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable WebFetch and WebSearch tools (reduce cost)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show prompts and config without executing",
    )
    run_parser.add_argument(
        "--bypass-permissions",
        action="store_true",
        help="Use bypassPermissions mode (use with caution)",
    )
    run_parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable history tracking",
    )

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume execution from history")
    resume_parser.add_argument("history_id", type=str, help="History ID to resume")
    resume_parser.add_argument(
        "--from",
        dest="resume_from",
        type=str,
        default=None,
        help="Resume point (e.g., task2_instruction, task3_validation). If not specified, resumes from where it stopped.",
    )
    resume_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for results JSON",
    )
    resume_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    resume_parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory (overrides saved config)",
    )

    # History command
    history_parser = subparsers.add_parser("history", help="List execution history")
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
        "--delete",
        type=str,
        default=None,
        metavar="HISTORY_ID",
        help="Delete a specific history entry",
    )

    # Sample command
    subparsers.add_parser("sample", help="Print sample YAML to stdout")

    return parser.parse_args()


def print_dry_run(task_def: TaskDefinition, config: OrchestratorConfig) -> None:
    """Print dry run information."""
    print("=" * 70)
    print("DRY RUN - Task Prompt Orchestrator")
    print("=" * 70)
    print()

    # Config section
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

    # Tasks section
    print(f"## Tasks ({len(task_def.tasks)} total)")
    print()

    for i, task in enumerate(task_def.tasks, start=1):
        print("-" * 70)
        print(f"### Task [{i}/{len(task_def.tasks)}]: {task.name}")
        print(f"    ID: {task.id}")
        if task.depends_on:
            print(f"    Depends on: {', '.join(task.depends_on)}")
        print()

        # Instruction prompt
        print("#### INSTRUCTION PROMPT:")
        print("```")
        instruction = build_instruction_prompt(task, cwd=config.cwd)
        print(instruction)
        print("```")
        print()

        # Validation prompt (with placeholder for instruction output)
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
    total = len(history.task_results) + (1 if history.current_task_id and history.current_task_id not in history.completed_task_ids else 0)
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
        lines.append(f"  Current:    {history.current_task_id}_{history.current_phase or 'unknown'}")

    if history.error:
        lines.append(f"  Error:      {history.error}")

    # Config snapshot
    cfg = history.config_snapshot
    if cfg:
        lines.append("")
        lines.append("Saved config:")
        lines.append(f"  cwd:        {cfg.get('cwd', '(none)')}")
        lines.append(f"  model:      {cfg.get('model', '(default)')}")
        lines.append(f"  retries:    {cfg.get('max_retries_per_task', 3)} per task, {cfg.get('max_total_retries', 10)} total")

    # Task results
    lines.append("")
    lines.append("Task results:")
    if history.task_results:
        for tr in history.task_results:
            status_mark = "✓" if tr.status.value == "approved" else "✗"
            error_info = f" - {tr.error[:50]}..." if tr.error and len(tr.error) > 50 else (f" - {tr.error}" if tr.error else "")
            lines.append(f"  {status_mark} {tr.task_id}: {tr.status.value} (attempts: {tr.attempts}){error_info}")
    else:
        lines.append("  (none)")

    # Resume points
    resume_points = history.get_resume_points()
    if resume_points and not history.completed:
        lines.append("")
        lines.append("Resume points:")
        for point in resume_points:
            lines.append(f"  - {point}")

    return "\n".join(lines)


def history_command(args: argparse.Namespace) -> int:
    """Execute the history command."""
    cwd = args.cwd or str(Path.cwd())
    history_manager = HistoryManager(cwd)

    # Handle delete
    if args.delete:
        if history_manager.delete_history(args.delete):
            print(f"Deleted history: {args.delete}")
            return 0
        logger.error(f"History not found: {args.delete}")
        return 1

    # Handle show single entry
    if args.show:
        try:
            history = history_manager.load_history(args.show)
            if args.json:
                print(json.dumps(history.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(format_history_detail(history))
                if not history.completed:
                    print()
                    print("To resume, use:")
                    print(f"  task-orchestrator resume {history.history_id}")
            return 0
        except FileNotFoundError:
            logger.error(f"History not found: {args.show}")
            return 1

    # List histories
    if args.all:
        histories = history_manager.list_histories()
    else:
        histories = history_manager.list_incomplete_histories()

    if not histories:
        if args.all:
            print("No execution history found.")
        else:
            print("No incomplete executions found. Use --all to show completed ones.")
        return 0

    if args.json:
        print(json.dumps([h.to_dict() for h in histories], indent=2, ensure_ascii=False))
    else:
        title = "All execution history:" if args.all else "Incomplete executions (resumable):"
        print(title)
        for history in histories:
            print(format_history_summary(history))

    return 0


async def resume_command(args: argparse.Namespace) -> int:
    """Execute the resume command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cwd = args.cwd or str(Path.cwd())
    history_manager = HistoryManager(cwd)

    try:
        history = history_manager.load_history(args.history_id)
    except FileNotFoundError:
        logger.error(f"History not found: {args.history_id}")
        return 1

    if history.completed:
        logger.error(f"History {args.history_id} is already completed. Nothing to resume.")
        return 1

    # Load task definition
    yaml_path = history.yaml_path
    if not Path(yaml_path).exists():
        logger.error(f"YAML file not found: {yaml_path}")
        return 1

    task_def = TaskDefinition.from_yaml(yaml_path)

    # Determine resume point
    resume_point: ResumePoint | None = None
    if args.resume_from:
        try:
            resume_point = ResumePoint.parse(args.resume_from)
        except ValueError as e:
            logger.error(str(e))
            return 1
    else:
        # Auto-determine resume point from history
        if history.current_task_id and history.current_phase:
            resume_point = ResumePoint(
                task_id=history.current_task_id,
                phase=ExecutionPhase(history.current_phase),
            )
        else:
            # Find the first non-approved task
            approved_ids = set(history.completed_task_ids)
            for task in task_def.tasks:
                if task.id not in approved_ids:
                    resume_point = ResumePoint(
                        task_id=task.id,
                        phase=ExecutionPhase.INSTRUCTION,
                    )
                    break

    if not resume_point:
        logger.error("Could not determine resume point.")
        return 1

    logger.info(f"Resuming from: {resume_point}")

    # Restore config from history
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
        yaml_path=yaml_path,
        history_manager=history_manager,
        resume_point=resume_point,
        resume_history=history,
    )

    # Output results
    result_dict = {
        "success": result.success,
        "summary": result.summary,
        "total_attempts": result.total_attempts,
        "resumed_from": str(resume_point),
        "history_id": history.history_id,
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

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Results written to: {output_path}")
    else:
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    if result.success:
        logger.info("All tasks completed successfully!")
        return 0
    logger.error(f"Orchestration failed: {result.summary}")
    return 1


async def run_command(args: argparse.Namespace) -> int:
    """Execute the run command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    yaml_path = Path(args.yaml_file)
    if not yaml_path.exists():
        logger.error(f"YAML file not found: {yaml_path}")
        return 1

    logger.info(f"Loading task definition from: {yaml_path}")
    task_def = TaskDefinition.from_yaml(str(yaml_path))
    logger.info(f"Loaded {len(task_def.tasks)} tasks")

    # Default cwd to current directory if not specified
    cwd = args.cwd or str(Path.cwd())

    # Build allowed_tools list
    allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"]
    if not args.no_web:
        allowed_tools.extend(["WebFetch", "WebSearch"])

    permission_mode = "bypassPermissions" if args.bypass_permissions else "acceptEdits"

    config = OrchestratorConfig(
        max_retries_per_task=args.max_retries,
        max_total_retries=args.max_total_retries,
        cwd=cwd,
        model=args.model,
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
    )

    # Handle dry run
    if args.dry_run:
        print_dry_run(task_def, config)
        return 0

    # Setup history manager
    history_manager: HistoryManager | None = None
    if not args.no_history:
        history_manager = HistoryManager(cwd)

    logger.info("Starting orchestrator...")
    result = await run_orchestrator(
        task_def,
        config,
        yaml_path=str(yaml_path),
        history_manager=history_manager,
    )

    # Output results
    result_dict = {
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

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Results written to: {output_path}")
    else:
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    if result.success:
        logger.info("All tasks completed successfully!")
        return 0
    logger.error(f"Orchestration failed: {result.summary}")
    return 1


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
