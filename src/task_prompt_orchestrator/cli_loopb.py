"""CLI functions for Loop B (requirements orchestrator)."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .orchestrator import HistoryManager, OrchestratorConfig
from .prompts import build_task_generation_prompt
from .requirements_orchestrator import (
    LoopBHistoryManager,
    RequirementsOrchestrator,
    RequirementsOrchestratorConfig,
    compute_file_hash,
)
from .schema import (
    LoopBExecutionHistory,
    LoopBStatus,
    RequirementDefinition,
)
from .templates import load_static_template

logger = logging.getLogger(__name__)


# =============================================================================
# Dry Run
# =============================================================================


def _get_loopb_flow_diagram() -> str:
    """Load the Loop B flow diagram from template."""
    return load_static_template("loopb_flow_diagram.txt")


DEFAULT_MAX_ITERATIONS = 3


def _print_loopb_config(
    requirements_path: Path, cwd: str, args: argparse.Namespace
) -> None:
    """Print Loop B configuration section."""
    print("## Configuration")
    print(f"  Requirements file:  {requirements_path}")
    print(f"  Working directory:  {cwd}")
    print(f"  Max iterations:     {DEFAULT_MAX_ITERATIONS}")
    print(f"  Max retries/task:   {args.max_retries}")
    print(f"  Max total retries:  {args.max_total_retries}")
    web_status = "disabled" if args.no_web else "enabled"
    perm_mode = "bypassPermissions" if args.bypass_permissions else "acceptEdits"
    print(f"  Web tools:          {web_status}")
    print(f"  Permission mode:    {perm_mode}")
    print()


def _print_loopb_requirements(requirements: RequirementDefinition) -> None:
    """Print requirements section."""
    print(f"## Requirements ({len(requirements.requirements)} total)")
    print()
    for req in requirements.requirements:
        print(f"### {req.id}: {req.name}")
        print("  Acceptance Criteria:")
        for i, criterion in enumerate(req.acceptance_criteria, 1):
            print(f"    {req.id}.{i}: {criterion}")
        print()


def print_loopb_dry_run(args: argparse.Namespace) -> int:
    """Print dry run information for Loop B."""
    requirements_path = Path(args.input_file)
    if not requirements_path.exists():
        logger.error(f"Requirements file not found: {requirements_path}")
        return 1

    requirements = RequirementDefinition.from_yaml(str(requirements_path))
    cwd = args.cwd or str(Path.cwd())

    print("=" * 70)
    print("DRY RUN - Loop B (Requirements-Driven Execution)")
    print("=" * 70)
    print()

    _print_loopb_config(requirements_path, cwd, args)
    _print_loopb_requirements(requirements)

    print("=" * 70)
    print("## Execution Flow Preview")
    print("=" * 70)
    print()
    print(f"Loop B will iterate up to {DEFAULT_MAX_ITERATIONS} times:")
    print()
    print(_get_loopb_flow_diagram())

    print("=" * 70)
    print("## Task Generation Prompt (iteration 1)")
    print("=" * 70)
    print()
    prompt = build_task_generation_prompt(requirements)
    if len(prompt) > 2000:
        print(prompt[:2000])
        print(f"\n... (truncated, total {len(prompt)} chars)")
    else:
        print(prompt)
    print()

    print("=" * 70)
    print("END DRY RUN")
    print("=" * 70)

    return 0


# =============================================================================
# History Formatting
# =============================================================================


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

    if history.loop_c_history_ids:
        lines.append("")
        lines.append("Loop C histories:")
        for hid in history.loop_c_history_ids:
            lines.append(f"  - {hid}")

    if history.completed_task_ids:
        lines.append("")
        lines.append(f"Completed tasks ({len(history.completed_task_ids)}):")
        for tid in history.completed_task_ids[:10]:
            lines.append(f"  - {tid}")
        if len(history.completed_task_ids) > 10:
            lines.append(f"  ... and {len(history.completed_task_ids) - 10} more")

    return "\n".join(lines)


# =============================================================================
# History Command Handlers
# =============================================================================


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
        if history.status not in {LoopBStatus.COMPLETED, LoopBStatus.FAILED}:
            print()
            print("To resume, use:")
            print(f"  task-orchestrator resume {history.history_id} --loopb")
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
    format_loopc_summary: Any,
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
    args: argparse.Namespace, cwd: str, format_loopc_summary: Any
) -> int:
    """Handle Loop B history subcommands."""
    loopb_manager = LoopBHistoryManager(cwd)

    if args.loopb_children:
        loopc_manager = HistoryManager(cwd)
        return handle_loopb_children(
            args, loopb_manager, loopc_manager, format_loopc_summary
        )

    if args.show:
        return handle_loopb_history_show(args, loopb_manager)

    return handle_loopb_history_list(args, loopb_manager)


# =============================================================================
# Result Output
# =============================================================================


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
    result_dict = build_loopb_result_dict(history)
    output_path = Path(args.output) if args.output else None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Results written to: {output_path}")
    else:
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    if history.status == LoopBStatus.COMPLETED:
        logger.info("All requirements fulfilled!")
        return 0

    logger.error(f"Loop B failed: {history.error or history.status.value}")
    return 1


# =============================================================================
# Run Loop B
# =============================================================================


def _build_loopb_config(args: argparse.Namespace) -> RequirementsOrchestratorConfig:
    """Build RequirementsOrchestratorConfig from parsed arguments."""
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

    return RequirementsOrchestratorConfig(
        max_iterations=DEFAULT_MAX_ITERATIONS,
        orchestrator_config=orchestrator_config,
    )


async def run_loopb_orchestrator(args: argparse.Namespace) -> LoopBExecutionHistory:
    """Execute Loop B orchestration and return history.

    Handles the following cases:
    - Case B: Incomplete history exists -> resume from last point
    - Case C: Completed history + YAML changed -> resume with new requirements
    - Case D: Completed history + YAML unchanged -> re-run verification with same history_id
    """
    requirements_path = Path(args.input_file)
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    cwd = args.cwd or str(Path.cwd())
    history_manager = LoopBHistoryManager(cwd)
    resolved_path = str(requirements_path.resolve())
    current_hash = compute_file_hash(resolved_path)

    # Case B: Check for incomplete history to auto-resume
    existing = history_manager.find_incomplete_by_path(str(requirements_path))
    if existing:
        logger.info(f"Found incomplete Loop B execution: {existing.history_id}")

        # Check if requirements changed (Case C)
        if existing.requirements_hash and existing.requirements_hash != current_hash:
            logger.info("Requirements file changed since last execution")
            logger.info("Resuming with updated requirements...")
            existing.requirements_hash = current_hash  # Update hash

        logger.info("Auto-resuming from previous state...")
        requirements = RequirementDefinition.from_yaml(str(requirements_path))
        config = _build_loopb_config(args)
        orchestrator = RequirementsOrchestrator(
            requirements=requirements,
            config=config,
            history_manager=history_manager,
            requirements_path=str(requirements_path),
        )
        return await orchestrator.resume(existing)

    # Check for completed history (Case C/D)
    completed = history_manager.find_history_by_path(str(requirements_path))
    if completed and completed.status == LoopBStatus.COMPLETED:
        requirements = RequirementDefinition.from_yaml(str(requirements_path))
        config = _build_loopb_config(args)

        if completed.requirements_hash and completed.requirements_hash != current_hash:
            # Case C: YAML changed -> resume from completed state with new requirements
            logger.info(f"Found completed history: {completed.history_id}")
            logger.info("Requirements file changed - re-running verification...")
            completed.requirements_hash = current_hash
            completed.status = LoopBStatus.VERIFYING_REQUIREMENTS
            completed.error = None
            completed.final_result = None
            history_manager.save_history(completed)

            orchestrator = RequirementsOrchestrator(
                requirements=requirements,
                config=config,
                history_manager=history_manager,
                requirements_path=str(requirements_path),
            )
            return await orchestrator.resume(completed)

        # Case D: YAML unchanged -> re-run verification with same history_id
        logger.info(f"Found completed history: {completed.history_id}")
        logger.info("Requirements unchanged - re-running verification...")
        completed.status = LoopBStatus.VERIFYING_REQUIREMENTS
        completed.error = None
        completed.final_result = None
        history_manager.save_history(completed)

        orchestrator = RequirementsOrchestrator(
            requirements=requirements,
            config=config,
            history_manager=history_manager,
            requirements_path=str(requirements_path),
        )
        return await orchestrator.resume(completed)

    # New execution
    logger.info(f"Loading requirements from: {requirements_path}")
    requirements = RequirementDefinition.from_yaml(str(requirements_path))
    logger.info(f"Loaded {len(requirements.requirements)} requirements")

    config = _build_loopb_config(args)

    orchestrator = RequirementsOrchestrator(
        requirements=requirements,
        config=config,
        history_manager=history_manager,
        requirements_path=str(requirements_path),
    )

    logger.info("Starting Loop B orchestrator...")
    return await orchestrator.run()


# =============================================================================
# Resume Loop B
# =============================================================================


def _load_resumable_loopb_history(
    history_manager: LoopBHistoryManager, history_id: str
) -> LoopBExecutionHistory | None:
    """Load Loop B history and validate it can be resumed."""
    try:
        history = history_manager.load_history(history_id)
    except FileNotFoundError:
        logger.error(f"Loop B history not found: {history_id}")
        return None

    if history.status == LoopBStatus.COMPLETED:
        logger.error(f"Loop B history {history_id} is already completed.")
        return None

    if history.status == LoopBStatus.FAILED:
        logger.warning(f"Loop B history {history_id} was failed, attempting to resume.")

    if not Path(history.requirements_path).exists():
        logger.error(f"Requirements file not found: {history.requirements_path}")
        return None

    return history


async def resume_loopb(args: argparse.Namespace) -> int:
    """Resume Loop B execution from history."""
    cwd = args.cwd or str(Path.cwd())
    history_manager = LoopBHistoryManager(cwd)

    history = _load_resumable_loopb_history(history_manager, args.history_id)
    if not history:
        return 1

    logger.info(f"Resuming Loop B from: {history.history_id}")
    logger.info(f"  Status: {history.status.value}")
    logger.info(f"  Iteration: {history.current_iteration}/{history.max_iterations}")

    # Load requirements
    requirements = RequirementDefinition.from_yaml(history.requirements_path)

    # Build config (use saved config where possible, allow overrides)
    allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"]
    if not args.no_web:
        allowed_tools.extend(["WebFetch", "WebSearch"])

    permission_mode = "bypassPermissions" if args.bypass_permissions else "acceptEdits"

    orchestrator_config = OrchestratorConfig(
        max_retries_per_task=3,  # Use defaults for resume
        max_total_retries=10,
        cwd=args.cwd or cwd,
        model=args.model,
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
    )

    config = RequirementsOrchestratorConfig(
        max_iterations=history.max_iterations,
        tasks_output_dir=history.tasks_output_dir,
        orchestrator_config=orchestrator_config,
    )

    orchestrator = RequirementsOrchestrator(
        requirements=requirements,
        config=config,
        history_manager=history_manager,
        requirements_path=history.requirements_path,
    )

    # Resume from existing history
    result = await orchestrator.resume(history)

    return output_loopb_result(result, args)
