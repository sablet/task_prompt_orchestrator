"""CLI functions for plan command (interactive Claude CLI)."""

import logging
import os
import shutil
import subprocess  # noqa: S404
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


def _load_plan_prompt(target: str) -> str:
    """Load plan prompt from package templates."""
    import importlib.resources

    templates = importlib.resources.files("task_prompt_orchestrator") / "templates"
    prompt_file = templates / f"plan_{target}.md"

    # Read the template content
    return prompt_file.read_text(encoding="utf-8")


def _build_plan_initial_prompt(target: str, file_path: str | None) -> str:
    """Build the initial prompt for the plan command."""
    # Load base prompt from template
    base_prompt = _load_plan_prompt(target)

    # Remove frontmatter (YAML header)
    if base_prompt.startswith("---"):
        end_idx = base_prompt.find("---", 3)
        if end_idx != -1:
            base_prompt = base_prompt[end_idx + 3 :].strip()

    # Add file context if specified
    if file_path:
        file_context = f"\n\n## 対象ファイル\n\n修正対象: `{file_path}`\n\nまずこのファイルを読み込んで内容を確認してください。"
        return base_prompt + file_context

    return (
        base_prompt
        + "\n\n## モード\n\n新規作成モードです。目的・背景をヒアリングしてください。"
    )


def _find_claude_cli() -> str | None:
    """Find claude CLI path."""
    # Check standard locations
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path

    # Check common installation paths
    home = os.path.expanduser("~")
    common_paths = [
        os.path.join(home, ".claude", "local", "claude"),
        os.path.join(home, ".claude", "local", "node_modules", ".bin", "claude"),
        "/usr/local/bin/claude",
    ]

    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def plan_command(args: "argparse.Namespace") -> int:
    """Execute the plan command (launches claude CLI in interactive mode)."""
    # Find claude CLI
    claude_path = _find_claude_cli()
    if not claude_path:
        logger.error("claude CLI not found. Please install Claude Code first.")
        logger.error("See: https://claude.ai/code")
        return 1

    # Build initial prompt
    initial_prompt = _build_plan_initial_prompt(args.target, args.file)

    logger.info(f"Launching interactive planning session for {args.target}...")
    if args.file:
        logger.info(f"Target file: {args.file}")

    # Launch claude CLI in interactive mode with acceptEdits permission mode
    try:
        result = subprocess.run(  # noqa: S603
            [claude_path, "--permission-mode", "acceptEdits", initial_prompt],
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        logger.info("Planning session interrupted.")
        return 0
