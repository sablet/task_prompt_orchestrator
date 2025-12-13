"""Loop B verification logic - requirement verification mixin."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .loopb_history import StepModeStop
from .models import MODEL_EXPLORATION
from .orchestrator import BOLD, CYAN, DIM, GREEN, RESET, YELLOW, run_claude_query
from .prompts import (
    build_shared_exploration_prompt,
    build_single_requirement_verification_prompt,
    get_unmet_requirement_ids,
    parse_verification_result,
)
from .schema import Requirement

if TYPE_CHECKING:
    from .schema import LoopBExecutionHistory, RequirementDefinition

    from .loopb_history import RequirementsOrchestratorConfig


class VerificationMixin:
    """Mixin providing requirement verification functionality."""

    # Type hints for attributes from RequirementsOrchestrator
    history: "LoopBExecutionHistory | None"
    requirements: "RequirementDefinition"
    requirements_path: str
    config: "RequirementsOrchestratorConfig"

    def _get_callback(self) -> Any:
        """Get stream callback."""
        ...

    def _save_history(self) -> None:
        """Save history."""
        ...

    def _print_step(self, step_name: str, description: str = "") -> None:
        """Print step."""
        ...

    def _print_verification_header(self, resume_index: int, total_reqs: int) -> None:
        """Print verification header."""
        config = self.config.orchestrator_config
        if not config.stream_output:
            return
        callback = self._get_callback()
        callback(f"\n{BOLD}{'-' * 60}{RESET}\n")
        callback(f"{BOLD}{CYAN}🔎 REQUIREMENT VERIFICATION{RESET}\n")
        if resume_index > 0:
            callback(
                f"{DIM}   Resuming from requirement {resume_index + 1}/{total_reqs}{RESET}\n"
            )
        else:
            callback(f"{DIM}   Total requirements: {total_reqs}{RESET}\n")
        callback(f"{BOLD}{'-' * 60}{RESET}\n")

    def _print_exploration_header(self) -> None:
        """Print shared exploration header."""
        config = self.config.orchestrator_config
        if not config.stream_output:
            return
        callback = self._get_callback()
        callback(f"\n{BOLD}{'=' * 60}{RESET}\n")
        callback(f"{BOLD}{CYAN}🔍 SHARED EXPLORATION{RESET}\n")
        callback(f"{DIM}   Gathering codebase context for all verifications{RESET}\n")
        callback(f"{BOLD}{'=' * 60}{RESET}\n")

    async def _run_shared_exploration(self, iteration: int) -> str:
        """Run shared codebase exploration before individual verifications.

        Args:
            iteration: Current iteration number (used in filename).

        Returns:
            Path to the exploration output file.
        """
        assert self.history is not None
        config = self.config.orchestrator_config
        callback = self._get_callback()

        self._print_exploration_header()

        prompt = build_shared_exploration_prompt(self.requirements_path)

        if config.stream_output:
            callback(f"{DIM}>>> PROMPT >>>{RESET}\n")
            callback(f"{DIM}{prompt}{RESET}\n")
            callback(f"{DIM}<<< END PROMPT <<<{RESET}\n\n")

        # Use Opus model for exploration to leverage comprehensive analysis capabilities
        output = await run_claude_query(
            prompt, config, phase="exploration", model_override=MODEL_EXPLORATION
        )

        # Write exploration output to file
        output_dir = Path(self.history.tasks_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exploration_path = output_dir / f"exploration_iter{iteration}.md"
        exploration_path.write_text(output, encoding="utf-8")

        if config.stream_output:
            callback(f"\n{GREEN}✓ Shared exploration completed{RESET}\n")
            callback(f"{DIM}   Saved to: {exploration_path}{RESET}\n")

        return str(exploration_path)

    def _build_verification_result(
        self, requirement_statuses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build final verification result from statuses."""
        all_met = all(s.get("met", False) for s in requirement_statuses)
        unmet_count = sum(1 for s in requirement_statuses if not s["met"])
        summary = (
            "All requirements met"
            if all_met
            else f"{unmet_count} of {len(requirement_statuses)} requirements not met"
        )
        feedback_parts = [
            f"{s['requirement_id']}: {', '.join(s['issues'])}"
            for s in requirement_statuses
            if not s["met"] and s.get("issues")
        ]
        config = self.config.orchestrator_config
        if config.stream_output:
            callback = self._get_callback()
            callback(f"\n{BOLD}{'-' * 60}{RESET}\n")
            callback(f"{CYAN}Verification Summary: {summary}{RESET}\n")
            callback(f"{BOLD}{'-' * 60}{RESET}\n")
        return {
            "all_requirements_met": all_met,
            "requirement_status": requirement_statuses,
            "summary": summary,
            "feedback_for_additional_tasks": "\n".join(feedback_parts),
        }

    async def _verify_single_requirement(
        self,
        req: Requirement,
        req_index: int,
        total_reqs: int,
        exploration_path: str | None = None,
    ) -> dict[str, Any]:
        """Verify a single requirement by executing verification steps."""
        prompt = build_single_requirement_verification_prompt(
            req,
            all_requirements_path=self.requirements_path,
            exploration_path=exploration_path,
        )
        config = self.config.orchestrator_config
        callback = self._get_callback()

        if config.stream_output:
            callback(f"\n{BOLD}{'=' * 60}{RESET}\n")
            callback(
                f"{BOLD}🔍 VERIFICATION [{req_index}/{total_reqs}]: {req.id}{RESET}\n"
            )
            callback(f"{DIM}   {req.name}{RESET}\n")
            callback(f"{BOLD}{'=' * 60}{RESET}\n")
            callback(f"{DIM}>>> PROMPT >>>{RESET}\n")
            callback(f"{DIM}{prompt}{RESET}\n")
            callback(f"{DIM}<<< END PROMPT <<<{RESET}\n\n")

        output = await run_claude_query(prompt, config, phase="verification")
        result = parse_verification_result(output)

        if "requirement_id" not in result:
            result["requirement_id"] = req.id

        return result

    async def _verify_requirements(
        self,
        iteration: int,
        resume_index: int = 0,
        partial_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Verify if requirements are met by checking each requirement individually.

        Args:
            iteration: Current iteration number.
            resume_index: Index to resume verification from.
            partial_results: Partial verification results from previous run.

        Raises:
            StepModeStop: If step mode is enabled and a step was executed.
        """
        assert self.history is not None
        config = self.config.orchestrator_config
        total_reqs = len(self.requirements.requirements)

        # Run shared exploration if not already done for this iteration
        exploration_path = self.history.shared_exploration_path
        exploration_iteration = self.history.shared_exploration_iteration

        if exploration_path is None or exploration_iteration != iteration:
            # Need new exploration (first time or iteration changed)
            exploration_path = await self._run_shared_exploration(iteration)
            self.history.shared_exploration_path = exploration_path
            self.history.shared_exploration_iteration = iteration
            self._save_history()

            # Check for step mode after exploration
            if config.step_mode and config._step_executed:
                raise StepModeStop("exploration")

        self._print_verification_header(resume_index, total_reqs)

        requirement_statuses: list[dict[str, Any]] = list(partial_results or [])

        for idx, req in enumerate(self.requirements.requirements):
            if idx < resume_index:
                continue
            self.history.current_verification_index = idx
            self.history.partial_verification_results = requirement_statuses
            self._save_history()

            result = await self._verify_single_requirement(
                req, idx + 1, total_reqs, exploration_path=exploration_path
            )
            met = result.get("met", False)
            status_record = {
                "requirement_id": req.id,
                "met": met,
                "evidence": result.get("summary", ""),
                "criteria_results": result.get("criteria_results", []),
                "issues": result.get("issues", []),
            }
            requirement_statuses.append(status_record)

            if config.stream_output:
                callback = self._get_callback()
                status_icon = f"{GREEN}✓{RESET}" if met else f"{YELLOW}✗{RESET}"
                callback(f"\n  {status_icon} {req.id}: {'met' if met else 'not met'}\n")

            if config.step_mode and config._step_executed:
                self.history.current_verification_index = idx + 1
                self.history.partial_verification_results = requirement_statuses
                self._save_history()
                raise StepModeStop("verification")

        self.history.current_verification_index = 0
        self.history.partial_verification_results = None
        self._save_history()

        return self._build_verification_result(requirement_statuses)

    async def _pre_verify_requirements(
        self,
        iteration: int,
        resume_index: int = 0,
        partial_results: list[dict[str, Any]] | None = None,
    ) -> list[Requirement]:
        """Verify all requirements and return unmet ones."""
        assert self.history is not None
        import logging

        logger = logging.getLogger(__name__)

        if resume_index > 0:
            logger.info(
                f"Loop B iteration {iteration + 1}: Resuming verification from requirement {resume_index + 1}"
            )
            self._print_step(
                "PRE-VERIFICATION",
                f"Resuming from requirement {resume_index + 1}/{len(self.requirements.requirements)}",
            )
        else:
            logger.info(f"Loop B iteration {iteration + 1}: Pre-verifying requirements")
            self._print_step(
                "PRE-VERIFICATION", "Checking which requirements need work"
            )

        from .schema import LoopBStatus

        self.history.status = LoopBStatus.VERIFYING_REQUIREMENTS
        self._save_history()

        verification = await self._verify_requirements(
            iteration, resume_index, partial_results
        )

        if self.history.iterations:
            self.history.iterations[-1].verification_result = verification
            self._save_history()

        unmet_ids = get_unmet_requirement_ids(verification)
        unmet_requirements = [
            req for req in self.requirements.requirements if req.id in unmet_ids
        ]

        config = self.config.orchestrator_config
        if config.stream_output:
            callback = self._get_callback()
            met_count = len(self.requirements.requirements) - len(unmet_requirements)
            callback(
                f"\n{DIM}Pre-verification: {met_count} met, "
                f"{len(unmet_requirements)} need work{RESET}\n"
            )

        return unmet_requirements
