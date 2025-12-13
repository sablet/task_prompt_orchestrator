"""Model configuration constants for task_prompt_orchestrator."""

# High-cost model for initial codebase exploration
# Currently using Opus 4.5 for comprehensive analysis
MODEL_EXPLORATION = "claude-opus-4-5-20251101"

# Default model for task execution
# Currently using Sonnet 4.5 for balanced performance and cost
MODEL_DEFAULT = "claude-sonnet-4-5-20250929"
