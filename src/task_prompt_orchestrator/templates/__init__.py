"""Template loader for Jinja2 templates."""

from collections.abc import Callable
from functools import lru_cache
from importlib import resources
from typing import Any

from jinja2 import BaseLoader, Environment, TemplateNotFound


class PackageLoader(BaseLoader):
    """Jinja2 loader that reads from package resources."""

    def __init__(self, package: str) -> None:
        self.package = package

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str, Callable[[], bool]]:
        try:
            files = resources.files(self.package)
            content = (files / template).read_text(encoding="utf-8")
            return content, template, lambda: True
        except (FileNotFoundError, TypeError) as e:
            raise TemplateNotFound(template) from e


@lru_cache(maxsize=1)
def get_environment() -> Environment:
    """Get cached Jinja2 environment."""
    return Environment(
        loader=PackageLoader("task_prompt_orchestrator.templates"),
        autoescape=False,  # noqa: S701 - Plain text prompts, not HTML
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(template_name: str, **context: Any) -> str:
    """Render a template with the given context."""
    env = get_environment()
    template = env.get_template(template_name)
    return template.render(**context)


def load_static_template(template_name: str) -> str:
    """Load a static template (no variable substitution)."""
    files = resources.files("task_prompt_orchestrator.templates")
    return (files / template_name).read_text(encoding="utf-8")
