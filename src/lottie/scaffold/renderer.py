"""TemplateRendererSkill — renders Jinja2 scaffold templates with a typed contract.

Templates are package data under `scaffold/templates/`, loaded via `PackageLoader`
so rendering works in an installed wheel, not just editable dev.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined

from lottie.core import BaseSkill
from lottie.scaffold.schema import RenderInput, RenderOutput


class TemplateRendererSkill(BaseSkill[RenderInput, RenderOutput]):
    """Render a named template against a typed context."""

    def __init__(
        self,
        *,
        name: str | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self._env = Environment(
            loader=PackageLoader("lottie.scaffold", "templates"),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            autoescape=False,  # rendering Python/Markdown/YAML, never HTML
        )

    def _execute(self, data: RenderInput) -> RenderOutput:
        template = self._env.get_template(data.template)
        return RenderOutput(content=template.render(**data.context.model_dump()))
