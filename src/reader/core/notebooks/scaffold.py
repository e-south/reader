from __future__ import annotations

from pathlib import Path

from .catalog import normalize_notebook_preset, resolve_notebook_template_descriptor


def write_experiment_notebook(
    target: Path,
    *,
    preset: str = "notebook/eda",
    overwrite: bool = False,
    plot_specs: list[dict] | None = None,
) -> tuple[Path, bool]:
    if target.exists() and not overwrite:
        return target, False
    descriptor = resolve_notebook_template_descriptor(normalize_notebook_preset(preset))
    template = descriptor.template
    if descriptor.uses == "notebook/eda" and "__PLOT_SPECS__" in template:
        payload = plot_specs or []
        template = template.replace("__PLOT_SPECS__", repr(payload))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    return target, True
