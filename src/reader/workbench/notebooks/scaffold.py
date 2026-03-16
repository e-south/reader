from __future__ import annotations

from pathlib import Path

from reader.workbench.templates import resolve_notebook_template_descriptor


def write_experiment_notebook(
    target: Path,
    *,
    template: str,
    overwrite: bool = False,
    plot_specs: list[dict] | None = None,
    allow_record_scan: bool = False,
) -> tuple[Path, bool]:
    if target.exists() and not overwrite:
        return target, False
    descriptor = resolve_notebook_template_descriptor(template)
    body = descriptor.load_body()
    if descriptor.capabilities.inject_plot_specs and "__PLOT_SPECS__" in body:
        payload = plot_specs or []
        body = body.replace("__PLOT_SPECS__", repr(payload))
    if "__ALLOW_RECORD_SCAN__" in body:
        body = body.replace("__ALLOW_RECORD_SCAN__", repr(bool(allow_record_scan)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target, True
