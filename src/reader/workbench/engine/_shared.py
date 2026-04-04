from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.graph import resolve_workbench


def digest_cfg(plugin_cfg: Any) -> str:
    if hasattr(plugin_cfg, "model_dump"):
        payload = plugin_cfg.model_dump(mode="json")
    elif isinstance(plugin_cfg, dict):
        payload = plugin_cfg
    else:
        payload = json.loads(json.dumps(plugin_cfg, default=str))
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def needs_plot_palette(steps: list[Any], palette: str | None) -> bool:
    if palette is None:
        return False
    return any(getattr(step, "plugin", "").startswith("plot/") for step in steps)


def collect_categories(steps: list[Any]) -> set[str]:
    categories: set[str] = set()
    for step in steps:
        plugin = getattr(step, "plugin", "")
        if "/" in plugin:
            categories.add(plugin.split("/", 1)[0])
    return categories


def has_cytometry_step(decl: WorkbenchDecl, *, runtime: ReaderRuntime | None = None) -> bool:
    return pipeline_has_plugin(decl, runtime=runtime, domain="cytometry")


def pipeline_has_plugin(
    decl: WorkbenchDecl,
    *,
    runtime: ReaderRuntime | None = None,
    plugin: str | None = None,
    domain: str | None = None,
    family: str | None = None,
    tag: str | None = None,
) -> bool:
    pipeline = list(resolve_workbench(decl).pipeline)
    if not pipeline:
        return False

    registry = None
    if domain is not None or family is not None or tag is not None:
        runtime = runtime or builtin_runtime()
        registry = runtime.plugins

    for step in pipeline:
        step_plugin = str(getattr(step, "plugin", ""))
        if plugin is not None and step_plugin != plugin:
            continue
        if domain is None and family is None and tag is None:
            return True
        if registry is None:
            continue
        descriptor = registry.resolve_descriptor(step_plugin)
        if domain is not None and descriptor.domain != domain:
            continue
        if family is not None and descriptor.family != family:
            continue
        if tag is not None and tag not in descriptor.tags:
            continue
        return True
    return False


def snapshot_dir(root: Path) -> dict[Path, float]:
    if not root.exists():
        return {}
    return {path: path.stat().st_mtime for path in root.rglob("*") if path.is_file()}


def diff_files(before: dict[Path, float], after: dict[Path, float]) -> list[Path]:
    changed: list[Path] = []
    for path, mtime in after.items():
        prev = before.get(path)
        if prev is None or mtime > prev + 1e-6:
            changed.append(path)
    return changed
