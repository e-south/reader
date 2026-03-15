from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reader.core.config import ReaderSpec
from reader.core.workbench import resolve_workbench


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
    return any(getattr(step, "uses", "").startswith("plot/") for step in steps)


def collect_categories(steps: list[Any]) -> set[str]:
    categories: set[str] = set()
    for step in steps:
        uses = getattr(step, "uses", "")
        if "/" in uses:
            categories.add(uses.split("/", 1)[0])
    return categories


def has_cytometry_step(spec: ReaderSpec) -> bool:
    return any(str(getattr(step, "uses", "")) == "ingest/flow_cytometer" for step in resolve_workbench(spec).pipeline)


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
