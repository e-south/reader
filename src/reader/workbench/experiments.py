from __future__ import annotations

from pathlib import Path

_SCAFFOLD_NAMES = frozenset({"template", "templates"})
_SCAFFOLD_PREFIXES = ("template_", "scaffold", "_template")


def is_scaffold_dir(path: Path) -> bool:
    name = path.name.strip().lower()
    if not name:
        return False
    return name in _SCAFFOLD_NAMES or any(name.startswith(prefix) for prefix in _SCAFFOLD_PREFIXES)


def discover_experiment_dirs(root: Path, *, include_scaffolds: bool = False) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    experiment_dirs: dict[Path, Path] = {}
    for cfg in root.glob("**/config.yaml"):
        exp_dir = cfg.parent.resolve()
        if not include_scaffolds and is_scaffold_dir(exp_dir):
            continue
        experiment_dirs[exp_dir] = cfg.resolve()
    return sorted(experiment_dirs)


def discover_experiment_configs(root: Path, *, include_scaffolds: bool = False) -> list[Path]:
    return [exp_dir / "config.yaml" for exp_dir in discover_experiment_dirs(root, include_scaffolds=include_scaffolds)]
