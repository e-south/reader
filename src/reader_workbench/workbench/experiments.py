from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reader_workbench.errors import ConfigError
from reader_workbench.workbench.config import ReaderSpec
from reader_workbench.workbench.paths import resolve_path_within_root

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
        relative = cfg.relative_to(root)
        if "outputs" in relative.parts:
            continue
        exp_dir = cfg.parent.resolve()
        try:
            exp_dir.relative_to(root.resolve())
        except ValueError:
            continue
        if not include_scaffolds and is_scaffold_dir(exp_dir):
            continue
        experiment_dirs[exp_dir] = cfg.resolve()
    return sorted(experiment_dirs)


def discover_experiment_configs(root: Path, *, include_scaffolds: bool = False) -> list[Path]:
    return [exp_dir / "config.yaml" for exp_dir in discover_experiment_dirs(root, include_scaffolds=include_scaffolds)]


@dataclass(frozen=True)
class ExperimentLocation:
    id: str
    root: Path
    config_path: Path
    outputs_dir: Path


class ExperimentCatalog:
    """Resolve experiment identities without assuming year or directory naming."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ConfigError("The experiments workspace is missing or is not a directory")
        self._locations: dict[str, list[ExperimentLocation]] | None = None
        self._invalid_config_count = 0

    @classmethod
    def from_experiment_root(cls, experiment_root: Path) -> ExperimentCatalog:
        return cls(find_experiments_root(experiment_root))

    def resolve(self, experiment_id: str) -> ExperimentLocation:
        identity = _safe_experiment_id(experiment_id)
        if self._locations is None:
            self._locations = self._build_index()
        matches = self._locations.get(identity, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            rendered = ", ".join(str(item.config_path.relative_to(self.root)) for item in matches)
            raise ConfigError(f"Experiment id {identity!r} is ambiguous in the experiments workspace: {rendered}")
        suffix = (
            f" ({self._invalid_config_count} invalid config(s) were ignored.)" if self._invalid_config_count else ""
        )
        raise ConfigError(f"Unknown experiment id {identity!r} in the experiments workspace.{suffix}")

    def _build_index(self) -> dict[str, list[ExperimentLocation]]:
        locations: dict[str, list[ExperimentLocation]] = {}
        for config_path in discover_experiment_configs(self.root):
            try:
                spec = ReaderSpec.load(config_path)
            except ConfigError:
                self._invalid_config_count += 1
                continue
            experiment_root = config_path.parent.resolve()
            try:
                outputs_dir = resolve_path_within_root(spec.paths.outputs, root=experiment_root)
            except ValueError:
                self._invalid_config_count += 1
                continue
            locations.setdefault(spec.experiment.id, []).append(
                ExperimentLocation(
                    id=spec.experiment.id,
                    root=experiment_root,
                    config_path=config_path.resolve(),
                    outputs_dir=outputs_dir,
                )
            )
        return locations


def find_experiments_root(experiment_root: Path) -> Path:
    resolved = Path(experiment_root).expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "experiments" and candidate.is_dir():
            return candidate
    raise ConfigError(
        f"Experiment {resolved} is not inside a canonical experiments/ workspace; "
        "cross-experiment record resources require that ownership boundary."
    )


def _safe_experiment_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Experiment id must be a non-empty string")
    identity = value.strip()
    if Path(identity).name != identity or identity in {".", ".."}:
        raise ConfigError(f"Experiment id must be one safe path segment: {identity!r}")
    return identity


__all__ = [
    "ExperimentCatalog",
    "ExperimentLocation",
    "discover_experiment_configs",
    "discover_experiment_dirs",
    "find_experiments_root",
    "is_scaffold_dir",
]
