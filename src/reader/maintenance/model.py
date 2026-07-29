from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MaintenanceReport:
    """Typed result from a repository-maintenance check."""

    check: str
    repo_root: Path
    checked: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "reader.maintenance/v1",
            "check": self.check,
            "status": "ok" if self.ok else "failed",
            "repo_root": str(self.repo_root),
            "checked": self.checked,
            "errors": list(self.errors),
        }
