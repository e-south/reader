"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/artifacts.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reader.core.contracts import BUILTIN, validate_df
from reader.core.errors import ArtifactError


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Artifact:
    label: str  # logical label ("merged/df")
    contract_id: str  # e.g., "tidy+map.v2" or "none"
    path: Path  # data path
    meta_path: Path  # meta.json path
    meta: dict[str, Any]  # loaded meta json

    def load_dataframe(self) -> pd.DataFrame:
        if self.path.suffix.lower() == ".parquet":
            return pd.read_parquet(self.path)
        raise ArtifactError(f"Artifact {self.label} not a dataframe parquet: {self.path}")


class ArtifactStore:
    """
    Per-experiment store rooted at `outputs/`.

    Layout:
      outputs/
        manifest.json
        artifacts/stepNN_id.plugin/
          df.parquet
          meta.json
        plots/
          <flat files emitted by plot steps>

    - Manifest tracks latest and historical versions for (step_id, output_name)
    - When config digest changes, a new revision directory __rN is created
    """

    def __init__(self, outputs_dir: Path, *, plots_subdir: str | None = "plots") -> None:
        self.root = outputs_dir
        self.artifacts_dir = self.root / "artifacts"
        # flatten when plots_subdir is None / "" / "."; otherwise use given subdir
        if plots_subdir in (None, "", ".", "./"):
            self.plots_dir = self.root
        else:
            self.plots_dir = self.root / plots_subdir
        self.manifest_path = self.root / "manifest.json"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_dir != self.root:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write_manifest({"artifacts": {}, "history": {}})

    # -------------- manifest --------------
    def _read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # -------------- helpers --------------
    def _revision_dir(self, step_dir: Path) -> Path:
        # allocate next __rN suffix if needed
        n = 1
        while True:
            d = step_dir if n == 1 else step_dir.with_name(step_dir.name + f"__r{n}")
            if not d.exists():
                return d
            n += 1

    def _data_path(self, step_dir: Path, out_name: str) -> Path:
        return step_dir / f"{out_name}.parquet"

    # -------------- public API --------------
    def latest(self, label: str) -> Artifact | None:
        m = self._read_manifest()
        entry = m["artifacts"].get(label)
        if not entry:
            return None
        return self._materialize(label, entry)

    def _materialize(self, label: str, entry: dict[str, Any]) -> Artifact:
        step_dir = self.artifacts_dir / entry["step_dir"]
        path = step_dir / entry["filename"]
        meta_path = step_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return Artifact(label=label, contract_id=entry["contract"], path=path, meta_path=meta_path, meta=meta)

    def read(self, label: str) -> Artifact:
        a = self.latest(label)
        if not a:
            raise ArtifactError(f"Artifact '{label}' missing; produce it in an earlier step.")
        return a

    def persist_dataframe(
        self,
        *,
        step_id: str,
        plugin_key: str,
        out_name: str,
        df: pd.DataFrame,
        contract_id: str,
        inputs: list[str],
        config_digest: str,
        code_digest: str | None = None,
    ) -> Artifact:
        # Choose base step directory; new revision when prior config differs
        base = f"{step_id}.{plugin_key}"
        step_dir = self.artifacts_dir / base
        man = self._read_manifest()
        label = f"{step_id}/{out_name}"
        prev = man["artifacts"].get(label)

        if prev:
            # if config_digest differs, allocate a revision dir
            prev_meta = json.loads((self.artifacts_dir / prev["step_dir"] / "meta.json").read_text(encoding="utf-8"))
            if prev_meta.get("config_digest") != config_digest:
                step_dir = self._revision_dir(step_dir)
            else:
                step_dir = self.artifacts_dir / prev["step_dir"]
        else:
            step_dir = self._revision_dir(step_dir)

        step_dir.mkdir(parents=True, exist_ok=True)
        data_path = step_dir / f"{out_name}.parquet"

        # validate against contract before writing
        if contract_id != "none":
            contract = BUILTIN.get(contract_id)
            if not contract:
                raise ArtifactError(f"Unknown contract id '{contract_id}'")
            validate_df(df, contract, where=f"{step_id}:{out_name}")

        df.to_parquet(data_path, index=False)  # pyarrow

        content_digest = _sha256_bytes(data_path.read_bytes())
        meta = {
            "artifact_id": f"{base}/{out_name}",
            "contract": contract_id,
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "producer": {"step_id": step_id, "plugin": plugin_key},
            "inputs": inputs,
            "config_digest": config_digest,
            "code_digest": code_digest or "",
            "content_digest": content_digest,
            "filename": data_path.name,
            "step_dir": step_dir.name,
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        # update manifest
        man["artifacts"][label] = {
            "step_dir": step_dir.name,
            "filename": data_path.name,
            "contract": contract_id,
        }
        man["history"].setdefault(label, []).append(meta)
        self._write_manifest(man)

        return Artifact(
            label=label, contract_id=contract_id, path=data_path, meta_path=step_dir / "meta.json", meta=meta
        )

    # ---- flat plots handling (not listed as dataframe artifacts) ----
    def plots_directory(self) -> Path:
        """Directory where plot steps should write. May be outputs/ or a subdir."""
        return self.plots_dir


class ReportStore:
    """Manifest for report outputs (plots + exports)."""

    def __init__(self, outputs_dir: Path, *, filename: str = "report_manifest.json") -> None:
        self.root = outputs_dir
        self.path = outputs_dir / filename
        if not self.path.exists():
            self._write({"reports": []})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _relpath(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(p)

    def persist_files(
        self,
        *,
        step_id: str,
        plugin_key: str,
        inputs: list[str],
        files: Any,
        config_digest: str,
    ) -> None:
        out_list: list[str] = []
        if files:
            if isinstance(files, (str, Path)):
                out_list = [self._relpath(Path(files))]
            elif isinstance(files, list):
                out_list = [self._relpath(Path(p)) for p in files]
        payload = self._read()
        payload.setdefault("reports", [])
        payload["reports"].append(
            {
                "step_id": step_id,
                "plugin": plugin_key,
                "inputs": inputs,
                "files": out_list,
                "config_digest": config_digest,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        self._write(payload)
