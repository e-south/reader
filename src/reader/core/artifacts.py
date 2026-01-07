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
    contract_id: str  # e.g., "tidy+map.v1" or "none"
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
        manifests/
          manifest.json
          plots_manifest.json
          exports_manifest.json
        artifacts/stepNN_id.plugin/
          df.parquet
          meta.json
        plots/
          <flat files emitted by plot specs>
        exports/
          <flat files emitted by export specs>
        notebooks/
          <scaffolded marimo notebooks>

    - Manifest tracks latest and historical versions for (step_id, output_name)
    - When config digest changes, a new revision directory __rN is created
    """

    def __init__(
        self, outputs_dir: Path, *, plots_subdir: str | None = "plots", exports_subdir: str | None = "exports"
    ) -> None:
        self.root = outputs_dir
        self.artifacts_dir = self.root / "artifacts"
        self.manifests_dir = self.root / "manifests"
        # flatten when plots_subdir is None / "" / "."; otherwise use given subdir
        if plots_subdir in (None, "", ".", "./"):
            self.plots_dir = self.root
        else:
            self.plots_dir = self.root / plots_subdir
        if exports_subdir in (None, "", ".", "./"):
            self.exports_dir = self.root
        else:
            self.exports_dir = self.root / exports_subdir
        self.manifest_path = self.manifests_dir / "manifest.json"
        self.plots_manifest_path = self.manifests_dir / "plots_manifest.json"
        self.exports_manifest_path = self.manifests_dir / "exports_manifest.json"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_dir != self.root:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        if self.exports_dir != self.root:
            self.exports_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write_manifest({"artifacts": {}, "history": {}})

    # -------------- manifest --------------
    def _read_manifest(self) -> dict[str, Any]:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ArtifactError(f"manifest.json is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ArtifactError("manifest.json must be a JSON object")
        if "artifacts" not in data or "history" not in data:
            raise ArtifactError("manifest.json must include 'artifacts' and 'history' objects")
        if not isinstance(data["artifacts"], dict) or not isinstance(data["history"], dict):
            raise ArtifactError("manifest.json 'artifacts' and 'history' must be JSON objects")
        return data

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # -------------- plots manifest --------------
    def _read_plots_manifest(self) -> dict[str, Any]:
        if not self.plots_manifest_path.exists():
            return {"schema_version": 1, "plots": []}
        try:
            data = json.loads(self.plots_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ArtifactError(f"plots_manifest.json is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ArtifactError("plots_manifest.json must be a JSON object")
        if "plots" not in data or not isinstance(data["plots"], list):
            raise ArtifactError("plots_manifest.json must include a 'plots' list")
        return data

    def _write_plots_manifest(self, payload: dict[str, Any]) -> None:
        self.plots_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def append_plot_entry(self, entry: dict[str, Any]) -> None:
        manifest = self._read_plots_manifest()
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("plots", [])
        manifest["plots"].append(entry)
        self._write_plots_manifest(manifest)

    # -------------- exports manifest --------------
    def _read_exports_manifest(self) -> dict[str, Any]:
        if not self.exports_manifest_path.exists():
            return {"schema_version": 1, "exports": []}
        try:
            data = json.loads(self.exports_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ArtifactError(f"exports_manifest.json is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ArtifactError("exports_manifest.json must be a JSON object")
        if "exports" not in data or not isinstance(data["exports"], list):
            raise ArtifactError("exports_manifest.json must include an 'exports' list")
        return data

    def _write_exports_manifest(self, payload: dict[str, Any]) -> None:
        self.exports_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def append_export_entry(self, entry: dict[str, Any]) -> None:
        manifest = self._read_exports_manifest()
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("exports", [])
        manifest["exports"].append(entry)
        self._write_exports_manifest(manifest)

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
        label: str,
        df: pd.DataFrame,
        contract_id: str,
        inputs: list[str],
        config_digest: str,
        code_digest: str | None = None,
        validate_contract: bool = True,
    ) -> Artifact:
        # Choose base step directory; new revision when prior config differs
        base = f"{step_id}.{plugin_key}"
        step_dir = self.artifacts_dir / base
        man = self._read_manifest()
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
        if validate_contract and contract_id != "none":
            contract = BUILTIN.get(contract_id)
            if not contract:
                raise ArtifactError(f"Unknown contract id '{contract_id}'")
            validate_df(df, contract, where=f"{step_id}:{out_name}")

        df.to_parquet(data_path, index=False)  # pyarrow

        content_digest = _sha256_bytes(data_path.read_bytes())
        meta = {
            "artifact_id": f"{base}/{out_name}",
            "label": label,
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
        """Directory where plot specs write. May be outputs/ or a subdir."""
        return self.plots_dir
