"""End-to-end response-window bundle core and atomic publication."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from reader.contracts import ContractCatalog
from reader.domains.plate_reader.analysis.response_window.contracts import (
    ResponseWindowRequest,
    load_response_window_request,
)
from reader.domains.plate_reader.analysis.response_window.materialize import materialize_experiment
from reader.domains.plate_reader.analysis.response_window.provenance import sha256_file
from reader.domains.plate_reader.analysis.response_window.sources import ExperimentSourceLoader
from reader.domains.plate_reader.plots.response_window.reporting import write_review_artifacts

from .publication import bundle_publication
from .verification import BUNDLE_SCHEMA_VERSION, RECORD_ARTIFACTS, RECORD_CONTRACTS, verify_bundle_payload


@dataclass(frozen=True)
class ResponseWindowBundle:
    root: Path
    manifest_path: Path
    notebook_path: Path
    manifest: dict[str, object]
    counts: dict[str, int]


def build_response_window_bundle(
    *,
    request_path: Path,
    out_dir: Path,
    contracts: ContractCatalog,
    source_loader: ExperimentSourceLoader,
    notebook_writer: Callable[[Path], Path],
    overwrite: bool = False,
) -> ResponseWindowBundle:
    """Materialize verified records and review artifacts for one request."""

    with bundle_publication(
        out_dir,
        bundle_label="response-window",
        overwrite=overwrite,
    ) as publication:
        request_file = Path(request_path).expanduser().resolve()
        request = load_response_window_request(request_file)
        _build_staged_bundle(
            request_path=request_file,
            request=request,
            staging=publication.staging,
            contracts=contracts,
            source_loader=source_loader,
            notebook_writer=notebook_writer,
        )
        return publication.publish(lambda root: verify_response_window_bundle(root, contracts=contracts))


def _build_staged_bundle(
    *,
    request_path: Path,
    request: ResponseWindowRequest,
    staging: Path,
    contracts: ContractCatalog,
    source_loader: ExperimentSourceLoader,
    notebook_writer: Callable[[Path], Path],
) -> ResponseWindowBundle:
    well_frames: list[pd.DataFrame] = []
    design_frames: list[pd.DataFrame] = []
    draw_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    source_records: list[dict[str, object]] = []
    for experiment_id in request.experiment_ids:
        source = source_loader(experiment_id, request.source, request.event)
        if source.experiment_id != experiment_id:
            raise ValueError(f"response-window source loader returned {source.experiment_id!r} for {experiment_id!r}.")
        wells, designs, draws, traces, events = materialize_experiment(source, request=request)
        well_frames.append(wells)
        design_frames.append(designs)
        draw_frames.append(draws)
        trace_frames.append(traces)
        event_frames.append(events)
        snapshot_root = staging / "sources" / experiment_id
        snapshot_root.mkdir(parents=True)
        config_artifact = (snapshot_root / "config.yaml").relative_to(staging).as_posix()
        records_artifact = (snapshot_root / "records.json").relative_to(staging).as_posix()
        shutil.copy2(source.config_path, staging / config_artifact)
        shutil.copy2(source.records_path, staging / records_artifact)
        source_records.append(
            {
                "experiment_id": experiment_id,
                "config_artifact": config_artifact,
                "records_artifact": records_artifact,
                "records": dict(sorted(source.record_digests.items())),
            }
        )

    wells = pd.concat(well_frames, ignore_index=True)
    designs = pd.concat(design_frames, ignore_index=True)
    draws = pd.concat(draw_frames, ignore_index=True)
    traces = pd.concat(trace_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    if events["experiment_id"].duplicated().any() or len(events) != len(request.experiment_ids):
        raise ValueError("response-window event records must contain each requested experiment exactly once.")
    if set(events["experiment_id"].astype(str)) != set(request.experiment_ids):
        raise ValueError("response-window event records drifted from the request experiment universe.")
    frames = {
        "wells": wells,
        "designs": designs,
        "bootstrap_draws": draws,
        "traces": traces,
        "events": events,
    }
    for record_id, frame in frames.items():
        contracts.validate(frame, contract_id=RECORD_CONTRACTS[record_id], where=f"response-window:{record_id}")

    observed_design_ids = set(designs["design_id"].astype(str))
    missing_examples = sorted({example.design_id for example in request.display.examples} - observed_design_ids)
    if missing_examples:
        raise ValueError(
            f"response-window display examples are absent from the requested experiments: {missing_examples}."
        )
    display = request.display.to_manifest(
        response_ratio=request.source.response_channel,
        magnitude_ratio=request.source.magnitude_channel,
        growth=request.source.growth_channel,
        reference_design_id=request.source.reference_design_id,
    )

    tables = staging / "tables"
    tables.mkdir()
    for record_id, frame in frames.items():
        frame.to_parquet(staging / RECORD_ARTIFACTS[record_id], index=False)
    plot_manifest = write_review_artifacts(
        designs,
        events,
        primary_reduction_id=request.primary_reduction.id,
        display=display,
        out_dir=staging,
    )
    notebook_writer(staging)
    request_artifact = staging / "request.yaml"
    shutil.copy2(request_path, request_artifact)

    primary = designs.loc[designs["reduction_role"].eq("primary") & ~designs["is_reference"].astype(bool)]
    repeated_design_count = int((primary.groupby("design_id")["experiment_id"].nunique() > 1).sum())
    counts = {
        "experiments": int(events["experiment_id"].nunique()),
        "well_rows": len(wells),
        "design_rows": len(designs),
        "bootstrap_draw_rows": len(draws),
        "trace_rows": len(traces),
        "unique_design_ids": int(designs["design_id"].nunique()),
        "repeated_design_ids": repeated_design_count,
        "reductions": int(designs["reduction_id"].nunique()),
        "plots": len(plot_manifest),
    }
    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        artifacts[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "study_id": request.study_id,
        "request_id": request.request_id,
        "state_order": list(request.state_order),
        "display": display,
        "created_at": datetime.now(UTC).isoformat(),
        "primary_reduction_id": request.primary_reduction.id,
        "request": {
            "artifact_id": "request.yaml",
            "sha256": sha256_file(request_artifact),
        },
        "contracts": RECORD_CONTRACTS,
        "records": {
            record_id: {
                "contract_id": RECORD_CONTRACTS[record_id],
                "artifact_id": artifact_id,
            }
            for record_id, artifact_id in RECORD_ARTIFACTS.items()
        },
        "counts": counts,
        "source_records": source_records,
        "artifacts": artifacts,
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verify_response_window_bundle(staging, contracts=contracts)


def verify_response_window_bundle(path: Path, *, contracts: ContractCatalog) -> ResponseWindowBundle:
    """Verify one published bundle and return its public paths and counts."""

    root = Path(path).expanduser().resolve()
    manifest_path, manifest, counts = verify_bundle_payload(root, contracts=contracts)
    return ResponseWindowBundle(
        root=root,
        manifest_path=manifest_path,
        notebook_path=root / "review.py",
        manifest=manifest,
        counts=counts,
    )


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "RECORD_ARTIFACTS",
    "RECORD_CONTRACTS",
    "ResponseWindowBundle",
    "build_response_window_bundle",
    "verify_response_window_bundle",
]
