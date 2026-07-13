"""Composed runtime service for response-window analysis."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from reader.domains.plate_reader.analysis.response_window.bundle import (
    ResponseWindowBundle,
)
from reader.domains.plate_reader.analysis.response_window.bundle import (
    build_response_window_bundle as build_bundle_core,
)
from reader.domains.plate_reader.analysis.response_window.bundle import (
    verify_response_window_bundle as verify_bundle_core,
)
from reader.domains.plate_reader.analysis.response_window.contracts import EventSpec, ResponseSourceSpec
from reader.domains.plate_reader.analysis.response_window.preflight import (
    ResponseWindowPreflight,
)
from reader.domains.plate_reader.analysis.response_window.preflight import (
    preflight_response_window_request as preflight_core,
)
from reader.domains.plate_reader.analysis.response_window.promoter_evidence_bundle import (
    PromoterEvidenceBundle,
    verify_promoter_evidence_bundle,
)
from reader.domains.plate_reader.analysis.response_window.promoter_evidence_bundle import (
    build_promoter_evidence_bundle as build_promoter_evidence_core,
)
from reader.domains.plate_reader.analysis.response_window.sources import (
    ExperimentSource,
    ResolvedExperimentSource,
    load_experiment_source,
)
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.decl import load_workbench_decl


def preflight_response_window_request(
    *,
    reader_root: Path,
    request_path: Path,
) -> ResponseWindowPreflight:
    runtime = builtin_runtime()
    return preflight_core(
        request_path=request_path,
        source_loader=_source_loader(reader_root=reader_root, runtime=runtime),
    )


def build_response_window_bundle(
    *,
    reader_root: Path,
    request_path: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> ResponseWindowBundle:
    runtime = builtin_runtime()
    return build_bundle_core(
        request_path=request_path,
        out_dir=out_dir,
        overwrite=overwrite,
        contracts=runtime.contracts,
        source_loader=_source_loader(reader_root=reader_root, runtime=runtime),
    )


def verify_response_window_bundle(path: Path) -> ResponseWindowBundle:
    return verify_bundle_core(path, contracts=builtin_runtime().contracts)


def build_promoter_evidence_bundle(
    *,
    response_bundle_root: Path,
    bindings_root: Path,
    out_dir: Path,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
    objective_overlay_path: Path | None = None,
    overwrite: bool = False,
) -> PromoterEvidenceBundle:
    response_bundle = verify_response_window_bundle(response_bundle_root)
    return build_promoter_evidence_core(
        response_bundle=response_bundle,
        bindings_root=bindings_root,
        out_dir=out_dir,
        experiment_id=experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
        objective_overlay_path=objective_overlay_path,
        overwrite=overwrite,
    )


def _source_loader(*, reader_root: Path, runtime: ReaderRuntime):
    return partial(_load_source, reader_root=Path(reader_root).expanduser().resolve(), runtime=runtime)


def _load_source(
    experiment_id: str,
    source_spec: ResponseSourceSpec,
    event_spec: EventSpec,
    *,
    reader_root: Path,
    runtime: ReaderRuntime,
) -> ExperimentSource:
    year = experiment_id[:4]
    if len(year) != 4 or not year.isdigit() or Path(experiment_id).name != experiment_id:
        raise ValueError(
            f"experiment id must be one safe path segment beginning with a four-digit year: {experiment_id!r}."
        )
    experiment_dir = reader_root / "experiments" / year / experiment_id
    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment config not found: {config_path}")
    declaration = load_workbench_decl(config_path, protocols=runtime.protocols)
    if declaration.experiment.id != experiment_id:
        raise ValueError(
            f"experiment config identity {declaration.experiment.id!r} disagrees with request {experiment_id!r}."
        )
    store = runtime.record_store(declaration.experiment_semantics.layout.outputs_dir, create=False)
    if not store.catalog_exists():
        raise FileNotFoundError(f"record manifest not found: {store.records_path}")
    record_ids = {
        source_spec.response_record_id,
        source_spec.magnitude_record_id,
        source_spec.trajectory_record_id,
        source_spec.reference_authority_record_id,
    }
    records = {record_id: store.read_dataframe(record_id) for record_id in sorted(record_ids)}
    for record in records.values():
        record.verify_content_digest()
    logic_map = declaration.experiment_semantics.annotations.resolve_logic_map(ref=source_spec.state_map_ref)
    resolved = ResolvedExperimentSource(
        experiment_id=experiment_id,
        experiment_dir=experiment_dir.resolve(),
        config_path=config_path.resolve(),
        records_path=store.records_path.resolve(),
        record_paths={record_id: record.path.resolve() for record_id, record in records.items()},
        record_contracts={record_id: record.contract_id for record_id, record in records.items()},
        record_digests={record_id: record.content_digest for record_id, record in records.items()},
        state_column=logic_map.column,
        treatment_map=logic_map.corners,
        state_values_case_sensitive=logic_map.case_sensitive,
    )
    return load_experiment_source(
        resolved,
        source_spec=source_spec,
        event_spec=event_spec,
        contracts=runtime.contracts,
    )


__all__ = [
    "build_promoter_evidence_bundle",
    "build_response_window_bundle",
    "preflight_response_window_request",
    "verify_promoter_evidence_bundle",
    "verify_response_window_bundle",
]
