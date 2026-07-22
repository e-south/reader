from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reader.domains.plate_reader.analysis import retron_review_semantics
from reader.errors import RecordError
from reader.runtime import builtin_runtime
from reader.workbench.notebooks import _retron_review_catalog as retron_review_catalog
from reader.workbench.notebooks import context as notebook_context
from reader.workbench.records import DataFrameArtifactRecord, discover_dataframe_records


@dataclass(frozen=True)
class RetronReviewSource:
    label: str
    experiment_id: str
    experiment_root: Path | None
    config_path: Path | None
    summary_path: Path
    trace_path: Path


@dataclass(frozen=True)
class RetronReviewBundle:
    manifest_path: Path
    sources: tuple[RetronReviewSource, ...]
    summary_df: pd.DataFrame
    trace_df: pd.DataFrame
    relevant_stress_map: dict[str, str]
    sensor_target_map: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class RetronReviewSourceSurface:
    experiment_title: str
    protocol_id: str
    plot_specs: tuple[dict[str, Any], ...]
    plot_selector_rows: tuple[dict[str, str], ...]
    plot_catalog_rows: tuple[dict[str, str], ...]
    record_paths: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _SourceSurfacePlotRows:
    selector_row: dict[str, str]
    catalog_row: dict[str, str]


@dataclass(frozen=True)
class _ResolvedSourcePaths:
    experiment_root: Path | None
    config_path: Path | None
    summary_path: Path
    trace_path: Path


_TRACE_FLAG_COLUMNS = (
    "in_pre_window",
    "in_primary_post_stress",
    "in_endpoint_window",
)
_SEMANTIC_CONTRACT_BY_RECORD_ID = {
    "semantic_metrics/summary": "plate_reader.sponge_summary.v1",
    "semantic_metrics/trace": "plate_reader.sponge_trace.v1",
}


def load_retron_semantic_maps_from_config(
    config_path: Path,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    analysis = ((payload.get("protocol") or {}).get("analysis") or {}).get("semantic_metrics") or {}
    if not isinstance(analysis, dict):
        raise ValueError("retron_review: protocol.analysis.semantic_metrics must be a mapping")
    return (
        _normalize_relevant_stress_map(analysis.get("relevant_stress_map") or {}),
        _normalize_sensor_target_map(analysis.get("sensor_target_map") or {}),
    )


def load_cached_parquet_frame(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _load_cached_parquet_frame(str(resolved), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _load_cached_parquet_frame(path: str, mtime_ns: int, size_bytes: int) -> pd.DataFrame:
    del mtime_ns, size_bytes
    return pd.read_parquet(path)


def load_retron_source_record_frame(
    source: RetronReviewSource,
    *,
    record_id: str,
    path: str | Path,
) -> pd.DataFrame:
    resolved_path = Path(path).expanduser().resolve()
    if source.experiment_root is None:
        return load_cached_parquet_frame(resolved_path)
    outputs_dir = source.experiment_root / "outputs"
    store = builtin_runtime().record_store(outputs_dir, create=False)
    record = store.read_dataframe(record_id)
    if record.path.resolve() != resolved_path:
        raise RecordError(f"retron_review: path for record {record_id!r} does not match the current record catalog")
    stat = record.path.stat()
    return _load_cached_record_frame(
        str(outputs_dir.resolve()),
        record_id,
        record.content_digest,
        stat.st_mtime_ns,
        stat.st_size,
    )


@lru_cache(maxsize=64)
def _load_cached_record_frame(
    outputs_dir: str,
    record_id: str,
    content_digest: str,
    mtime_ns: int,
    size_bytes: int,
) -> pd.DataFrame:
    del mtime_ns, size_bytes
    store = builtin_runtime().record_store(Path(outputs_dir), create=False)
    record = store.read_dataframe(record_id)
    if record.content_digest != content_digest:
        raise RecordError(f"retron_review: record {record_id!r} changed while loading; reload the source review")
    return record.load_dataframe()


def load_cached_semantic_frame(
    path: str | Path,
    *,
    kind: str,
) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _load_cached_semantic_frame(
        str(resolved),
        stat.st_mtime_ns,
        stat.st_size,
        kind,
    )


@lru_cache(maxsize=128)
def _load_cached_semantic_frame(
    path: str,
    mtime_ns: int,
    size_bytes: int,
    kind: str,
) -> pd.DataFrame:
    del mtime_ns, size_bytes
    frame = _read_semantic_table(Path(path), kind=kind)
    _validate_semantic_frame(frame, record_id=f"semantic_metrics/{kind}", where=path)
    return frame


@lru_cache(maxsize=64)
def _load_cached_record_semantic_frame(
    outputs_dir: str,
    record_id: str,
    content_digest: str,
    mtime_ns: int,
    size_bytes: int,
    kind: str,
) -> pd.DataFrame:
    del mtime_ns, size_bytes
    record = _resolve_semantic_record(Path(outputs_dir), record_id=record_id)
    if record.content_digest != content_digest:
        raise RecordError(f"retron_review: record {record_id!r} changed while loading; reload the review bundle")
    return _normalize_semantic_frame(record.load_dataframe(), kind=kind)


def load_retron_source_semantic_datasets(
    source: RetronReviewSource,
    *,
    record_ids: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    selected = set(record_ids or ("semantic_metrics/summary", "semantic_metrics/trace"))
    datasets: dict[str, pd.DataFrame] = {}
    if "semantic_metrics/summary" in selected:
        datasets["semantic_metrics/summary"] = _load_source_semantic_frame(
            source,
            record_id="semantic_metrics/summary",
            kind="summary",
        )
    if "semantic_metrics/trace" in selected:
        datasets["semantic_metrics/trace"] = _load_source_semantic_frame(
            source,
            record_id="semantic_metrics/trace",
            kind="trace",
        )
    return datasets


def retron_plot_rendered_files(plots_dir: Path, *, plot_id: str) -> list[str]:
    return sorted(path.name for path in plots_dir.glob(f"{plot_id}*.pdf"))


def load_retron_source_surface(source: RetronReviewSource) -> RetronReviewSourceSurface:
    if source.config_path is None:
        raise ValueError(f"retron_review: source {source.label!r} has no config path for scoped review")
    config_path = source.config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"retron_review: source config not found for scoped review: {config_path}")
    return _load_retron_source_surface(str(config_path))


def retron_visible_plot_specs(plot_specs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    redundant_plot_ids = _redundant_retron_surface_plot_ids(plot_specs)
    return tuple(dict(spec) for spec in plot_specs if str(spec.get("id", "")) not in redundant_plot_ids)


def load_retron_review_bundle(
    manifest_path: Path,
    *,
    source_root: Path | None = None,
) -> RetronReviewBundle:
    payload = _load_manifest_payload(manifest_path)
    sources = _resolve_sources(
        manifest_path,
        payload,
        source_root=source_root.expanduser().resolve() if source_root is not None else None,
    )
    if not sources:
        raise ValueError("retron_review: review manifest must declare at least one source entry")
    relevant_stress_map, sensor_target_map = _resolve_semantic_maps(payload, sources=sources)
    summary_frames = []
    trace_frames = []
    for source in sources:
        summary_frame = _load_source_semantic_frame(
            source,
            record_id="semantic_metrics/summary",
            kind="summary",
        )
        trace_frame = _load_source_semantic_frame(
            source,
            record_id="semantic_metrics/trace",
            kind="trace",
        )
        summary_frames.append(_annotate_source(summary_frame, source=source))
        trace_frames.append(_annotate_source(trace_frame, source=source))
    return RetronReviewBundle(
        manifest_path=manifest_path.resolve(),
        sources=tuple(sources),
        summary_df=pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
        trace_df=pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame(),
        relevant_stress_map=relevant_stress_map,
        sensor_target_map=sensor_target_map,
    )


def _load_retron_source_surface(path: str) -> RetronReviewSourceSurface:
    source_context = notebook_context.load_notebook_workbench_context(Path(path))
    plot_specs = _visible_source_plot_specs(source_context)
    plot_selector_rows, plot_catalog_rows = _source_surface_plot_rows(
        source_context=source_context,
        plot_specs=plot_specs,
    )
    record_paths = _source_surface_record_paths(source_context.outputs_dir)
    return RetronReviewSourceSurface(
        experiment_title=source_context.decl.experiment.title or source_context.decl.experiment.id,
        protocol_id=source_context.decl.experiment_semantics.protocol.id,
        plot_specs=plot_specs,
        plot_selector_rows=tuple(plot_selector_rows),
        plot_catalog_rows=tuple(plot_catalog_rows),
        record_paths=record_paths,
    )


def _visible_source_plot_specs(source_context: Any) -> tuple[dict[str, Any], ...]:
    return retron_visible_plot_specs(tuple(spec.to_dict() for spec in source_context.workbench.plots))


def _source_surface_plot_rows(
    *,
    source_context: Any,
    plot_specs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = []
    for plot_spec in plot_specs:
        plot_id = str(plot_spec.get("id", ""))
        rows.append(
            (
                retron_review_catalog.experiment_plot_display_order(plot_id),
                _source_surface_plot_row(source_context=source_context, plot_spec=plot_spec),
            )
        )
    rows.sort(key=lambda item: (item[0], item[1].selector_row["Stage"], item[1].selector_row["Plot"]))
    plot_selector_rows = [row.selector_row for _, row in rows]
    plot_catalog_rows = [row.catalog_row for _, row in rows]
    return plot_selector_rows, plot_catalog_rows


def _source_surface_plot_row(
    *,
    source_context: Any,
    plot_spec: Mapping[str, Any],
) -> _SourceSurfacePlotRows:
    plot_id = str(plot_spec.get("id", ""))
    guide = retron_review_catalog.experiment_plot_guide(plot_id)
    title = str((plot_spec.get("with") or {}).get("title") or guide.title or plot_id)
    rendered = _source_surface_rendered(source_context=source_context, plot_id=plot_id, plot_spec=plot_spec)
    selector_title = str(guide.selector_title or title).strip() or str(title)
    selector_tag = str(guide.selector_tag or "").strip()
    selector_label = selector_title if not selector_tag else f"[{selector_tag}] {selector_title}"
    return _SourceSurfacePlotRows(
        selector_row={
            "Selector label": selector_label,
            "Stage": guide.stage,
            "Plot": title,
            "Plot id": plot_id,
        },
        catalog_row={
            "Stage": guide.stage,
            "Plot": title,
            "Plot id": plot_id,
            "Rendered": rendered,
            "Math / transform": guide.math,
            "How to read": guide.meaning,
        },
    )


def _source_surface_rendered(
    *,
    source_context: Any,
    plot_id: str,
    plot_spec: Mapping[str, Any],
) -> str:
    rendered_files = retron_plot_rendered_files(
        source_context.plots_dir,
        plot_id=plot_id,
    )
    return "yes" if rendered_files else "no"


def _source_surface_record_paths(outputs_dir: Path) -> tuple[tuple[str, str], ...]:
    record_info, _, _, _ = discover_dataframe_records(outputs_dir, allow_scan=False)
    return tuple(
        sorted(
            (
                str(info.get("record_id")),
                str(Path(info["path"]).expanduser().resolve()),
            )
            for info in record_info.values()
            if info.get("record_id") and info.get("path")
        )
    )


def _redundant_retron_surface_plot_ids(plot_specs: Sequence[Mapping[str, Any]]) -> set[str]:
    plot_ids = {str(spec.get("id", "")) for spec in plot_specs}
    redundant: set[str] = set()
    if {"raw_kinetics", "support_kinetics"}.issubset(plot_ids):
        redundant.add("support_kinetics")
    redundant.update({"baseline_shifted_kinetics", "stress_modulation_scores", "pareto_ranking"} & plot_ids)
    return redundant


def _load_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"retron_review: review manifest not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("retron_review: review manifest must be a mapping")
    return payload


def _resolve_sources(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    source_root: Path | None = None,
) -> list[RetronReviewSource]:
    return [
        _resolve_source_entry(
            manifest_path=manifest_path,
            idx=idx,
            raw_source=raw_source,
            source_root=source_root,
        )
        for idx, raw_source in _validated_manifest_sources(payload)
    ]


def _validated_manifest_sources(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("retron_review: manifest 'sources' must be a list")
    validated: list[tuple[int, dict[str, Any]]] = []
    for idx, raw_source in enumerate(raw_sources, start=1):
        if not isinstance(raw_source, dict):
            raise ValueError(f"retron_review: sources[{idx}] must be a mapping")
        validated.append((idx, raw_source))
    return validated


def _resolve_source_entry(
    *,
    manifest_path: Path,
    idx: int,
    raw_source: dict[str, Any],
    source_root: Path | None,
) -> RetronReviewSource:
    label = _source_label(raw_source, idx=idx)
    paths = _resolve_source_paths(
        manifest_path=manifest_path,
        raw_source=raw_source,
        source_root=source_root,
    )
    _ensure_source_tables_exist(label=label, paths=paths)
    return RetronReviewSource(
        label=label,
        experiment_id=_source_experiment_id(raw_source, experiment_root=paths.experiment_root, label=label),
        experiment_root=paths.experiment_root,
        config_path=paths.config_path,
        summary_path=paths.summary_path.resolve(),
        trace_path=paths.trace_path.resolve(),
    )


def _source_label(raw_source: Mapping[str, Any], *, idx: int) -> str:
    return str(raw_source.get("label") or raw_source.get("family") or f"source_{idx}").strip()


def _source_experiment_id(
    raw_source: Mapping[str, Any],
    *,
    experiment_root: Path | None,
    label: str,
) -> str:
    return str(raw_source.get("experiment_id") or (experiment_root.name if experiment_root is not None else label))


def _resolve_source_paths(
    *,
    manifest_path: Path,
    raw_source: Mapping[str, Any],
    source_root: Path | None,
) -> _ResolvedSourcePaths:
    _validate_source_mode(raw_source)
    experiment_root, config_path = _resolve_source_scope(
        manifest_path=manifest_path,
        raw_source=raw_source,
        source_root=source_root,
    )
    return _ResolvedSourcePaths(
        experiment_root=experiment_root,
        config_path=config_path,
        summary_path=_resolve_source_export_path(
            manifest_path=manifest_path,
            raw_source=raw_source,
            field="summary",
            experiment_root=experiment_root,
            record_id="semantic_metrics/summary",
        ),
        trace_path=_resolve_source_export_path(
            manifest_path=manifest_path,
            raw_source=raw_source,
            field="trace",
            experiment_root=experiment_root,
            record_id="semantic_metrics/trace",
        ),
    )


def _validate_source_mode(raw_source: Mapping[str, Any]) -> None:
    scope_fields = [field for field in ("experiment", "config") if raw_source.get(field) is not None]
    explicit_fields = [field for field in ("summary", "trace") if raw_source.get(field) is not None]
    if len(scope_fields) > 1:
        raise ValueError("retron_review: a source must not declare both 'experiment' and 'config'")
    if scope_fields and explicit_fields:
        raise ValueError(
            "retron_review: experiment/config sources use cataloged semantic records and must not declare "
            "explicit summary or trace paths"
        )


def _resolve_source_scope(
    *,
    manifest_path: Path,
    raw_source: Mapping[str, Any],
    source_root: Path | None,
) -> tuple[Path | None, Path | None]:
    experiment_raw = raw_source.get("experiment")
    config_raw = raw_source.get("config")
    if experiment_raw is not None:
        experiment_root = _resolve_manifest_path(
            manifest_path,
            str(experiment_raw),
            relative_to=source_root,
        )
        return experiment_root, experiment_root / "config.yaml"
    if config_raw is not None:
        config_path = _resolve_manifest_path(
            manifest_path,
            str(config_raw),
            relative_to=source_root,
        )
        return config_path.parent, config_path
    return None, None


def _resolve_source_export_path(
    *,
    manifest_path: Path,
    raw_source: Mapping[str, Any],
    field: str,
    experiment_root: Path | None,
    record_id: str,
) -> Path:
    raw_value = raw_source.get(field)
    if raw_value is not None:
        return _resolve_manifest_path(manifest_path, str(raw_value))
    if experiment_root is None:
        raise ValueError(
            "retron_review: each source must declare either 'experiment' or explicit 'summary'/'trace' paths"
        )
    return _resolve_default_semantic_table(experiment_root, record_id=record_id)


def _ensure_source_tables_exist(*, label: str, paths: _ResolvedSourcePaths) -> None:
    missing = [str(path) for path in (paths.summary_path, paths.trace_path) if not path.exists()]
    if not missing:
        return
    command = ""
    if paths.config_path is not None:
        command = f" Run 'uv run reader run {paths.config_path}'."
    raise FileNotFoundError(f"retron_review: source semantic tables are missing for {label}: {missing}.{command}")


def _resolve_semantic_maps(
    payload: dict[str, Any],
    *,
    sources: list[RetronReviewSource],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    manifest_relevant = payload.get("relevant_stress_map")
    manifest_targets = payload.get("sensor_target_map")
    relevant_stress_map = _normalize_relevant_stress_map(manifest_relevant or {})
    sensor_target_map = _normalize_sensor_target_map(manifest_targets or {})
    if relevant_stress_map and sensor_target_map:
        return relevant_stress_map, sensor_target_map
    derived_relevant: dict[str, str] = {}
    derived_targets: dict[str, tuple[str, ...]] = {}
    for source in sources:
        if source.config_path is None or not source.config_path.exists():
            continue
        config = yaml.safe_load(source.config_path.read_text(encoding="utf-8")) or {}
        analysis = ((config.get("protocol") or {}).get("analysis") or {}).get("semantic_metrics") or {}
        if not isinstance(analysis, dict):
            continue
        candidate_relevant = _normalize_relevant_stress_map(analysis.get("relevant_stress_map") or {})
        candidate_targets = _normalize_sensor_target_map(analysis.get("sensor_target_map") or {})
        derived_relevant = _merge_semantic_map(
            derived_relevant,
            candidate_relevant,
            label="relevant_stress_map",
        )
        derived_targets = _merge_semantic_map(
            derived_targets,
            candidate_targets,
            label="sensor_target_map",
        )
    relevant_stress_map = relevant_stress_map or derived_relevant
    sensor_target_map = sensor_target_map or derived_targets
    if not relevant_stress_map:
        raise ValueError("retron_review: manifest must declare relevant_stress_map or point to source configs that do")
    if not sensor_target_map:
        raise ValueError("retron_review: manifest must declare sensor_target_map or point to source configs that do")
    return relevant_stress_map, sensor_target_map


def _merge_semantic_map(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in candidate.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"retron_review: inconsistent {label} for {key!r}: {merged[key]!r} vs {value!r}")
        merged[key] = value
    return merged


def _normalize_relevant_stress_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("retron_review: relevant_stress_map must be a mapping when provided")
    return {str(key): str(item) for key, item in value.items()}


def _normalize_sensor_target_map(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ValueError("retron_review: sensor_target_map must be a mapping when provided")
    normalized: dict[str, tuple[str, ...]] = {}
    for key, items in value.items():
        if not isinstance(items, list):
            raise ValueError("retron_review: sensor_target_map values must be lists of motif labels")
        normalized[str(key)] = tuple(str(item) for item in items)
    return normalized


def _resolve_manifest_path(
    manifest_path: Path,
    raw: str,
    *,
    relative_to: Path | None = None,
) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        anchor = relative_to if relative_to is not None else manifest_path.parent
        path = (anchor / path).resolve()
    return path


def _resolve_default_semantic_table(experiment_root: Path, *, record_id: str) -> Path:
    return _resolve_semantic_record(experiment_root / "outputs", record_id=record_id).path.resolve()


def _resolve_semantic_record(outputs_dir: Path, *, record_id: str) -> DataFrameArtifactRecord:
    expected_contract = _SEMANTIC_CONTRACT_BY_RECORD_ID.get(record_id)
    if expected_contract is None:
        raise ValueError(f"retron_review: unsupported semantic record id {record_id!r}")
    store = builtin_runtime().record_store(outputs_dir, create=False)
    if not store.catalog_exists():
        raise FileNotFoundError(
            f"retron_review: record catalog is missing for {record_id!r}: {store.records_path}. "
            "Run the source experiment before opening the review."
        )
    try:
        record = store.read_dataframe(record_id)
    except RecordError as exc:
        raise FileNotFoundError(
            f"retron_review: required dataframe record {record_id!r} is missing from {store.records_path}. "
            "Run the source experiment before opening the review."
        ) from exc
    if record.contract_id != expected_contract:
        raise RecordError(
            f"retron_review: record {record_id!r} declares contract {record.contract_id!r}; "
            f"expected {expected_contract!r}"
        )
    return record


def _load_source_semantic_frame(
    source: RetronReviewSource,
    *,
    record_id: str,
    kind: str,
) -> pd.DataFrame:
    source_path = source.summary_path if kind == "summary" else source.trace_path
    if source.experiment_root is None:
        return load_cached_semantic_frame(source_path, kind=kind)
    outputs_dir = source.experiment_root / "outputs"
    record = _resolve_semantic_record(outputs_dir, record_id=record_id)
    if record.path.resolve() != source_path.resolve():
        raise RecordError(f"retron_review: source path for {record_id!r} does not match the current cataloged record")
    stat = record.path.stat()
    return _load_cached_record_semantic_frame(
        str(outputs_dir.resolve()),
        record_id,
        record.content_digest,
        stat.st_mtime_ns,
        stat.st_size,
        kind,
    )


def _read_semantic_table(path: Path, *, kind: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"retron_review: unsupported semantic table format for {path}")
    return _normalize_semantic_frame(frame, kind=kind)


def _normalize_semantic_frame(frame: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    if kind not in {"summary", "trace"}:
        raise ValueError(f"retron_review: unsupported semantic table kind {kind!r}")
    normalized = frame.copy()
    for column in (
        "plate_id",
        "sensor",
        "sponge",
        "genotype_id",
        "replicate_id",
        "stress_condition",
        "IPTG",
        "metric",
        "sponge_family_size",
        "matched_tetO_group",
        "matched_control_key",
    ):
        if column in normalized.columns:
            normalized[column] = normalized[column].astype("string")
    if "value" in normalized.columns:
        normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    for column in ("relevant_sensor_pair", "is_relevant_stress"):
        if column in normalized.columns:
            normalized[column] = retron_review_semantics.coerce_optional_bool_series(
                normalized[column],
                label=column,
            )
    if kind == "trace":
        for column in _TRACE_FLAG_COLUMNS:
            if column in normalized.columns:
                normalized[column] = retron_review_semantics.coerce_optional_bool_series(
                    normalized[column],
                    label=column,
                )
    normalized["source_kind"] = kind
    return normalized


def _validate_semantic_frame(frame: pd.DataFrame, *, record_id: str, where: str) -> None:
    contract_id = _SEMANTIC_CONTRACT_BY_RECORD_ID.get(record_id)
    if contract_id is None:
        raise ValueError(f"retron_review: unsupported semantic record id {record_id!r}")
    builtin_runtime().contracts.validate(frame, contract_id=contract_id, where=f"retron_review:{where}")


def _annotate_source(frame: pd.DataFrame, *, source: RetronReviewSource) -> pd.DataFrame:
    out = frame.copy()
    out["source_label"] = source.label
    out["source_experiment_id"] = source.experiment_id
    out["source_summary_path"] = str(source.summary_path)
    out["source_trace_path"] = str(source.trace_path)
    return out
