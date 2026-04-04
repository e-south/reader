from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reader.plugins.transform.retron_sponge_metrics import RetronSpongeMetricsCfg
from reader.workbench.notebooks import _retron_review_catalog as retron_review_catalog
from reader.workbench.notebooks import _retron_review_shared as retron_review_shared
from reader.workbench.notebooks import context as notebook_context
from reader.workbench.records import discover_dataframe_records


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


_SUMMARY_IDENTITY_COLUMNS = (
    "plate_id",
    "sensor",
    "sponge",
    "genotype_id",
    "stress_condition",
    "IPTG",
)
_DEFAULT_MIN_ABS_G_SENSOR = float(RetronSpongeMetricsCfg().min_abs_g_sensor)
_LEGACY_TRACE_FLAG_COLUMNS = (
    "in_pre_window",
    "in_primary_post_stress",
    "in_endpoint_window",
)
_DEFAULT_CONTROL_NAME = str(RetronSpongeMetricsCfg().control_name)
_LEGACY_TRACE_SCOPE_COLUMNS = (
    "plate_id",
    "sensor",
    "sponge",
    "stress_condition",
)


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


def load_cached_semantic_frame(
    path: str | Path,
    *,
    kind: str,
    min_abs_g_sensor: float = _DEFAULT_MIN_ABS_G_SENSOR,
    control_name: str = _DEFAULT_CONTROL_NAME,
) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _load_cached_semantic_frame(
        str(resolved),
        stat.st_mtime_ns,
        stat.st_size,
        kind,
        float(min_abs_g_sensor),
        str(control_name),
    )


@lru_cache(maxsize=128)
def _load_cached_semantic_frame(
    path: str,
    mtime_ns: int,
    size_bytes: int,
    kind: str,
    min_abs_g_sensor: float,
    control_name: str,
) -> pd.DataFrame:
    del mtime_ns, size_bytes
    return _read_semantic_table(
        Path(path),
        kind=kind,
        min_abs_g_sensor=min_abs_g_sensor,
        control_name=control_name,
    )


def load_retron_source_semantic_datasets(
    source: RetronReviewSource,
    *,
    record_ids: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    min_abs_g_sensor = _resolve_min_abs_g_sensor(source.config_path)
    control_name = _resolve_control_name(source.config_path)
    selected = set(record_ids or ("semantic_metrics/summary", "semantic_metrics/trace"))
    datasets: dict[str, pd.DataFrame] = {}
    if "semantic_metrics/summary" in selected:
        datasets["semantic_metrics/summary"] = load_cached_semantic_frame(
            source.summary_path,
            kind="summary",
            min_abs_g_sensor=min_abs_g_sensor,
            control_name=control_name,
        )
    if "semantic_metrics/trace" in selected:
        datasets["semantic_metrics/trace"] = load_cached_semantic_frame(
            source.trace_path,
            kind="trace",
            min_abs_g_sensor=min_abs_g_sensor,
            control_name=control_name,
        )
    return datasets


def retron_plot_rendered_files(plots_dir: Path, *, plot_id: str, plugin: str) -> list[str]:
    patterns = [f"{plot_id}*.pdf"]
    if str(plot_id) == "raw_kinetics" and str(plugin) == "plot/time_series":
        patterns.append("ts_*.pdf")
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(path.name for path in plots_dir.glob(pattern))
    return sorted(set(matches))


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
        min_abs_g_sensor = _resolve_min_abs_g_sensor(source.config_path)
        control_name = _resolve_control_name(source.config_path)
        summary_frame = load_cached_semantic_frame(
            source.summary_path,
            kind="summary",
            min_abs_g_sensor=min_abs_g_sensor,
            control_name=control_name,
        )
        trace_frame = load_cached_semantic_frame(
            source.trace_path,
            kind="trace",
            min_abs_g_sensor=min_abs_g_sensor,
            control_name=control_name,
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
        plugin=str(plot_spec.get("plugin", "")),
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
    _ensure_source_exports_exist(label=label, paths=paths)
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
            export_name="semantic_summary.csv",
        ),
        trace_path=_resolve_source_export_path(
            manifest_path=manifest_path,
            raw_source=raw_source,
            field="trace",
            experiment_root=experiment_root,
            record_id="semantic_metrics/trace",
            export_name="semantic_trace.csv",
        ),
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
    export_name: str,
) -> Path:
    raw_value = raw_source.get(field)
    if raw_value is not None:
        return _resolve_manifest_path(manifest_path, str(raw_value))
    if experiment_root is None:
        raise ValueError(
            "retron_review: each source must declare either 'experiment' or explicit 'summary'/'trace' paths"
        )
    return _resolve_default_semantic_table(
        experiment_root,
        record_id=record_id,
        export_name=export_name,
    )


def _ensure_source_exports_exist(*, label: str, paths: _ResolvedSourcePaths) -> None:
    missing = [str(path) for path in (paths.summary_path, paths.trace_path) if not path.exists()]
    if not missing:
        return
    command = ""
    if paths.config_path is not None:
        command = f" Run 'uv run reader run {paths.config_path}' and 'uv run reader export {paths.config_path}'."
    raise FileNotFoundError(f"retron_review: source exports are missing for {label}: {missing}.{command}")


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


def _resolve_default_semantic_table(experiment_root: Path, *, record_id: str, export_name: str) -> Path:
    export_path = experiment_root / "outputs" / "exports" / "retron" / export_name
    if export_path.exists():
        return export_path
    record_info, _, _, _ = discover_dataframe_records(experiment_root / "outputs", allow_scan=False)
    for info in record_info.values():
        if str(info.get("record_id")) == str(record_id):
            return Path(info["path"]).resolve()
    return export_path


def _resolve_min_abs_g_sensor(config_path: Path | None) -> float:
    if config_path is None or not config_path.exists():
        return _DEFAULT_MIN_ABS_G_SENSOR
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    analysis = ((payload.get("protocol") or {}).get("analysis") or {}).get("semantic_metrics") or {}
    if not isinstance(analysis, dict):
        return _DEFAULT_MIN_ABS_G_SENSOR
    raw_value = analysis.get("min_abs_g_sensor")
    if raw_value is None:
        return _DEFAULT_MIN_ABS_G_SENSOR
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"retron_review: protocol.analysis.semantic_metrics.min_abs_g_sensor must be numeric in {config_path}"
        ) from exc


def _resolve_control_name(config_path: Path | None) -> str:
    if config_path is None or not config_path.exists():
        return _DEFAULT_CONTROL_NAME
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    analysis = ((payload.get("protocol") or {}).get("analysis") or {}).get("semantic_metrics") or {}
    if not isinstance(analysis, dict):
        return _DEFAULT_CONTROL_NAME
    raw_value = analysis.get("control_name")
    return str(raw_value).strip() if raw_value is not None and str(raw_value).strip() else _DEFAULT_CONTROL_NAME


def _read_semantic_table(
    path: Path,
    *,
    kind: str,
    min_abs_g_sensor: float = _DEFAULT_MIN_ABS_G_SENSOR,
    control_name: str = _DEFAULT_CONTROL_NAME,
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"retron_review: unsupported semantic table format for {path}")
    if "metric" in frame.columns:
        frame["metric"] = frame["metric"].astype(str)
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    for column in ("relevant_sensor_pair", "is_relevant_stress"):
        if column in frame.columns:
            frame[column] = retron_review_shared.coerce_optional_bool_series(frame[column], label=column)
    if kind == "trace":
        for column in _LEGACY_TRACE_FLAG_COLUMNS:
            if column in frame.columns:
                frame[column] = retron_review_shared.coerce_optional_bool_series(frame[column], label=column)
    if kind == "summary":
        frame = _upgrade_legacy_summary_metrics(
            frame,
            min_abs_g_sensor=min_abs_g_sensor,
            control_name=control_name,
        )
    elif kind == "trace":
        frame = _upgrade_legacy_trace_contract(frame)
    frame["source_kind"] = kind
    return frame


def _upgrade_legacy_trace_contract(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    upgraded = frame.copy()
    for column in ("time", "time_from_stress", "configured_max_post_stress_hours"):
        if column in upgraded.columns:
            upgraded[column] = pd.to_numeric(upgraded[column], errors="coerce")
    if "matched_control_key" not in upgraded.columns and {
        "plate_id",
        "sensor",
        "stress_condition",
    }.issubset(upgraded.columns):
        upgraded["matched_control_key"] = upgraded.apply(
            lambda row: f"{row['plate_id']}::{row['sensor']}::{row['stress_condition']}",
            axis=1,
        )
    if "stress_time_zero_h" not in upgraded.columns and {"time", "time_from_stress"}.issubset(upgraded.columns):
        upgraded["stress_time_zero_h"] = pd.to_numeric(upgraded["time"], errors="coerce") - pd.to_numeric(
            upgraded["time_from_stress"],
            errors="coerce",
        )
    metadata = _legacy_trace_scope_metadata(upgraded)
    if not metadata.empty:
        upgraded = upgraded.merge(
            metadata,
            on=[column for column in _LEGACY_TRACE_SCOPE_COLUMNS if column in upgraded.columns],
            how="left",
            validate="many_to_one",
            suffixes=("", "__legacy"),
        )
        upgraded = _coalesce_string_column(upgraded, "matched_control_key")
        for column in (
            "summary_window_start_h",
            "summary_window_end_h",
            "summary_window_duration_h",
            "pre_stress_read_count",
            "post_stress_read_count",
            "matched_group_sample_count",
            "stress_time_zero_h",
            "stress_addition_gap_h",
        ):
            upgraded = _coalesce_numeric_column(upgraded, column)
    return upgraded


def _legacy_trace_scope_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "plate_id",
        "sensor",
        "sponge",
        "stress_condition",
        "replicate_id",
        "time",
        "time_from_stress",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    stress_gap_by_plate = _legacy_trace_stress_gap_by_plate(frame)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(_LEGACY_TRACE_SCOPE_COLUMNS), dropna=False, sort=False):
        plate_id, sensor, sponge, stress_condition = keys
        post = group[group.get("in_primary_post_stress", pd.Series(False, index=group.index)).fillna(False)].copy()
        time_from_stress = pd.to_numeric(post.get("time_from_stress"), errors="coerce").to_numpy(dtype=float)
        finite = time_from_stress[np.isfinite(time_from_stress)]
        if finite.size == 0:
            start_h = end_h = duration_h = np.nan
        else:
            start_h = float(finite.min())
            end_h = float(finite.max())
            duration_h = float(end_h - start_h)
        rows.append(
            {
                "plate_id": plate_id,
                "sensor": sensor,
                "sponge": sponge,
                "stress_condition": stress_condition,
                "matched_control_key": f"{plate_id}::{sensor}::{stress_condition}",
                "summary_window_start_h": start_h,
                "summary_window_end_h": end_h,
                "summary_window_duration_h": duration_h,
                "pre_stress_read_count": _legacy_flagged_unique_time_count(group, flag_column="in_pre_window"),
                "post_stress_read_count": _legacy_flagged_unique_time_count(
                    group,
                    flag_column="in_primary_post_stress",
                ),
                "matched_group_sample_count": float(group["replicate_id"].astype(str).nunique()),
                "stress_time_zero_h": _legacy_stress_time_zero(group),
                "stress_addition_gap_h": stress_gap_by_plate.get(_normalize_key_component(plate_id), np.nan),
            }
        )
    return pd.DataFrame(rows)


def _legacy_stress_time_zero(frame: pd.DataFrame) -> float:
    if not {"time", "time_from_stress"}.issubset(frame.columns):
        return float("nan")
    stress_zero = pd.to_numeric(frame["time"], errors="coerce") - pd.to_numeric(
        frame["time_from_stress"], errors="coerce"
    )
    finite = stress_zero[np.isfinite(stress_zero)]
    if finite.empty:
        return float("nan")
    return float(finite.median())


def _legacy_trace_stress_gap_by_plate(frame: pd.DataFrame) -> dict[object, float]:
    if "time_from_stress" not in frame.columns or "plate_id" not in frame.columns:
        return {}
    resolved: dict[object, float] = {}
    for plate_id, group in frame.groupby("plate_id", dropna=False, sort=False):
        offsets = np.sort(pd.unique(pd.to_numeric(group["time_from_stress"], errors="coerce").dropna()))
        pre = offsets[offsets < 0]
        post = offsets[offsets > 0]
        resolved[_normalize_key_component(plate_id)] = (
            float("nan") if len(pre) == 0 or len(post) == 0 else float(post[0] - pre[-1])
        )
    return resolved


def _legacy_flagged_unique_time_count(frame: pd.DataFrame, *, flag_column: str) -> float:
    if (
        frame.empty
        or flag_column not in frame.columns
        or "replicate_id" not in frame.columns
        or "time" not in frame.columns
    ):
        return float("nan")
    flagged = frame[frame[flag_column].fillna(False)].copy()
    if flagged.empty:
        return float("nan")
    counts = flagged.groupby("replicate_id", dropna=False)["time"].nunique().to_numpy(dtype=float, copy=False)
    finite = counts[np.isfinite(counts)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _coalesce_numeric_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    legacy_column = f"{column}__legacy"
    if legacy_column not in frame.columns:
        return frame
    existing = pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")
    legacy = pd.to_numeric(frame[legacy_column], errors="coerce")
    frame[column] = existing.where(existing.notna(), legacy)
    return frame.drop(columns=[legacy_column])


def _coalesce_string_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    legacy_column = f"{column}__legacy"
    if legacy_column not in frame.columns:
        return frame
    existing = frame.get(column, pd.Series(pd.NA, index=frame.index))
    frame[column] = existing.where(existing.notna(), frame[legacy_column])
    return frame.drop(columns=[legacy_column])


def _upgrade_legacy_summary_metrics(
    frame: pd.DataFrame,
    *,
    min_abs_g_sensor: float,
    control_name: str,
) -> pd.DataFrame:
    upgraded = frame.copy()
    upgraded = _append_preload_metric(
        upgraded,
        source_metric="R_pre",
        target_metric="P_pre",
        control_name=control_name,
    )
    del min_abs_g_sensor
    return upgraded


def _append_preload_metric(
    frame: pd.DataFrame,
    *,
    source_metric: str,
    target_metric: str,
    control_name: str,
) -> pd.DataFrame:
    required = {
        "metric",
        "plate_id",
        "sensor",
        "sponge",
        "genotype_id",
        "stress_condition",
        "IPTG",
        "value",
    }
    if not required.issubset(frame.columns):
        return frame
    source_rows = frame[frame["metric"] == str(source_metric)].copy()
    if source_rows.empty:
        return frame
    control_rows = source_rows[source_rows["sponge"].astype(str) == str(control_name)].copy()
    sample_rows = source_rows[source_rows["sponge"].astype(str) != str(control_name)].copy()
    if control_rows.empty or sample_rows.empty:
        return frame
    control_lookup = (
        control_rows.rename(columns={"value": "control_value"})
        .loc[:, ["plate_id", "sensor", "stress_condition", "IPTG", "control_value"]]
        .drop_duplicates()
    )
    preload_rows = sample_rows.merge(
        control_lookup,
        on=["plate_id", "sensor", "stress_condition", "IPTG"],
        how="left",
        validate="many_to_one",
    )
    preload_rows["value"] = pd.to_numeric(preload_rows["value"], errors="coerce") - pd.to_numeric(
        preload_rows["control_value"],
        errors="coerce",
    )
    candidate_rows = _summary_difference_by_state(
        preload_rows,
        value_column="value",
        positive_state="+IPTG",
        negative_state="-IPTG",
        target_metric=target_metric,
    )
    candidate_rows = _filter_missing_metric_rows(frame, candidate_rows, target_metric=target_metric)
    if candidate_rows.empty:
        return frame
    return pd.concat([frame, candidate_rows], ignore_index=True)


def _append_expected_direction_metric(
    frame: pd.DataFrame,
    *,
    source_metric: str,
    target_metric: str,
) -> pd.DataFrame:
    if "metric" not in frame.columns or "expected_decoy_sign" not in frame.columns:
        return frame
    source_rows = frame[frame["metric"] == str(source_metric)].copy()
    if source_rows.empty:
        return frame
    source_rows["expected_decoy_sign"] = pd.to_numeric(source_rows["expected_decoy_sign"], errors="coerce")
    source_rows["value"] = pd.to_numeric(source_rows["value"], errors="coerce") * source_rows["expected_decoy_sign"]
    source_rows["metric"] = str(target_metric)
    source_rows = _filter_missing_metric_rows(frame, source_rows, target_metric=target_metric)
    if source_rows.empty:
        return frame
    return pd.concat([frame, source_rows], ignore_index=True)


def _append_scaled_metric(
    frame: pd.DataFrame,
    *,
    source_metric: str,
    target_metric: str,
    min_abs_g_sensor: float,
) -> pd.DataFrame:
    if "metric" not in frame.columns or "sensor" not in frame.columns or "plate_id" not in frame.columns:
        return frame
    source_rows = frame[frame["metric"] == str(source_metric)].copy()
    if "is_relevant_stress" in source_rows.columns:
        source_rows = source_rows[source_rows["is_relevant_stress"].fillna(False)]
    if source_rows.empty:
        return frame
    g_sensor_rows = frame[frame["metric"] == "G_sensor"].copy()
    if g_sensor_rows.empty:
        return frame
    g_sensor_rows["value"] = pd.to_numeric(g_sensor_rows["value"], errors="coerce")
    g_sensor_lookup = {
        (
            _normalize_key_component(row.get("plate_id")),
            _normalize_key_component(row.get("sensor")),
        ): row["value"]
        for _, row in g_sensor_rows.iterrows()
    }
    scaled_rows: list[dict[str, Any]] = []
    for _, row in source_rows.iterrows():
        native = g_sensor_lookup.get(
            (
                _normalize_key_component(row.get("plate_id")),
                _normalize_key_component(row.get("sensor")),
            )
        )
        native_abs = np.nan if native is None or not np.isfinite(native) else abs(float(native))
        scaled_row = row.to_dict()
        if not np.isfinite(native_abs):
            scaled_row["value"] = np.nan
            scaled_row["scaling_available"] = False
            scaled_row["warning_flag"] = "missing_G_sensor"
        elif native_abs < float(min_abs_g_sensor):
            scaled_row["value"] = np.nan
            scaled_row["scaling_available"] = False
            scaled_row["warning_flag"] = "unstable_scaled_metric"
        else:
            scaled_row["value"] = float(row["value"]) / native_abs
            scaled_row["scaling_available"] = True
            scaled_row["warning_flag"] = None
        scaled_row["scale_reference_abs_g_sensor"] = native_abs
        scaled_row["scale_min_abs_g_sensor"] = float(min_abs_g_sensor)
        scaled_row["metric"] = str(target_metric)
        scaled_rows.append(scaled_row)
    candidate_rows = _filter_missing_metric_rows(
        frame,
        pd.DataFrame(scaled_rows),
        target_metric=target_metric,
    )
    if candidate_rows.empty:
        return frame
    return pd.concat([frame, candidate_rows], ignore_index=True)


def _summary_difference_by_state(
    frame: pd.DataFrame,
    *,
    value_column: str,
    positive_state: str,
    negative_state: str,
    target_metric: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    index_columns = [
        column
        for column in (
            "plate_id",
            "sensor",
            "sponge",
            "genotype_id",
            "stress_condition",
            "expected_decoy_sign",
            "is_relevant_stress",
            "relevant_sensor_pair",
            "sponge_family_size",
            "matched_control_key",
            "summary_window_start_h",
            "summary_window_end_h",
            "summary_window_duration_h",
            "pre_stress_read_count",
            "post_stress_read_count",
            "matched_group_sample_count",
            "stress_time_zero_h",
            "stress_addition_gap_h",
        )
        if column in frame.columns
    ]
    pivot = frame.pivot_table(index=index_columns, columns="IPTG", values=value_column, aggfunc="first").reset_index()
    if positive_state not in pivot.columns or negative_state not in pivot.columns:
        return pd.DataFrame()
    pivot["metric"] = str(target_metric)
    pivot["IPTG"] = pd.NA
    pivot["value"] = pd.to_numeric(pivot[positive_state], errors="coerce") - pd.to_numeric(
        pivot[negative_state],
        errors="coerce",
    )
    keep_columns = [*index_columns, "IPTG", "metric", "value"]
    return pivot.loc[:, keep_columns]


def _filter_missing_metric_rows(
    frame: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    *,
    target_metric: str,
) -> pd.DataFrame:
    if candidate_rows.empty:
        return candidate_rows
    identity_columns = [column for column in _SUMMARY_IDENTITY_COLUMNS if column in candidate_rows.columns]
    if not identity_columns:
        return (
            candidate_rows
            if str(target_metric) not in set(frame.get("metric", pd.Series(dtype=str)).astype(str))
            else pd.DataFrame()
        )
    existing_rows = frame[frame["metric"] == str(target_metric)]
    existing_keys = {_summary_row_identity(row, identity_columns) for _, row in existing_rows.iterrows()}
    keep_mask = [
        _summary_row_identity(row, identity_columns) not in existing_keys for _, row in candidate_rows.iterrows()
    ]
    return candidate_rows.loc[keep_mask].reset_index(drop=True)


def _summary_row_identity(row: Mapping[str, Any], identity_columns: Sequence[str]) -> tuple[object, ...]:
    return tuple(_normalize_key_component(row.get(column)) for column in identity_columns)


def _normalize_key_component(value: object) -> object:
    return None if pd.isna(value) else value


def _annotate_source(frame: pd.DataFrame, *, source: RetronReviewSource) -> pd.DataFrame:
    out = frame.copy()
    out["source_label"] = source.label
    out["source_experiment_id"] = source.experiment_id
    out["source_summary_path"] = str(source.summary_path)
    out["source_trace_path"] = str(source.trace_path)
    return out
