from __future__ import annotations

from pathlib import Path
from typing import Any

from reader.domains.logic.sfxi.vec8_aggregate.constants import SFXI_VEC8_RECORD_ID
from reader.domains.logic.sfxi.vec8_aggregate.model import (
    LoadedSFXIVec8Source,
    SFXIVec8Aggregate,
    SFXIVec8AggregateArtifacts,
)
from reader.domains.logic.sfxi.vec8_aggregate.sources import (
    aggregate_sfxi_vec8_sources,
    load_sfxi_vec8_table,
)
from reader.domains.logic.sfxi.vec8_aggregate.writer import write_sfxi_vec8_aggregate as write_aggregate_bundle
from reader.errors import RecordError, SFXIError
from reader.workbench.config import ReaderSpec
from reader.workbench.paths import resolve_path_within_root

from . import builtin_runtime


def load_sfxi_vec8_sources(sources: list[str | Path] | tuple[str | Path, ...]) -> SFXIVec8Aggregate:
    """Resolve workbench sources, then delegate vec8 semantics to the SFXI domain."""

    return aggregate_sfxi_vec8_sources([_load_source(Path(source).expanduser()) for source in sources])


def write_sfxi_vec8_aggregate(
    *,
    sources: list[str | Path] | tuple[str | Path, ...],
    out_dir: Path,
    title: str | None = None,
    filename: str = "sfxi_vec8_heatmap",
    dpi: int = 300,
    overwrite: bool = False,
) -> SFXIVec8AggregateArtifacts:
    """Resolve source records and atomically write the domain-owned artifact bundle."""

    return write_aggregate_bundle(
        aggregate=load_sfxi_vec8_sources(sources),
        out_dir=out_dir,
        title=title,
        filename=filename,
        dpi=dpi,
        overwrite=overwrite,
    )


def _load_source(path: Path) -> LoadedSFXIVec8Source:
    resolved = path.resolve()
    if resolved.is_dir():
        config_path = resolved / "config.yaml"
        if config_path.exists():
            return _load_experiment_config(config_path)
        records_path = resolved / "manifests" / "records.json"
        if records_path.exists():
            return _load_outputs_dir(resolved, source_id=resolved.parent.name, source_path=resolved)
        raise SFXIError(
            "SFXI vec8 aggregate directory sources must be experiment directories, outputs directories, "
            f"or table files: {resolved}"
        )
    if not resolved.exists():
        raise SFXIError(f"SFXI vec8 aggregate source does not exist: {resolved}")
    if resolved.name == "config.yaml" or resolved.suffix.lower() in {".yaml", ".yml"}:
        return _load_experiment_config(resolved)
    return load_sfxi_vec8_table(resolved)


def _load_experiment_config(config_path: Path) -> LoadedSFXIVec8Source:
    spec = ReaderSpec.load(config_path)
    root = config_path.parent.resolve()
    try:
        outputs_dir = resolve_path_within_root(spec.paths.outputs, root=root)
    except ValueError as exc:
        raise SFXIError(f"SFXI vec8 aggregate could not resolve paths.outputs for {config_path}.") from exc
    return _load_outputs_dir(outputs_dir, source_id=spec.experiment.id, source_path=config_path.resolve())


def _load_outputs_dir(
    outputs_dir: Path,
    *,
    source_id: str,
    source_path: Path,
) -> LoadedSFXIVec8Source:
    store = builtin_runtime().record_store(outputs_dir, create=False)
    if not store.catalog_exists():
        raise SFXIError(
            f"SFXI vec8 aggregate could not find {SFXI_VEC8_RECORD_ID!r} because records catalog is missing under "
            f"{outputs_dir}. "
            "Run `uv run reader run <config>` first or pass an explicit vec8 table file."
        )
    try:
        record = store.latest_dataframe(SFXI_VEC8_RECORD_ID)
    except RecordError as exc:
        raise SFXIError(f"SFXI vec8 aggregate could not read records catalog: {store.records_path}") from exc
    if record is None:
        raise SFXIError(
            f"SFXI vec8 aggregate could not find {SFXI_VEC8_RECORD_ID!r} under {outputs_dir}. "
            "Run `uv run reader run <config>` first or pass an explicit vec8 table file."
        )
    return LoadedSFXIVec8Source(
        source_id=source_id,
        source_path=source_path.resolve(),
        table_path=record.path.resolve(),
        source_kind="record",
        frame=_load_record_frame(record, outputs_dir=outputs_dir, record_id=SFXI_VEC8_RECORD_ID),
        record_id=SFXI_VEC8_RECORD_ID,
        record_metadata=_record_metadata(record),
    )


def _load_record_frame(record: Any, *, outputs_dir: Path, record_id: str):
    try:
        return record.load_dataframe()
    except Exception as exc:
        raise SFXIError(
            f"SFXI vec8 aggregate could not load {record_id!r} dataframe artifact under {outputs_dir}: {record.path}"
        ) from exc


def _record_metadata(record: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_id": record.contract_id,
        "content_digest": record.content_digest,
        "config_digest": record.config_digest,
        "created_at": record.created_at,
        "producer": record.producer.to_dict(),
    }
    if getattr(record, "code_digest", ""):
        payload["code_digest"] = record.code_digest
    return payload
