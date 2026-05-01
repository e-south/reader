"""
dnadesign boundary adapter for the SFXI triptych sequence plot.

This module is intentionally narrow: reader owns plot semantics and bundle
publication, while dnadesign owns USR access and BaseRender sequence panels.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from reader.errors import SFXIError

DNADESIGN_SEQUENCE_PANEL_CONTRACT_ID = "dnadesign.baserender.sequence_panel.v1"
READER_SUPPORTED_SEQUENCE_PANEL_CONTRACT_VERSION = "1"


def require_dnadesign_sequence_panel_api():
    try:
        baserender = importlib.import_module("dnadesign.baserender")
        usr = importlib.import_module("dnadesign.usr")
    except ImportError as exc:
        _raise_dnadesign_import_error(exc)

    actual = getattr(baserender, "BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION", None)
    if str(actual) != READER_SUPPORTED_SEQUENCE_PANEL_CONTRACT_VERSION:
        raise SFXIError(
            "Unsupported dnadesign BaseRender sequence-panel contract version "
            f"{actual!r}; reader expects {READER_SUPPORTED_SEQUENCE_PANEL_CONTRACT_VERSION!r}. "
            "Update reader[dnadesign] or sync a compatible dnadesign checkout."
        )
    for attr in ("render_sequence_panel_image", "sequence_panel_config_for_adapter"):
        _require_public_attr(baserender, attr, module_name="dnadesign.baserender")
    for attr in ("Dataset", "default_usr_root"):
        _require_public_attr(usr, attr, module_name="dnadesign.usr")
    return baserender, usr


def _require_public_attr(module, attr: str, *, module_name: str) -> None:
    try:
        getattr(module, attr)
    except AttributeError as exc:
        raise SFXIError(f"{module_name} is missing required public API: {attr}.") from exc
    except ImportError as exc:
        _raise_dnadesign_import_error(exc)


def _raise_dnadesign_import_error(exc: ImportError) -> None:
    raise SFXIError(
        "SFXI triptych sequence requires dnadesign public APIs. Install or sync the optional dependency "
        "with `uv sync --extra dnadesign` or install `reader[dnadesign]`."
    ) from exc


def resolve_usr_root(*, usr, root: object, exp_dir: Path) -> Path:
    if root in (None, ""):
        return Path(usr.default_usr_root())
    root_path = Path(str(root)).expanduser()
    return root_path if root_path.is_absolute() else (exp_dir / root_path).resolve()


def require_usr_sequence_dataset(*, usr, root: object, dataset_name: str, exp_dir: Path) -> Path:
    usr_root = resolve_usr_root(usr=usr, root=root, exp_dir=exp_dir)
    dataset = usr.Dataset.open(usr_root, dataset_name)
    records_path = Path(dataset.records_path)
    if not records_path.exists():
        raise SFXIError(f"sfxi_triptych_sequence could not find USR dataset {dataset_name!r} at {records_path}.")
    return records_path


def load_usr_sequence_rows(*, usr, cfg: Mapping[str, Any], exp_dir: Path) -> pd.DataFrame:
    source = cfg["sequence_source"]
    usr_root = resolve_usr_root(usr=usr, root=source.get("root"), exp_dir=exp_dir)
    dataset = usr.Dataset.open(Path(usr_root), str(source["dataset"]))
    columns = [
        str(source["id_column"]),
        str(source["sequence_column"]),
        str(source["label_column"]),
        str(source["annotations_column"]),
    ]
    include_overlays: bool | list[str] = list(source["required_overlays"]) or True
    batches = list(dataset.scan(columns=columns, include_overlays=include_overlays))
    if not batches:
        raise SFXIError(f"USR dataset has no rows: {Path(usr_root) / str(source['dataset'])}")
    frame = pa.Table.from_batches(batches).to_pandas()
    _require_columns(frame, columns, where=f"USR dataset {source['dataset']}")
    out = frame.rename(
        columns={
            str(source["id_column"]): "usr_sequence_id",
            str(source["sequence_column"]): "usr_sequence",
            str(source["label_column"]): "usr_label",
            str(source["annotations_column"]): "usr_annotations",
        }
    )
    out["usr_dataset"] = str(source["dataset"])
    out["sequence_adapter_kind"] = str(source["adapter_kind"])
    if out["usr_sequence_id"].duplicated().any():
        dupes = sorted(out.loc[out["usr_sequence_id"].duplicated(), "usr_sequence_id"].astype(str).unique())
        raise SFXIError(f"USR dataset {source['dataset']} has duplicate ids: {dupes}")
    return out


def draw_sequence_panel(ax, *, row: pd.Series, baserender, cfg: Mapping[str, Any]):
    panel = cfg["sequence_panel"]
    record_row = {
        "id": str(row["usr_sequence_id"]),
        "sequence": str(row["usr_sequence"]),
    }
    adapter_kind = str(row["sequence_adapter_kind"])
    if adapter_kind == "densegen_tfbs":
        record_row["densegen__used_tfbs_detail"] = row["usr_annotations"]
    elif adapter_kind == "usr_genbank_annotations_v1":
        record_row["seq_annot__features"] = row["usr_annotations"]
        record_row["usr_label__primary"] = str(row.get("usr_label", row["display_label"]))
    else:
        raise SFXIError(f"Unsupported sequence adapter kind: {adapter_kind!r}")

    result = baserender.render_sequence_panel_image(
        record_row,
        adapter_kind=adapter_kind,
        style_profile=str(panel["profile"]),
        style_overrides=dict(panel["style_overrides"]),
        target_width_px=int(panel["target_width_px"]),
        target_height_px=int(panel["target_height_px"]),
        vertical_anchor=str(panel["vertical_anchor"]),
        canvas_top_pad_px=int(panel["canvas_top_pad_px"]),
    )
    ax.imshow(result.image)
    ax.set_axis_off()
    return result.diagnostics


def _require_columns(df: pd.DataFrame, columns: list[str], *, where: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SFXIError(f"{where}: missing required columns {missing}")


__all__ = [
    "DNADESIGN_SEQUENCE_PANEL_CONTRACT_ID",
    "READER_SUPPORTED_SEQUENCE_PANEL_CONTRACT_VERSION",
    "draw_sequence_panel",
    "load_usr_sequence_rows",
    "require_dnadesign_sequence_panel_api",
    "require_usr_sequence_dataset",
    "resolve_usr_root",
]
