"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/notebooks.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from reader.core.errors import ConfigError

EXPERIMENT_EDA_BASE_TEMPLATE = """import marimo

__generated_with = "0.19.1"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    from pathlib import Path
    import json

    import marimo as mo
    try:
        import polars as pl
    except Exception:
        pl = None
    from reader.core.config_model import ReaderSpec

    return (
        Path,
        json,
        mo,
        pl,
        ReaderSpec,
    )

@app.cell(hide_code=True)
def _(Path, ReaderSpec):
    def _find_experiment_root(start: Path) -> Path:
        for base in [start] + list(start.parents):
            if (base / "config.yaml").exists():
                return base
        raise RuntimeError(
            "No config.yaml found. Place this notebook under an experiment directory "
            "or set exp_dir manually."
        )

    def _load_spec(root: Path):
        cfg_path = root / "config.yaml"
        try:
            return ReaderSpec.load(cfg_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read config.yaml: {exc}") from exc

    exp_dir = _find_experiment_root(Path(__file__).resolve())
    spec = _load_spec(exp_dir)
    outputs_dir = Path(spec.paths.outputs).resolve()
    exp_meta = {
        "id": spec.experiment.id,
        "title": spec.experiment.title or "",
    }
    pipeline_step_ids = [step.id for step in spec.pipeline.steps]
    return (
        spec,
        exp_dir,
        exp_meta,
        outputs_dir,
        pipeline_step_ids,
    )

@app.cell(hide_code=True)
def _(json, outputs_dir):
    artifact_info = {}
    artifact_note = ""
    artifact_warning = ""
    _artifacts_dir = outputs_dir / "artifacts"
    _manifest_path = outputs_dir / "manifests" / "manifest.json"

    def _parse_step_dir(step_dir: str):
        base = step_dir.split("__r")[0]
        if "." in base:
            step_id, plugin_key = base.split(".", 1)
        else:
            step_id, plugin_key = base, ""
        return base, step_id, plugin_key

    def _register(display_label, *, step_dir, path, source, entry_label):
        label = display_label or entry_label or step_dir
        if label in artifact_info:
            suffix = entry_label or step_dir
            label = f"{label}:{suffix}"
        base, step_id, plugin_key = _parse_step_dir(step_dir)
        artifact_info[label] = {
            "path": path,
            "step_dir": step_dir,
            "step_id": step_id,
            "plugin_key": plugin_key,
            "source": source,
            "artifact_label": entry_label or "",
            "base_label": base or label,
        }

    if _manifest_path.exists():
        try:
            payload = json.loads(_manifest_path.read_text(encoding="utf-8"))
            artifacts = payload.get("artifacts", {})
            if isinstance(artifacts, dict):
                for entry_label, entry in artifacts.items():
                    step_dir = entry.get("step_dir")
                    filename = entry.get("filename")
                    if not step_dir or not filename:
                        continue
                    if str(filename) != "df.parquet":
                        continue
                    path = _artifacts_dir / step_dir / filename
                    if not path.exists():
                        continue
                    base_label, _, _ = _parse_step_dir(step_dir)
                    _register(
                        base_label or step_dir,
                        step_dir=step_dir,
                        path=path,
                        source="manifest",
                        entry_label=entry_label,
                    )
            if not artifact_info:
                artifact_note = "No df.parquet artifacts listed in outputs/manifests/manifest.json."
        except Exception as exc:
            artifact_note = f"Failed to read manifest.json: {exc}"

    if not artifact_info:
        if not _artifacts_dir.exists():
            if not artifact_note:
                artifact_note = "No outputs/artifacts directory found. Run `reader run` first."
        else:
            df_files = sorted(_artifacts_dir.rglob("df.parquet"))
            for path in df_files:
                step_dir = path.parent.name
                base_label, _, _ = _parse_step_dir(step_dir)
                _register(
                    base_label or step_dir,
                    step_dir=step_dir,
                    path=path,
                    source="scan",
                    entry_label=None,
                )
            if not artifact_info and not artifact_note:
                artifact_note = "No df.parquet artifacts found yet. Run `reader run` first."

    artifact_labels = sorted(artifact_info)
    if any(info.get("source") == "scan" for info in artifact_info.values()):
        artifact_warning = (
            "Warning: dataset list was built by scanning outputs/artifacts because "
            "outputs/manifests/manifest.json was missing, unreadable, or incomplete. "
            "Run `reader run` to regenerate manifests for canonical discovery."
        )
    return artifact_info, artifact_labels, artifact_note, artifact_warning

@app.cell(hide_code=True)
def _(artifact_info, artifact_labels, artifact_note, artifact_warning, mo, pipeline_step_ids):
    if artifact_warning:
        mo.md(artifact_warning)
    if not artifact_labels:
        note = artifact_note or "No datasets found. Run `reader run` first."
        mo.md(note)
        artifact_dropdown = None
    else:
        _default_label = None
        if pipeline_step_ids:
            for _step_id in reversed(pipeline_step_ids):
                _matches = [
                    label for label, info in artifact_info.items() if info.get("step_id") == _step_id
                ]
                if _matches:
                    _default_label = sorted(_matches)[0]
                    break
        if _default_label is None:
            _latest_label = None
            _latest_mtime = None
            for _label in artifact_labels:
                _path = artifact_info[_label]["path"]
                try:
                    _mtime = _path.stat().st_mtime
                except Exception:
                    continue
                if _latest_mtime is None or _mtime > _latest_mtime:
                    _latest_mtime = _mtime
                    _latest_label = _label
            _default_label = _latest_label or artifact_labels[0]
        mo.md(
            f"This run has {len(artifact_labels)} artifact dataset(s). Select one to explore:"
        )
        artifact_dropdown = mo.ui.dropdown(
            options=artifact_labels,
            value=_default_label,
            label="Dataset (artifact df.parquet)",
            full_width=True,
        )
    return artifact_dropdown

@app.cell(hide_code=True)
def _(artifact_dropdown, artifact_info):
    if artifact_dropdown is None:
        selected_label = None
        artifact_path = None
    else:
        selected_label = artifact_dropdown.value
        artifact_path = artifact_info.get(selected_label, {}).get("path")
    return selected_label, artifact_path

@app.cell(hide_code=True)
def _(artifact_path, pl):
    df = None
    df_error = None
    _pl_error = None

    if artifact_path is not None:
        if pl is None:
            df_error = "Polars is required to read parquet. Install the notebooks group."
        else:
            try:
                df = pl.read_parquet(artifact_path)
            except Exception as exc:
                _pl_error = str(exc)
        if df is None and df_error is None:
            _suffix = _pl_error or "unknown error"
            df_error = f"Failed to read parquet with polars ({_suffix})."
    return df, df_error

@app.cell(hide_code=True)
def _(df, pl):
    design_treatment_rows = []
    design_treatment_note = ""
    if df is None:
        design_treatment_note = "No dataset selected yet."
    else:
        _columns = list(df.columns) if hasattr(df, "columns") else []
        _design_col = "design_id" if "design_id" in _columns else None
        _treatment_col = "treatment" if "treatment" in _columns else None

        if _design_col is None or _treatment_col is None:
            _missing = []
            if _design_col is None:
                _missing.append("design_id")
            if _treatment_col is None:
                _missing.append("treatment")
            design_treatment_note = f"Missing column(s): {', '.join(_missing)}."
        else:
            def _unique_values(df, col):
                values = []
                try:
                    if pl is not None and df.__class__.__module__.startswith("polars"):
                        _series = df.select(pl.col(col).drop_nulls().unique()).to_series()
                        values = _series.to_list()
                except Exception:
                    values = []
                values = [str(_v) for _v in values if _v is not None]
                return sorted(values)

            _design_vals = _unique_values(df, _design_col)
            _treatment_vals = _unique_values(df, _treatment_col)
            _max_len = max(len(_design_vals), len(_treatment_vals), 1)
            for _i in range(_max_len):
                design_treatment_rows.append(
                    {
                        "Design IDs": _design_vals[_i] if _i < len(_design_vals) else None,
                        "Treatments": _treatment_vals[_i] if _i < len(_treatment_vals) else None,
                    }
                )
    return design_treatment_rows, design_treatment_note

@app.cell(hide_code=True)
def _(design_treatment_note, design_treatment_rows, exp_dir, exp_meta, mo):
    _exp_id = exp_meta.get("id") or exp_dir.name
    _exp_title = exp_meta.get("title") or _exp_id
    if design_treatment_rows:
        _design_table = mo.ui.table(design_treatment_rows, page_size=len(design_treatment_rows))
    else:
        _design_table = mo.md(design_treatment_note or "No design/treatment summary available.")
    eda_overview_panel = mo.vstack(
        [
            mo.md(f"# {_exp_title}\\n**Experiment id:** `{_exp_id}`"),
            mo.md("**Design IDs + treatments**"),
            _design_table,
        ]
    )
    return eda_overview_panel

@app.cell(hide_code=True)
def _(artifact_dropdown, artifact_note, df_error, mo):
    _elements = [mo.md("## Dataset selection")]
    if artifact_dropdown is None:
        _elements.append(mo.md(artifact_note or "No datasets found."))
    else:
        _elements.append(artifact_dropdown)
        if df_error:
            _elements.append(mo.md(f"**Load error:** `{df_error}`"))
    eda_dataset_panel = mo.vstack(_elements)
    return eda_dataset_panel

@app.cell(hide_code=True)
def _(df, df_error, mo, selected_label):
    if df_error:
        mo.stop(True, mo.md(f"Failed to load `{selected_label}`: {df_error}"))
    if df is None:
        mo.stop(True, mo.md("Select a dataset to explore."))
    data_ready = True
    return data_ready

@app.cell(hide_code=True)
def _(df, data_ready, mo, pl):
    _columns = list(df.columns) if hasattr(df, "columns") else []
    _elements = [mo.md("## Dataset table explorer")]
    df_table = df
    if len(_columns) > 40:
        _display_cols = _columns[:40]
        if pl is not None and df.__class__.__module__.startswith("polars"):
            df_table = df.select(_display_cols)
        _elements.append(mo.md(f"Showing first 40 columns of {len(_columns)}."))
    _elements.append(mo.ui.table(df_table, page_size=10))
    eda_table_panel = mo.vstack(_elements)
    return eda_table_panel

@app.cell(hide_code=True)
def _(eda_dataset_panel, eda_overview_panel, eda_table_panel, mo):
    eda_base_panel = mo.vstack(
        [
            eda_overview_panel,
            eda_dataset_panel,
            eda_table_panel,
        ]
    )
    return eda_base_panel

"""

EXPERIMENT_EDA_BASE_LAYOUT_TEMPLATE = """
@app.cell(hide_code=True)
def _(eda_base_panel):
    eda_base_panel
"""

EXPERIMENT_EDA_TEMPLATE_FOOTER = """
if __name__ == "__main__":
    app.run()
"""

EXPERIMENT_EDA_CYTOMETRY_EXTENSION_TEMPLATE = '''
@app.cell(hide_code=True)
def _():
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    return np, pd, plt

@app.cell(hide_code=True)
def _(data_ready, df, mo, pl):
    _columns = list(df.columns) if hasattr(df, "columns") else []
    _required = ["channel", "value", "sample_id", "event_index"]
    _missing = [c for c in _required if c not in _columns]
    if _missing:
        mo.stop(
            True,
            mo.md(
                "Cytometry EDA requires columns: "
                f"{', '.join(_missing)}. Select a tidy cytometry dataset with "
                "`channel`, `value`, `sample_id`, and `event_index` "
                "(plus optional `treatment`, `design_id`, or `sample_label`)."
            ),
        )

    _channels = []
    if pl is not None and df.__class__.__module__.startswith("polars"):
        try:
            _series = df.select(pl.col("channel").drop_nulls().unique()).to_series()
            _channels = [str(_c) for _c in _series.to_list() if _c is not None]
        except Exception:
            _channels = []
    else:
        try:
            _channels = [str(_c) for _c in df["channel"].dropna().unique().tolist()]
        except Exception:
            _channels = []
    _channels = sorted({c for c in _channels if c})
    if not _channels:
        mo.stop(True, mo.md("No channel values found in the selected dataset."))

    def _unique_values(col):
        values = []
        if col not in _columns:
            return values
        try:
            if pl is not None and df.__class__.__module__.startswith("polars"):
                _series = df.select(pl.col(col).drop_nulls().unique()).to_series()
                values = _series.to_list()
            else:
                values = df[col].dropna().unique().tolist()
        except Exception:
            values = []
        values = [str(_v) for _v in values if _v is not None]
        return sorted({v for v in values if v})

    _treatment_vals = _unique_values("treatment")
    _design_vals = _unique_values("design_id")
    _sample_vals = _unique_values("sample_id")
    _sample_label_vals = _unique_values("sample_label")
    _hue_candidates = [
        c
        for c in ("sample_label", "sample_id", "treatment", "design_id")
        if c in _columns
    ]
    if not _hue_candidates:
        _hue_candidates = ["sample_id"]

    if "treatment" in _hue_candidates:
        _hue_default = "treatment"
    elif "sample_id" in _hue_candidates:
        _hue_default = "sample_id"
    elif "sample_label" in _hue_candidates:
        _hue_default = "sample_label"
    else:
        _hue_default = _hue_candidates[0]

    _threshold_group_default = "sample_id"
    for _candidate in ("treatment", "sample_label", "sample_id"):
        if _candidate in _columns:
            _threshold_group_default = _candidate
            break

    cyto_channel_values = _channels
    cyto_hue_candidates = _hue_candidates
    cyto_hue_default = _hue_default
    cyto_design_values = _design_vals
    cyto_treatment_values = _treatment_vals
    cyto_sample_values = _sample_vals
    cyto_sample_label_values = _sample_label_vals
    cyto_threshold_group_default = _threshold_group_default
    return (
        cyto_channel_values,
        cyto_hue_candidates,
        cyto_hue_default,
        cyto_design_values,
        cyto_treatment_values,
        cyto_sample_values,
        cyto_sample_label_values,
        cyto_threshold_group_default,
    )

@app.cell(hide_code=True)
def _(
    cyto_channel_values,
    cyto_design_values,
    cyto_sample_label_values,
    cyto_sample_values,
    cyto_treatment_values,
    mo,
):
    _parts = []
    if cyto_treatment_values:
        _parts.append(f"{len(cyto_treatment_values)} treatment(s)")
    if cyto_design_values:
        _parts.append(f"{len(cyto_design_values)} design_id(s)")
    if cyto_sample_label_values:
        _parts.append(f"{len(cyto_sample_label_values)} sample label(s)")
    if cyto_sample_values:
        _parts.append(f"{len(cyto_sample_values)} sample_id(s)")
    _channel_list = ", ".join(f"`{c}`" for c in cyto_channel_values)
    _facet_note = (
        "Faceting is optional; choose a small-cardinality column when needed."
        if _parts
        else ""
    )
    cyto_intro_panel = mo.md(
        f"""## Cytometer exploratory data analysis
Detected `{len(cyto_channel_values)}` channel(s): {_channel_list}. {_facet_note}

This notebook starts from a tidy, event-level cytometry table (typically derived upstream from raw FCS) and supports a standard flow:
- **Filter** to a subset of samples (design/treatment/sample_id)
- **Choose channels** for cell selection, singlet isolation, and a fluorescence channel of interest
- **Gate events** (cells → singlets), then inspect **scatter + fluorescence distributions**
- Review **per-sample** and **group** summaries, including `% positive` from a configurable threshold

Operational note: set **Filters → Channels → Gates** first, then inspect plots and stats. Plot-only controls (downsampling, low-clip, colors/facets, axis scales) affect visualization but do not change gate counts or summary statistics.

Histogram note: binning and transforms matter. Most histogram functions bin in **linear** space, even if you later display the x-axis on a **log** scale. With a wide range, early bins can span multiple decades and appear as one big, flat rectangle. This notebook uses **log-spaced bins** for `log` histograms and applies **asinh/symlog** transforms before binning when selected. If many values are ≤ 0 (common in compensated cytometry), prefer **symlog/asinh/logicle** rather than a strict log axis; compensation/logicle transforms are not performed here yet.
"""
    )
    return cyto_intro_panel

@app.cell(hide_code=True)
def _(
    cyto_channel_values,
    cyto_design_values,
    cyto_hue_candidates,
    cyto_hue_default,
    cyto_sample_label_values,
    cyto_sample_values,
    cyto_treatment_values,
    cyto_threshold_group_default,
    mo,
):
    def _pick_channel(needles):
        for needle in needles:
            for name in cyto_channel_values:
                if needle in name.lower():
                    return name
        return cyto_channel_values[0]

    def _pick_distinct(primary, fallback):
        if primary != fallback:
            return fallback
        for name in cyto_channel_values:
            if name != primary:
                return name
        return fallback

    _cells_x_default = _pick_channel(["fsc-a", "fsc"])
    _cells_y_default = _pick_channel(["ssc-a", "ssc"])
    _cells_y_default = _pick_distinct(_cells_x_default, _cells_y_default)

    _scatter_x_default = _cells_x_default
    _scatter_y_default = _cells_y_default

    _singlet_x_default = _pick_channel(["fsc-a", "fsc"])
    _singlet_y_default = _pick_channel(["fsc-h", "fsc-w", "fsc"])
    _singlet_y_default = _pick_distinct(_singlet_x_default, _singlet_y_default)

    _fluor_default = _pick_channel(["mcherry-a", "mcherry", "rfp", "dsred", "texas", "tx-red"])
    _fluor_options = [c for c in cyto_channel_values if c.lower() not in {"time"}]
    if not _fluor_options:
        _fluor_options = cyto_channel_values

    _facet_options = ["None"]
    for _col, _values in (
        ("design_id", cyto_design_values),
        ("treatment", cyto_treatment_values),
        ("sample_label", cyto_sample_label_values),
        ("sample_id", cyto_sample_values),
    ):
        if _values:
            _facet_options.append(_col)

    _facet_default = "design_id" if "design_id" in _facet_options else "None"

    cyto_cells_x_dropdown = mo.ui.dropdown(
        options=cyto_channel_values,
        value=_cells_x_default,
        label="Cells gate X channel",
        full_width=True,
    )
    cyto_cells_y_dropdown = mo.ui.dropdown(
        options=cyto_channel_values,
        value=_cells_y_default,
        label="Cells gate Y channel",
        full_width=True,
    )
    cyto_singlet_x_dropdown = mo.ui.dropdown(
        options=cyto_channel_values,
        value=_singlet_x_default,
        label="Singlets gate X channel",
        full_width=True,
    )
    cyto_singlet_y_dropdown = mo.ui.dropdown(
        options=cyto_channel_values,
        value=_singlet_y_default,
        label="Singlets gate Y channel",
        full_width=True,
    )
    cyto_scatter_x_dropdown = mo.ui.dropdown(
        options=cyto_channel_values,
        value=_scatter_x_default,
        label="Scatter X channel",
        full_width=True,
    )
    cyto_scatter_y_dropdown = mo.ui.dropdown(
        options=cyto_channel_values,
        value=_scatter_y_default,
        label="Scatter Y channel",
        full_width=True,
    )
    cyto_hue_dropdown = mo.ui.dropdown(
        options=cyto_hue_candidates,
        value=cyto_hue_default,
        label="Color by",
        full_width=True,
    )
    cyto_scatter_scale_x = mo.ui.dropdown(
        options=["linear", "log", "symlog"],
        value="linear",
        label="Scatter X scale",
        full_width=True,
    )
    cyto_scatter_scale_y = mo.ui.dropdown(
        options=["linear", "log", "symlog"],
        value="linear",
        label="Scatter Y scale",
        full_width=True,
    )
    cyto_fluor_dropdown = mo.ui.dropdown(
        options=_fluor_options,
        value=_fluor_default if _fluor_default in _fluor_options else _fluor_options[0],
        label="Fluorescence channel",
        full_width=True,
    )
    cyto_hist_scale = mo.ui.dropdown(
        options=["log", "linear", "symlog", "asinh"],
        value="log",
        label="Fluorescence scale",
        full_width=True,
    )
    cyto_facet_dropdown = mo.ui.dropdown(
        options=_facet_options,
        value=_facet_default,
        label="Facet by",
        full_width=True,
    )
    cyto_design_filter = mo.ui.dropdown(
        options=["All"] + cyto_design_values,
        value="All",
        label="Filter: design_id",
        full_width=True,
    )
    cyto_treatment_filter = mo.ui.dropdown(
        options=["All"] + cyto_treatment_values,
        value="All",
        label="Filter: treatment",
        full_width=True,
    )
    cyto_sample_filter = mo.ui.dropdown(
        options=["All"] + cyto_sample_values,
        value="All",
        label="Filter: sample_id",
        full_width=True,
    )
    cyto_downsample_target = mo.ui.slider(
        1_000,
        50_000,
        value=25_000,
        step=1_000,
        label="Max events to plot",
        full_width=True,
    )
    cyto_low_clip_dropdown = mo.ui.dropdown(
        options=["off", "0.001", "0.01", "0.05"],
        value="off",
        label="Scatter low-clip quantile",
        full_width=True,
    )
    cyto_cells_gate_enabled = mo.ui.checkbox(
        label="Enable cells gate", value=True
    )
    cyto_singlet_gate_enabled = mo.ui.checkbox(
        label="Enable singlets gate", value=True
    )
    cyto_threshold_mode = mo.ui.dropdown(
        options=["manual", "from_control_quantile"],
        value="manual",
        label="Positive threshold mode",
        full_width=True,
    )
    cyto_threshold_value = mo.ui.number(
        value=0.0,
        label="Manual threshold (a.u.)",
        full_width=True,
    )
    cyto_threshold_quantile = mo.ui.slider(
        0.5,
        0.999,
        value=0.99,
        step=0.001,
        label="Control quantile",
        full_width=True,
    )
    _threshold_options = [
        opt
        for opt in ("treatment", "sample_label", "sample_id")
        if (
            (opt == "treatment" and cyto_treatment_values)
            or (opt == "sample_label" and cyto_sample_label_values)
            or opt == "sample_id"
        )
    ]
    _threshold_default = (
        cyto_threshold_group_default
        if cyto_threshold_group_default in _threshold_options
        else _threshold_options[0]
    )
    cyto_threshold_group_dropdown = mo.ui.dropdown(
        options=_threshold_options,
        value=_threshold_default,
        label="Control group column",
        full_width=True,
    )

    _filters_row = mo.hstack(
        [cyto_design_filter, cyto_treatment_filter, cyto_sample_filter],
        gap=1,
        align="start",
        justify="start",
    )
    _channels_row_1 = mo.hstack(
        [
            cyto_cells_x_dropdown,
            cyto_cells_y_dropdown,
            cyto_singlet_x_dropdown,
            cyto_singlet_y_dropdown,
        ],
        gap=1,
        align="start",
        justify="start",
    )
    _channels_row_2 = mo.hstack(
        [
            cyto_scatter_x_dropdown,
            cyto_scatter_y_dropdown,
            cyto_fluor_dropdown,
        ],
        gap=1,
        align="start",
        justify="start",
    )
    _display_row_1 = mo.hstack(
        [
            cyto_hue_dropdown,
            cyto_facet_dropdown,
            cyto_scatter_scale_x,
            cyto_scatter_scale_y,
        ],
        gap=1,
        align="start",
        justify="start",
    )
    _display_row_2 = mo.hstack(
        [
            cyto_hist_scale,
            cyto_downsample_target,
            cyto_low_clip_dropdown,
        ],
        gap=1,
        align="start",
        justify="start",
    )

    cyto_controls_panel = mo.vstack(
        [
            mo.md("### Filters"),
            mo.md("Subset the dataset before pivoting/gating. Filters affect plots, stats, QC, and exports."),
            _filters_row,
            mo.md("### Channels"),
            mo.md("Choose channels used for gates and which fluorescence channel to summarize."),
            _channels_row_1,
            _channels_row_2,
            mo.md("### Display"),
            mo.md("Plot-only controls: coloring, faceting, axis scales, clipping, and downsampling."),
            _display_row_1,
            _display_row_2,
        ]
    )
    return (
        cyto_scatter_x_dropdown,
        cyto_scatter_y_dropdown,
        cyto_cells_x_dropdown,
        cyto_cells_y_dropdown,
        cyto_singlet_x_dropdown,
        cyto_singlet_y_dropdown,
        cyto_hue_dropdown,
        cyto_scatter_scale_x,
        cyto_scatter_scale_y,
        cyto_fluor_dropdown,
        cyto_hist_scale,
        cyto_facet_dropdown,
        cyto_design_filter,
        cyto_treatment_filter,
        cyto_sample_filter,
        cyto_downsample_target,
        cyto_low_clip_dropdown,
        cyto_threshold_mode,
        cyto_threshold_value,
        cyto_threshold_quantile,
        cyto_threshold_group_dropdown,
        cyto_controls_panel,
        cyto_cells_gate_enabled,
        cyto_singlet_gate_enabled,
    )

@app.cell(hide_code=True)
def _(
    cyto_sample_label_values,
    cyto_sample_values,
    cyto_threshold_group_dropdown,
    cyto_treatment_values,
    mo,
):
    _group = str(cyto_threshold_group_dropdown.value)
    if _group == "treatment":
        _options = cyto_treatment_values
    elif _group == "sample_label":
        _options = cyto_sample_label_values
    else:
        _options = cyto_sample_values
    if not _options:
        _options = ["(none)"]
    cyto_threshold_control_dropdown = mo.ui.dropdown(
        options=_options,
        value=_options[0],
        label="Control group value",
        full_width=True,
    )
    return cyto_threshold_control_dropdown

@app.cell(hide_code=True)
def _(
    cyto_cells_x_dropdown,
    cyto_cells_y_dropdown,
    cyto_design_filter,
    cyto_fluor_dropdown,
    cyto_sample_filter,
    cyto_scatter_x_dropdown,
    cyto_scatter_y_dropdown,
    cyto_singlet_x_dropdown,
    cyto_singlet_y_dropdown,
    cyto_treatment_filter,
    df,
    mo,
    np,
    pd,
    pl,
):
    if pd is None or np is None:
        mo.stop(True, mo.md("Pandas and NumPy are required for cytometry plots."))

    _scatter_x = str(cyto_scatter_x_dropdown.value)
    _scatter_y = str(cyto_scatter_y_dropdown.value)
    _cells_x = str(cyto_cells_x_dropdown.value)
    _cells_y = str(cyto_cells_y_dropdown.value)
    _singlet_x = str(cyto_singlet_x_dropdown.value)
    _singlet_y = str(cyto_singlet_y_dropdown.value)
    _fluor = str(cyto_fluor_dropdown.value)

    if _scatter_x == _scatter_y:
        mo.stop(True, mo.md("Scatter X and Y channels must be different."))
    if _cells_x == _cells_y:
        mo.stop(True, mo.md("Cells gate X and Y channels must be different."))
    if _singlet_x == _singlet_y:
        mo.stop(True, mo.md("Singlets gate X and Y channels must be different."))

    _channels = sorted({_scatter_x, _scatter_y, _cells_x, _cells_y, _singlet_x, _singlet_y, _fluor})

    _meta_cols = ["treatment", "design_id", "sample_label"]
    _meta_cols = [c for c in _meta_cols if c in getattr(df, "columns", [])]
    _index_cols = ["sample_id", "event_index"] + _meta_cols

    _is_polars = pl is not None and df.__class__.__module__.startswith("polars")
    if _is_polars:
        _df = df.filter(pl.col("channel").is_in(_channels))
        if cyto_design_filter.value != "All" and "design_id" in df.columns:
            _df = _df.filter(pl.col("design_id") == cyto_design_filter.value)
        if cyto_treatment_filter.value != "All" and "treatment" in df.columns:
            _df = _df.filter(pl.col("treatment") == cyto_treatment_filter.value)
        if cyto_sample_filter.value != "All" and "sample_id" in df.columns:
            _df = _df.filter(pl.col("sample_id") == cyto_sample_filter.value)
        _df = _df.with_columns(pl.col("value").cast(pl.Float64))
        _wide = _df.pivot(values="value", index=_index_cols, on="channel", aggregate_function="first")
        cyto_event_wide = _wide.to_pandas(use_pyarrow_extension_array=False)
    else:
        _df_pd = df.copy()
        if cyto_design_filter.value != "All" and "design_id" in _df_pd.columns:
            _df_pd = _df_pd[_df_pd["design_id"] == cyto_design_filter.value]
        if cyto_treatment_filter.value != "All" and "treatment" in _df_pd.columns:
            _df_pd = _df_pd[_df_pd["treatment"] == cyto_treatment_filter.value]
        if cyto_sample_filter.value != "All" and "sample_id" in _df_pd.columns:
            _df_pd = _df_pd[_df_pd["sample_id"] == cyto_sample_filter.value]
        _df_pd = _df_pd[_df_pd["channel"].astype(str).isin(_channels)].copy()
        _df_pd["value"] = pd.to_numeric(_df_pd["value"], errors="coerce")
        _wide = _df_pd.pivot_table(index=_index_cols, columns="channel", values="value", aggfunc="first")
        _wide = _wide.reset_index()
        if isinstance(_wide.columns, pd.MultiIndex):
            _flat_cols = []
            for _col in _wide.columns:
                if isinstance(_col, tuple):
                    _parts = [c for c in _col if c not in (None, "")]
                    _flat_cols.append(str(_parts[-1]) if _parts else "")
                else:
                    _flat_cols.append(str(_col))
            _wide.columns = _flat_cols
        cyto_event_wide = _wide

    if cyto_event_wide.empty:
        mo.stop(True, mo.md("No events remain after filtering."))

    _missing = [c for c in _channels if c not in cyto_event_wide.columns]
    if _missing:
        mo.stop(True, mo.md(f"Missing channels after pivot: {', '.join(_missing)}."))

    cyto_meta_cols = _meta_cols
    return cyto_event_wide, cyto_meta_cols

@app.cell(hide_code=True)
def _(
    cyto_cells_gate_enabled,
    cyto_cells_x_dropdown,
    cyto_cells_y_dropdown,
    cyto_event_wide,
    cyto_singlet_gate_enabled,
    cyto_singlet_x_dropdown,
    cyto_singlet_y_dropdown,
    mo,
    np,
    pd,
):
    def _safe_range(values, low_q=0.01, high_q=0.99):
        _vals = np.asarray(pd.to_numeric(values, errors="coerce"))
        _vals = _vals[np.isfinite(_vals)]
        if _vals.size == 0:
            mo.stop(True, mo.md("No finite values available for gate defaults."))
        _min = float(np.nanmin(_vals))
        _max = float(np.nanmax(_vals))
        if not np.isfinite(_min) or not np.isfinite(_max):
            mo.stop(True, mo.md("Gate range could not be determined."))
        if _max <= _min:
            _max = _min + 1e-6
        _q = np.nanquantile(_vals, [low_q, high_q])
        _q_low = float(_q[0])
        _q_high = float(_q[1])
        _q_low = max(_q_low, _min)
        _q_high = min(_q_high, _max)
        if _q_high <= _q_low:
            _q_low, _q_high = _min, _max
        return _min, _max, (_q_low, _q_high)

    _cells_x = str(cyto_cells_x_dropdown.value)
    _cells_y = str(cyto_cells_y_dropdown.value)
    _singlet_x = str(cyto_singlet_x_dropdown.value)
    _singlet_y = str(cyto_singlet_y_dropdown.value)

    _cells_x_min, _cells_x_max, _cells_x_default = _safe_range(cyto_event_wide[_cells_x])
    _cells_y_min, _cells_y_max, _cells_y_default = _safe_range(cyto_event_wide[_cells_y])

    _singlet_x_vals = pd.to_numeric(cyto_event_wide[_singlet_x], errors="coerce").to_numpy()
    _singlet_y_vals = pd.to_numeric(cyto_event_wide[_singlet_y], errors="coerce").to_numpy()
    _ratio = np.divide(
        _singlet_y_vals,
        _singlet_x_vals,
        out=np.full_like(_singlet_y_vals, np.nan),
        where=_singlet_x_vals != 0,
    )
    _ratio_min, _ratio_max, _ratio_default = _safe_range(_ratio)

    cyto_cells_x_range = mo.ui.range_slider(
        _cells_x_min,
        _cells_x_max,
        value=_cells_x_default,
        label="Cells gate X range",
        full_width=True,
    )
    cyto_cells_y_range = mo.ui.range_slider(
        _cells_y_min,
        _cells_y_max,
        value=_cells_y_default,
        label="Cells gate Y range",
        full_width=True,
    )
    cyto_singlet_ratio_range = mo.ui.range_slider(
        _ratio_min,
        _ratio_max,
        value=_ratio_default,
        label="Singlets ratio (Y / X)",
        full_width=True,
    )

    _cells_gate_row = mo.hstack(
        [cyto_cells_gate_enabled, cyto_cells_x_range, cyto_cells_y_range],
        gap=1,
        align="start",
        justify="start",
    )
    _singlet_gate_row = mo.hstack(
        [cyto_singlet_gate_enabled, cyto_singlet_ratio_range],
        gap=1,
        align="start",
        justify="start",
    )

    cyto_gate_panel = mo.vstack(
        [
            mo.md("### Gates"),
            mo.md(
                "Define gates in order: **cells** (FSC/SSC rectangle) → **singlets** (FSC-H/FSC-A ratio band). These gates affect downstream plots and statistics."
            ),
            _cells_gate_row,
            _singlet_gate_row,
        ]
    )
    return (
        cyto_cells_x_range,
        cyto_cells_y_range,
        cyto_singlet_ratio_range,
        cyto_gate_panel,
    )

@app.cell(hide_code=True)
def _(
    cyto_cells_gate_enabled,
    cyto_cells_x_dropdown,
    cyto_cells_x_range,
    cyto_cells_y_dropdown,
    cyto_cells_y_range,
    cyto_design_filter,
    cyto_event_wide,
    cyto_facet_dropdown,
    cyto_fluor_dropdown,
    cyto_hist_scale,
    cyto_hue_dropdown,
    cyto_low_clip_dropdown,
    cyto_meta_cols,
    cyto_sample_filter,
    cyto_scatter_scale_x,
    cyto_scatter_scale_y,
    cyto_scatter_x_dropdown,
    cyto_scatter_y_dropdown,
    cyto_singlet_gate_enabled,
    cyto_singlet_ratio_range,
    cyto_singlet_x_dropdown,
    cyto_singlet_y_dropdown,
    cyto_treatment_filter,
    cyto_threshold_control_dropdown,
    cyto_threshold_group_dropdown,
    cyto_threshold_mode,
    cyto_threshold_quantile,
    cyto_threshold_value,
    cyto_downsample_target,
    mo,
    np,
    pd,
):
    _df = cyto_event_wide.copy()

    _cells_x = str(cyto_cells_x_dropdown.value)
    _cells_y = str(cyto_cells_y_dropdown.value)
    _singlet_x = str(cyto_singlet_x_dropdown.value)
    _singlet_y = str(cyto_singlet_y_dropdown.value)
    _fluor = str(cyto_fluor_dropdown.value)

    _cells_x_vals = pd.to_numeric(_df[_cells_x], errors="coerce").to_numpy()
    _cells_y_vals = pd.to_numeric(_df[_cells_y], errors="coerce").to_numpy()
    _singlet_x_vals = pd.to_numeric(_df[_singlet_x], errors="coerce").to_numpy()
    _singlet_y_vals = pd.to_numeric(_df[_singlet_y], errors="coerce").to_numpy()

    _cells_mask = np.isfinite(_cells_x_vals) & np.isfinite(_cells_y_vals)
    if cyto_cells_gate_enabled.value:
        _x_lo, _x_hi = cyto_cells_x_range.value
        _y_lo, _y_hi = cyto_cells_y_range.value
        _cells_mask &= (_cells_x_vals >= _x_lo) & (_cells_x_vals <= _x_hi)
        _cells_mask &= (_cells_y_vals >= _y_lo) & (_cells_y_vals <= _y_hi)

    _ratio = np.divide(
        _singlet_y_vals,
        _singlet_x_vals,
        out=np.full_like(_singlet_y_vals, np.nan),
        where=_singlet_x_vals != 0,
    )
    _singlet_mask = np.isfinite(_ratio)
    if cyto_singlet_gate_enabled.value:
        _r_lo, _r_hi = cyto_singlet_ratio_range.value
        _singlet_mask &= (_ratio >= _r_lo) & (_ratio <= _r_hi)

    _gate_mask = _cells_mask & _singlet_mask
    if not _gate_mask.any():
        mo.stop(True, mo.md("No events remain after gating. Adjust ranges."))

    cyto_gated_events = _df.loc[_gate_mask].copy()

    _count_cols = ["sample_id"] + [c for c in cyto_meta_cols if c in _df.columns]
    _df_counts = _df[_count_cols].copy()
    _df_counts["_cells_mask"] = _cells_mask
    _df_counts["_gate_mask"] = _gate_mask

    _counts = (
        _df_counts.groupby("sample_id", dropna=False)
        .agg(
            n_total_events=("_cells_mask", "size"),
            n_cells_gate=("_cells_mask", "sum"),
            n_singlets=("_gate_mask", "sum"),
        )
        .reset_index()
    )

    _meta = (
        _df_counts.groupby("sample_id", dropna=False)[cyto_meta_cols]
        .first()
        .reset_index()
        if cyto_meta_cols
        else _counts[["sample_id"]]
    )

    cyto_gate_counts_sample = _meta.merge(_counts, on="sample_id", how="right")
    cyto_gate_counts_sample["pct_cells"] = np.where(
        cyto_gate_counts_sample["n_total_events"] > 0,
        100.0 * cyto_gate_counts_sample["n_cells_gate"] / cyto_gate_counts_sample["n_total_events"],
        np.nan,
    )
    cyto_gate_counts_sample["pct_singlets_of_cells"] = np.where(
        cyto_gate_counts_sample["n_cells_gate"] > 0,
        100.0 * cyto_gate_counts_sample["n_singlets"] / cyto_gate_counts_sample["n_cells_gate"],
        np.nan,
    )
    cyto_gate_counts_sample["pct_final"] = np.where(
        cyto_gate_counts_sample["n_total_events"] > 0,
        100.0 * cyto_gate_counts_sample["n_singlets"] / cyto_gate_counts_sample["n_total_events"],
        np.nan,
    )

    _threshold_mode = str(cyto_threshold_mode.value)
    _threshold_value = float(cyto_threshold_value.value)
    if _threshold_mode == "from_control_quantile":
        _group_col = str(cyto_threshold_group_dropdown.value)
        if _group_col not in cyto_gated_events.columns:
            mo.stop(True, mo.md(f"Control column `{_group_col}` is missing."))
        _control_value = str(cyto_threshold_control_dropdown.value)
        _control_mask = cyto_gated_events[_group_col].astype(str) == _control_value
        _control_vals = pd.to_numeric(
            cyto_gated_events.loc[_control_mask, _fluor], errors="coerce"
        ).to_numpy()
        _control_vals = _control_vals[np.isfinite(_control_vals)]
        if _control_vals.size == 0:
            mo.stop(True, mo.md("No control events available for thresholding."))
        _q = float(cyto_threshold_quantile.value)
        _threshold_value = float(np.nanquantile(_control_vals, _q))

    if not np.isfinite(_threshold_value):
        mo.stop(True, mo.md("Threshold value is not finite."))

    def _summarize(values):
        _vals = pd.to_numeric(values, errors="coerce").to_numpy()
        _vals = _vals[np.isfinite(_vals)]
        if _vals.size == 0:
            return pd.Series(
                {
                    "fluor_median": np.nan,
                    "fluor_mean": np.nan,
                    "fluor_geomean": np.nan,
                    "fluor_p90": np.nan,
                    "fluor_p99": np.nan,
                    "pct_positive": np.nan,
                }
            )
        _pos = _vals[_vals > 0]
        _geom = float(np.exp(np.mean(np.log(_pos)))) if _pos.size else np.nan
        return pd.Series(
            {
                "fluor_median": float(np.nanmedian(_vals)),
                "fluor_mean": float(np.nanmean(_vals)),
                "fluor_geomean": _geom,
                "fluor_p90": float(np.nanpercentile(_vals, 90)),
                "fluor_p99": float(np.nanpercentile(_vals, 99)),
                "pct_positive": float(100.0 * np.mean(_vals > _threshold_value)),
            }
        )

    _applied = cyto_gated_events.groupby("sample_id", dropna=False)[_fluor].apply(_summarize)
    if isinstance(_applied, pd.Series):
        _sample_stats = _applied.unstack().reset_index()
    else:
        _sample_stats = _applied.reset_index()
    _required_stats = [
        "fluor_median",
        "fluor_mean",
        "fluor_geomean",
        "fluor_p90",
        "fluor_p99",
        "pct_positive",
    ]
    for _col in _required_stats:
        if _col not in _sample_stats.columns:
            _sample_stats[_col] = np.nan
    _sample_stats = _sample_stats[["sample_id"] + _required_stats]
    cyto_stats_sample = cyto_gate_counts_sample.merge(_sample_stats, on="sample_id", how="left")

    cyto_stats_group = None
    _group_col = None
    for _candidate in ("treatment", "design_id"):
        if _candidate in cyto_stats_sample.columns:
            _group_col = _candidate
            break
    _needed_for_group = {"fluor_median", "fluor_geomean", "pct_positive"}
    if _group_col and _needed_for_group.issubset(set(cyto_stats_sample.columns)):
        cyto_stats_group = (
            cyto_stats_sample.groupby(_group_col, dropna=False)
            .agg(
                n_samples=("sample_id", "nunique"),
                fluor_median_mean=("fluor_median", "mean"),
                fluor_median_std=("fluor_median", "std"),
                fluor_geomean_mean=("fluor_geomean", "mean"),
                pct_positive_mean=("pct_positive", "mean"),
            )
            .reset_index()
        )

    def _nonpositive_pct(values):
        _vals = pd.to_numeric(values, errors="coerce").to_numpy()
        _vals = _vals[np.isfinite(_vals)]
        if _vals.size == 0:
            return np.nan
        return float(100.0 * np.mean(_vals <= 0))

    cyto_qc_table = (
        _df.groupby("sample_id", dropna=False)
        .agg(
            pct_nonpositive=(_fluor, _nonpositive_pct),
        )
        .reset_index()
    )

    cyto_gate_config = {
        "filters": {
            "design_id": str(cyto_design_filter.value),
            "treatment": str(cyto_treatment_filter.value),
            "sample_id": str(cyto_sample_filter.value),
        },
        "scatter": {
            "x_channel": str(cyto_scatter_x_dropdown.value),
            "y_channel": str(cyto_scatter_y_dropdown.value),
            "x_scale": str(cyto_scatter_scale_x.value),
            "y_scale": str(cyto_scatter_scale_y.value),
        },
        "hue": str(cyto_hue_dropdown.value),
        "facet_by": str(cyto_facet_dropdown.value),
        "histogram": {
            "scale": str(cyto_hist_scale.value),
            "low_clip_quantile": str(cyto_low_clip_dropdown.value),
        },
        "downsample_target": int(cyto_downsample_target.value),
        "cells_gate": {
            "enabled": bool(cyto_cells_gate_enabled.value),
            "x_channel": _cells_x,
            "y_channel": _cells_y,
            "x_range": list(cyto_cells_x_range.value),
            "y_range": list(cyto_cells_y_range.value),
        },
        "singlets_gate": {
            "enabled": bool(cyto_singlet_gate_enabled.value),
            "x_channel": _singlet_x,
            "y_channel": _singlet_y,
            "ratio_range": list(cyto_singlet_ratio_range.value),
        },
        "fluor": {
            "channel": _fluor,
        },
        "threshold": {
            "mode": _threshold_mode,
            "value": _threshold_value,
            "group_column": str(cyto_threshold_group_dropdown.value),
            "control_value": str(cyto_threshold_control_dropdown.value),
            "quantile": float(cyto_threshold_quantile.value),
        },
    }

    cyto_threshold_value_final = _threshold_value
    return (
        cyto_gated_events,
        cyto_gate_counts_sample,
        cyto_stats_sample,
        cyto_stats_group,
        cyto_threshold_value_final,
        cyto_gate_config,
        cyto_qc_table,
    )

@app.cell
def _(
    cyto_cells_x_dropdown,
    cyto_cells_x_range,
    cyto_cells_y_dropdown,
    cyto_cells_y_range,
    cyto_event_wide,
    cyto_facet_dropdown,
    cyto_fluor_dropdown,
    cyto_gated_events,
    cyto_hue_dropdown,
    cyto_hist_scale,
    cyto_low_clip_dropdown,
    cyto_scatter_scale_x,
    cyto_scatter_scale_y,
    cyto_scatter_x_dropdown,
    cyto_scatter_y_dropdown,
    cyto_singlet_ratio_range,
    cyto_singlet_x_dropdown,
    cyto_singlet_y_dropdown,
    cyto_threshold_value_final,
    cyto_downsample_target,
    mo,
    np,
    pd,
    plt,
):
    if pd is None or np is None:
        mo.stop(True, mo.md("Pandas and NumPy are required for cytometry plots."))
    if plt is None:
        mo.stop(True, mo.md("Matplotlib is required for cytometry plots."))

    _hue_col = str(cyto_hue_dropdown.value)
    _facet_col = str(cyto_facet_dropdown.value)
    if _facet_col == "None":
        _facet_col = None

    def _downsample(df, max_events, group_cols):
        if df is None or df.empty:
            return df
        if len(df) <= max_events:
            return df
        if group_cols:
            _grouped = df.groupby(group_cols, dropna=False, group_keys=False)
            _per_group = max(1, int(max_events / max(_grouped.ngroups, 1)))
            return (
                _grouped.apply(
                    lambda g: g.sample(n=min(len(g), _per_group), random_state=0)
                )
                .reset_index(drop=True)
            )
        return df.sample(n=max_events, random_state=0).reset_index(drop=True)

    def _unique_cols(cols):
        seen = set()
        out = []
        for col in cols:
            if col in seen:
                continue
            seen.add(col)
            out.append(col)
        return out

    def _col_series(df, col):
        _data = df[col]
        if isinstance(_data, pd.DataFrame):
            return _data.iloc[:, 0]
        return _data

    _palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
        "#8c564b",
        "#e377c2",
    ]

    _cells_x = str(cyto_cells_x_dropdown.value)
    _cells_y = str(cyto_cells_y_dropdown.value)
    _singlet_x = str(cyto_singlet_x_dropdown.value)
    _singlet_y = str(cyto_singlet_y_dropdown.value)

    _gating_cols = _unique_cols([_cells_x, _cells_y, _singlet_x, _singlet_y, "sample_id"])
    if _hue_col in cyto_event_wide.columns:
        _gating_cols.append(_hue_col)
    if _facet_col and _facet_col in cyto_event_wide.columns:
        _gating_cols.append(_facet_col)
    _gating_df = cyto_event_wide[_gating_cols].copy()
    if _gating_df.columns.duplicated().any():
        _gating_df = _gating_df.loc[:, ~_gating_df.columns.duplicated()]

    _max_events = int(cyto_downsample_target.value)
    _gating_df = _downsample(_gating_df, _max_events, [c for c in [_facet_col, _hue_col] if c])

    _hue_values = sorted({str(v) for v in _gating_df[_hue_col].dropna().unique()}) if _hue_col in _gating_df.columns else ["all"]
    if not _hue_values:
        _hue_values = ["all"]
        _gating_df["_hue"] = "all"
        _hue_col = "_hue"
    _color_map = {hue: _palette[idx % len(_palette)] for idx, hue in enumerate(_hue_values)}

    cyto_gating_fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    _ax_cells, _ax_singlets = axes
    _ax_cells.set_facecolor("white")
    _ax_singlets.set_facecolor("white")

    for _hue in _hue_values:
        _group = _gating_df[_gating_df[_hue_col].astype(str) == _hue]
        if _group.empty:
            continue
        _x_vals = pd.to_numeric(_col_series(_group, _cells_x), errors="coerce")
        _y_vals = pd.to_numeric(_col_series(_group, _cells_y), errors="coerce")
        _mask = np.isfinite(_x_vals) & np.isfinite(_y_vals)
        if not _mask.any():
            continue
        _ax_cells.scatter(
            _x_vals[_mask],
            _y_vals[_mask],
            s=10,
            alpha=0.35,
            color=_color_map.get(_hue, "#000000"),
            label=_hue,
        )
    _x_lo, _x_hi = cyto_cells_x_range.value
    _y_lo, _y_hi = cyto_cells_y_range.value
    _ax_cells.add_patch(
        plt.Rectangle(
            (_x_lo, _y_lo),
            _x_hi - _x_lo,
            _y_hi - _y_lo,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )
    _ax_cells.set_xlabel(_cells_x, color="black")
    _ax_cells.set_ylabel(_cells_y, color="black")
    _ax_cells.set_title("Cells gate", color="black")
    _ax_cells.tick_params(colors="black")
    for _spine in ("left", "bottom", "right", "top"):
        _ax_cells.spines[_spine].set_visible(True)
        _ax_cells.spines[_spine].set_color("black")
    _cells_leg = _ax_cells.legend(loc="upper right", frameon=False, fontsize=8)
    if _cells_leg is not None:
        for _text in _cells_leg.get_texts():
            _text.set_color("black")

    for _hue in _hue_values:
        _group = _gating_df[_gating_df[_hue_col].astype(str) == _hue]
        if _group.empty:
            continue
        _x_vals = pd.to_numeric(_col_series(_group, _singlet_x), errors="coerce")
        _y_vals = pd.to_numeric(_col_series(_group, _singlet_y), errors="coerce")
        _mask = np.isfinite(_x_vals) & np.isfinite(_y_vals)
        if not _mask.any():
            continue
        _ax_singlets.scatter(
            _x_vals[_mask],
            _y_vals[_mask],
            s=10,
            alpha=0.35,
            color=_color_map.get(_hue, "#000000"),
            label=_hue,
        )
    _r_lo, _r_hi = cyto_singlet_ratio_range.value
    _x_vals = pd.to_numeric(_col_series(_gating_df, _singlet_x), errors="coerce")
    _x_vals = _x_vals[np.isfinite(_x_vals)].to_numpy()
    if _x_vals.size:
        _x_min = float(np.nanmin(_x_vals))
        _x_max = float(np.nanmax(_x_vals))
        _line_x = np.linspace(_x_min, _x_max, 200)
        _ax_singlets.plot(_line_x, _r_lo * _line_x, color="black", linewidth=1.0)
        _ax_singlets.plot(_line_x, _r_hi * _line_x, color="black", linewidth=1.0)
    _ax_singlets.set_xlabel(_singlet_x, color="black")
    _ax_singlets.set_ylabel(_singlet_y, color="black")
    _ax_singlets.set_title("Singlets gate", color="black")
    _ax_singlets.tick_params(colors="black")
    for _spine in ("left", "bottom", "right", "top"):
        _ax_singlets.spines[_spine].set_visible(True)
        _ax_singlets.spines[_spine].set_color("black")
    _singlet_leg = _ax_singlets.legend(loc="upper right", frameon=False, fontsize=8)
    if _singlet_leg is not None:
        for _text in _singlet_leg.get_texts():
            _text.set_color("black")
    cyto_gating_fig.patch.set_facecolor("white")

    _scatter_x = str(cyto_scatter_x_dropdown.value)
    _scatter_y = str(cyto_scatter_y_dropdown.value)
    _fluor = str(cyto_fluor_dropdown.value)

    _plot_cols = _unique_cols([_scatter_x, _scatter_y, _fluor, "sample_id"])
    if _hue_col in cyto_gated_events.columns:
        _plot_cols.append(_hue_col)
    if _facet_col and _facet_col in cyto_gated_events.columns:
        _plot_cols.append(_facet_col)
    _plot_df = cyto_gated_events[_plot_cols].copy()
    if _plot_df.columns.duplicated().any():
        _plot_df = _plot_df.loc[:, ~_plot_df.columns.duplicated()]

    _plot_df[_scatter_x] = pd.to_numeric(_col_series(_plot_df, _scatter_x), errors="coerce")
    _plot_df[_scatter_y] = pd.to_numeric(_col_series(_plot_df, _scatter_y), errors="coerce")
    _plot_df[_fluor] = pd.to_numeric(_col_series(_plot_df, _fluor), errors="coerce")
    _plot_df = _plot_df[np.isfinite(_plot_df[_scatter_x]) & np.isfinite(_plot_df[_scatter_y])]

    _low_clip = str(cyto_low_clip_dropdown.value)
    if _low_clip != "off":
        _q = float(_low_clip)
        if _facet_col:
            _mask = np.zeros(len(_plot_df), dtype=bool)
            for _, _group in _plot_df.groupby(_facet_col, dropna=False):
                _x = _group[_scatter_x].to_numpy()
                _y = _group[_scatter_y].to_numpy()
                _x_lo = float(np.nanquantile(_x, _q))
                _y_lo = float(np.nanquantile(_y, _q))
                _mask[_group.index] = (_x >= _x_lo) & (_y >= _y_lo)
            _plot_df = _plot_df.loc[_mask]
        else:
            _x = _plot_df[_scatter_x].to_numpy()
            _y = _plot_df[_scatter_y].to_numpy()
            _x_lo = float(np.nanquantile(_x, _q))
            _y_lo = float(np.nanquantile(_y, _q))
            _plot_df = _plot_df[(_plot_df[_scatter_x] >= _x_lo) & (_plot_df[_scatter_y] >= _y_lo)]

    _x_scale = str(cyto_scatter_scale_x.value)
    _y_scale = str(cyto_scatter_scale_y.value)
    if _x_scale == "log":
        _plot_df = _plot_df[_plot_df[_scatter_x] > 0]
    if _y_scale == "log":
        _plot_df = _plot_df[_plot_df[_scatter_y] > 0]
    if _plot_df.empty:
        mo.stop(True, mo.md("No scatter data available after filtering."))

    _plot_df = _downsample(_plot_df, _max_events, [c for c in [_facet_col, _hue_col] if c])

    _plot_df["legend_hue"] = _plot_df[_hue_col].fillna("(missing)").astype(str) if _hue_col in _plot_df.columns else "(missing)"
    _hue_values = sorted({str(v) for v in _plot_df["legend_hue"].dropna().unique()})
    if not _hue_values:
        _hue_values = ["(missing)"]
        _plot_df["legend_hue"] = "(missing)"

    _color_map = {hue: _palette[idx % len(_palette)] for idx, hue in enumerate(_hue_values)}
    _fallback_title = None
    if _facet_col is None and "design_id" in _plot_df.columns:
        _design_vals = [str(v) for v in _plot_df["design_id"].dropna().unique()]
        if len(_design_vals) == 1:
            _fallback_title = f"design_id: {_design_vals[0]}"

    _hist_df = _plot_df.copy()
    _hist_scale = str(cyto_hist_scale.value)
    _xlabel = f"{_fluor} (a.u.)"
    _hist_value_col = _fluor
    _hist_xscale = "linear"
    if _hist_scale == "log":
        _hist_df = _hist_df[_hist_df[_fluor] > 0].copy()
        if _hist_df.empty:
            mo.stop(True, mo.md("Log scale requires positive fluor values."))
        _xlabel = f"{_fluor} (log)"
        _hist_xscale = "log"
    elif _hist_scale == "asinh":
        _cofactor = 150.0
        _hist_df["fluor_plot_value"] = np.arcsinh(_hist_df[_fluor].to_numpy() / _cofactor)
        _xlabel = f"asinh({_fluor}/{_cofactor:g})"
        _hist_value_col = "fluor_plot_value"
    elif _hist_scale == "symlog":
        _linthresh = 50.0
        _vals = _hist_df[_fluor].to_numpy()
        _hist_df["fluor_plot_value"] = np.sign(_vals) * np.log10(1 + np.abs(_vals) / _linthresh)
        _xlabel = f"symlog({_fluor})"
        _hist_value_col = "fluor_plot_value"

    _hist_values = _hist_df[_hist_value_col].to_numpy()
    _hist_values = _hist_values[np.isfinite(_hist_values)]
    if _hist_values.size == 0:
        mo.stop(True, mo.md("No finite histogram values available."))
    _hist_min = float(np.nanmin(_hist_values))
    _hist_max = float(np.nanmax(_hist_values))
    if not np.isfinite(_hist_min) or not np.isfinite(_hist_max):
        mo.stop(True, mo.md("Histogram range could not be determined."))
    if _hist_max <= _hist_min:
        _hist_max = _hist_min + 1e-6
    if _hist_xscale == "log":
        _hist_bins = np.logspace(np.log10(_hist_min), np.log10(_hist_max), 201)
    else:
        _hist_bins = np.linspace(_hist_min, _hist_max, 201)

    _facet_values = [None]
    if _facet_col and _facet_col in _plot_df.columns:
        _facet_values = sorted({str(v) for v in _plot_df[_facet_col].dropna().unique()})
        if not _facet_values:
            _facet_values = [None]

    _ncols = max(len(_facet_values), 1)
    cyto_marker_fig, axes = plt.subplots(
        nrows=2,
        ncols=_ncols,
        figsize=(4.0 * _ncols, 6.0),
        constrained_layout=True,
    )
    if _ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    _hist_ylabel = "Count"
    for _col_idx, _facet in enumerate(_facet_values):
        _scatter_ax = axes[0, _col_idx]
        _hist_ax = axes[1, _col_idx]

        _scatter_ax.set_facecolor("white")
        _hist_ax.set_facecolor("white")

        _facet_plot = _plot_df
        _facet_hist = _hist_df
        if _facet is not None and _facet_col:
            _facet_plot = _facet_plot[_facet_plot[_facet_col].astype(str) == _facet]
            _facet_hist = _facet_hist[_facet_hist[_facet_col].astype(str) == _facet]

        for _hue in _hue_values:
            _group = _facet_plot[_facet_plot["legend_hue"] == _hue]
            if _group.empty:
                continue
            _x_vals = pd.to_numeric(_col_series(_group, _scatter_x), errors="coerce")
            _y_vals = pd.to_numeric(_col_series(_group, _scatter_y), errors="coerce")
            _mask = np.isfinite(_x_vals) & np.isfinite(_y_vals)
            if not _mask.any():
                continue
            _scatter_ax.scatter(
                _x_vals[_mask],
                _y_vals[_mask],
                s=12,
                alpha=0.35,
                color=_color_map.get(_hue, "#000000"),
                label=_hue,
            )

        if _x_scale == "log":
            _scatter_ax.set_xscale("log")
        elif _x_scale == "symlog":
            _scatter_ax.set_xscale("symlog", linthresh=50.0)
        if _y_scale == "log":
            _scatter_ax.set_yscale("log")
        elif _y_scale == "symlog":
            _scatter_ax.set_yscale("symlog", linthresh=50.0)

        _scatter_ax.set_xlabel(_scatter_x, color="black")
        _scatter_ax.set_ylabel(_scatter_y, color="black")
        _scatter_ax.tick_params(colors="black")
        _scatter_ax.spines["left"].set_visible(True)
        _scatter_ax.spines["bottom"].set_visible(True)
        for _spine in ("left", "bottom"):
            _scatter_ax.spines[_spine].set_color("black")

        _title_text = None
        if _facet is not None and _facet_col:
            _title_text = f"{_facet_col}: {_facet}"
        elif _facet is not None:
            _title_text = _facet
        elif _fallback_title:
            _title_text = _fallback_title
        if _title_text is not None:
            _scatter_ax.set_title(_title_text, color="black")

        _scatter_leg = _scatter_ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=8,
        )
        if _scatter_leg is not None:
            for _text in _scatter_leg.get_texts():
                _text.set_color("black")

        for _hue in _hue_values:
            _group = _facet_hist[_facet_hist["legend_hue"] == _hue]
            if _group.empty:
                continue
            _values = _group[_hist_value_col].to_numpy()
            _hist_ax.hist(
                _values,
                bins=_hist_bins,
                alpha=0.25,
                histtype="stepfilled",
                color=_color_map.get(_hue, "#000000"),
                label=_hue,
            )

        if _hist_xscale == "log":
            _hist_ax.set_xscale("log")
        _hist_ax.set_xlabel(_xlabel, color="black")
        _hist_ax.set_ylabel(_hist_ylabel, color="black")
        _hist_ax.tick_params(colors="black")
        _hist_ax.spines["left"].set_visible(True)
        _hist_ax.spines["bottom"].set_visible(True)
        for _spine in ("left", "bottom"):
            _hist_ax.spines[_spine].set_color("black")

        _threshold_plot = cyto_threshold_value_final
        if np.isfinite(_threshold_plot):
            if _hist_scale == "asinh":
                _threshold_plot = np.arcsinh(_threshold_plot / 150.0)
            elif _hist_scale == "symlog":
                _threshold_plot = np.sign(_threshold_plot) * np.log10(
                    1 + np.abs(_threshold_plot) / 50.0
                )
            _hist_ax.axvline(_threshold_plot, color="black", linestyle="--", linewidth=1.0)

        _hist_leg = _hist_ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=8,
        )
        if _hist_leg is not None:
            for _text in _hist_leg.get_texts():
                _text.set_color("black")

    cyto_marker_fig.patch.set_facecolor("white")
    return cyto_gating_fig, cyto_marker_fig

@app.cell(hide_code=True)
def _(
    cyto_stats_group,
    cyto_stats_sample,
    cyto_threshold_control_dropdown,
    cyto_threshold_group_dropdown,
    cyto_threshold_mode,
    cyto_threshold_quantile,
    cyto_threshold_value,
    mo,
):
    _threshold_row_1 = mo.hstack(
        [
            cyto_threshold_mode,
            cyto_threshold_group_dropdown,
            cyto_threshold_control_dropdown,
        ],
        gap=1,
        align="start",
        justify="start",
    )
    _threshold_row_2 = mo.hstack(
        [cyto_threshold_quantile, cyto_threshold_value],
        gap=1,
        align="start",
        justify="start",
    )
    _elements = [
        mo.md("### Statistics and thresholding"),
        mo.md(
            "Threshold sets `% positive` and the dashed histogram line. **Manual** uses the numeric threshold; **control quantile** derives the threshold from the selected control group."
        ),
        _threshold_row_1,
        _threshold_row_2,
        mo.md("### Per-sample statistics"),
        mo.ui.table(cyto_stats_sample, page_size=10),
    ]
    if cyto_stats_group is not None:
        _elements.extend(
            [
                mo.md("### Group summary statistics"),
                mo.ui.table(cyto_stats_group, page_size=10),
            ]
        )
    cyto_stats_panel = mo.vstack(_elements)
    return cyto_stats_panel

@app.cell(hide_code=True)
def _(cyto_qc_table, mo):
    cyto_qc_panel = mo.vstack(
        [
            mo.md("### Quality control summary"),
            mo.md(
                "`pct_nonpositive` is computed on the selected fluorescence channel before gating; high values often indicate compensated data and can break strict log plots."
            ),
            mo.ui.table(cyto_qc_table, page_size=10),
        ]
    )
    return cyto_qc_panel

@app.cell(hide_code=True)
def _(mo, outputs_dir, spec):
    exports_cfg = spec.paths.exports
    exports_dir = outputs_dir if exports_cfg in ("", ".", "./") else outputs_dir / str(exports_cfg)
    cyto_export_format = mo.ui.dropdown(
        options=["pdf", "png", "svg"],
        value="pdf",
        label="Plot format",
        full_width=True,
    )
    cyto_plot_export_path = mo.ui.text(
        value=str(exports_dir / "cytometry_eda.pdf"),
        label="Plot export path",
        full_width=True,
    )
    cyto_stats_export_path = mo.ui.text(
        value=str(exports_dir / "cytometry_stats.csv"),
        label="Stats export path (CSV)",
        full_width=True,
    )
    cyto_gate_export_path = mo.ui.text(
        value=str(exports_dir / "cytometry_gates.json"),
        label="Gate config path (JSON)",
        full_width=True,
    )
    cyto_export_button = mo.ui.run_button(
        label="Export cytometry outputs",
        kind="success",
    )
    cyto_export_panel = mo.vstack(
        [
            cyto_export_format,
            cyto_plot_export_path,
            cyto_stats_export_path,
            cyto_gate_export_path,
            cyto_export_button,
        ]
    )
    return (
        exports_dir,
        cyto_export_format,
        cyto_plot_export_path,
        cyto_stats_export_path,
        cyto_gate_export_path,
        cyto_export_button,
        cyto_export_panel,
    )

@app.cell
def _(
    Path,
    cyto_export_button,
    cyto_export_format,
    cyto_gate_config,
    cyto_gate_export_path,
    cyto_marker_fig,
    cyto_plot_export_path,
    cyto_stats_export_path,
    cyto_stats_sample,
    exports_dir,
    json,
    mo,
):
    if cyto_marker_fig is None:
        mo.stop(True, mo.md("No cytometry plot available to export."))
    if cyto_stats_sample is None:
        mo.stop(True, mo.md("No cytometry stats available to export."))
    _export_message = None
    if cyto_export_button.value:
        _format = str(cyto_export_format.value).lower()

        _plot_target = Path(str(cyto_plot_export_path.value)).expanduser()
        if not _plot_target.is_absolute():
            _plot_target = (exports_dir / _plot_target).resolve()
        if _plot_target.suffix and _plot_target.suffix.lower() != f".{_format}":
            mo.stop(True, mo.md("Plot export path extension must match selected format."))
        _plot_target.parent.mkdir(parents=True, exist_ok=True)
        cyto_marker_fig.savefig(_plot_target, format=_format, bbox_inches="tight", facecolor="white")

        _stats_target = Path(str(cyto_stats_export_path.value)).expanduser()
        if not _stats_target.is_absolute():
            _stats_target = (exports_dir / _stats_target).resolve()
        _stats_target.parent.mkdir(parents=True, exist_ok=True)
        cyto_stats_sample.to_csv(_stats_target, index=False)

        _gate_target = Path(str(cyto_gate_export_path.value)).expanduser()
        if not _gate_target.is_absolute():
            _gate_target = (exports_dir / _gate_target).resolve()
        _gate_target.parent.mkdir(parents=True, exist_ok=True)
        _gate_target.write_text(json.dumps(cyto_gate_config, indent=2), encoding="utf-8")

        _export_message = (
            f"Saved plot to `{_plot_target}`; stats to `{_stats_target}`; gate config to `{_gate_target}`."
        )
    if _export_message is not None:
        _ = mo.md(_export_message)
    return

@app.cell(hide_code=True)
def _(
    cyto_intro_panel,
    cyto_controls_panel,
    cyto_gate_panel,
    cyto_gating_fig,
    cyto_marker_fig,
    cyto_qc_panel,
    cyto_stats_panel,
    cyto_export_panel,
    eda_base_panel,
    mo,
):
    cyto_tabs = mo.vstack(
        [
            eda_base_panel,
            cyto_intro_panel,
            mo.md("## Controls"),
            cyto_controls_panel,
            mo.md("## Gating"),
            cyto_gate_panel,
            cyto_gating_fig,
            mo.md("## Fluorescence"),
            cyto_marker_fig,
            mo.md("## Statistics"),
            cyto_stats_panel,
            cyto_qc_panel,
            mo.md("## Export"),
            cyto_export_panel,
        ],
        gap=1,
    )
    cyto_tabs
'''

EXPERIMENT_EDA_BASIC_TEMPLATE = (
    EXPERIMENT_EDA_BASE_TEMPLATE + EXPERIMENT_EDA_BASE_LAYOUT_TEMPLATE + EXPERIMENT_EDA_TEMPLATE_FOOTER
)
EXPERIMENT_EDA_MICROPLATE_TEMPLATE = EXPERIMENT_EDA_BASIC_TEMPLATE
EXPERIMENT_EDA_CYTOMETRY_TEMPLATE = (
    EXPERIMENT_EDA_BASE_TEMPLATE + EXPERIMENT_EDA_CYTOMETRY_EXTENSION_TEMPLATE + EXPERIMENT_EDA_TEMPLATE_FOOTER
)
EXPERIMENT_NOTEBOOK_EDA_TEMPLATE = EXPERIMENT_EDA_BASIC_TEMPLATE
EXPERIMENT_SFXI_EXTENSION_TEMPLATE = '''
@app.cell(hide_code=True)
def _():
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        import numpy as np
    except Exception:
        np = None

    altair_err = None
    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()
    except Exception as exc:
        alt = None
        altair_err = exc

    from reader.lib.sfxi.api import load_sfxi_config
    from reader.lib.sfxi.run import build_vec8_from_tidy
    from reader.lib.sfxi.selection import cornerize_and_aggregate, REQUIRED_COLS

    return (
        pd,
        np,
        alt,
        altair_err,
        load_sfxi_config,
        build_vec8_from_tidy,
        cornerize_and_aggregate,
        REQUIRED_COLS,
    )

@app.cell(hide_code=True)
def _(outputs_dir, spec):
    exports_cfg = spec.paths.exports
    exports_dir = outputs_dir if exports_cfg in ("", ".", "./") else outputs_dir / str(exports_cfg)
    return exports_dir

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## SFXI 8-vector Builder
This section mirrors the **setpoint‑fidelity × intensity** definition used in OPAL: each design is summarized by an **8‑vector** with four **logic** values (v00..v11 in [0,1]) and four **intensity** values (y*00..y*11 in log2, reference‑normalized). The logic half captures **shape** (which corners turn on/off), while the intensity half captures **effect size** after reference normalization to make runs comparable.

Workflow:
- choose a **time slice** and map treatments to the **00/10/01/11** corners
- logic: `log2(YFP/CFP)` → per‑design min–max → **v00..v11**
- intensity: `log2((YFP/OD600)/(reference+α)+δ)` → **y*00..y*11**

The 8-vector here uses the same `transform/sfxi` code and writes XLSX + JSON logs to the experiment's exports folder."""
    )

@app.cell(hide_code=True)
def _(mo, spec):
    sfxi_step = None
    for step in spec.pipeline.steps:
        if str(getattr(step, "uses", "")) == "transform/sfxi":
            sfxi_step = step
    if sfxi_step is None:
        mo.stop(
            True,
            mo.md(
                "No `transform/sfxi` step found in this experiment. "
                "Add an SFXI step to `config.yaml` or see `docs/sfxi_vec8_in_reader.md`."
            ),
        )
    return sfxi_step

@app.cell(hide_code=True)
def _(mo, sfxi_step):
    def _step_to_dict(step):
        if hasattr(step, "model_dump"):
            return dict(step.model_dump(by_alias=True))
        if isinstance(step, dict):
            return dict(step)
        return {
            _k: getattr(step, _k)
            for _k in dir(step)
            if not _k.startswith("_") and not callable(getattr(step, _k))
        }

    step_dict = _step_to_dict(sfxi_step)
    sfxi_step_cfg = step_dict.get("with") or step_dict.get("with_") or {}
    sfxi_step_id = step_dict.get("id", "")
    if not sfxi_step_cfg:
        mo.stop(True, mo.md(f"Step `{sfxi_step_id}` has no SFXI config (`with`)."))
    return sfxi_step_cfg, sfxi_step_id

@app.cell(hide_code=True)
def _(load_sfxi_config, mo, sfxi_step_cfg, sfxi_step_id):
    try:
        sfxi_cfg = load_sfxi_config(sfxi_step_cfg)
    except Exception as exc:
        mo.stop(True, mo.md(f"SFXI config error in `{sfxi_step_id}`: `{exc}`"))
    return sfxi_cfg

@app.cell(hide_code=True)
def _(REQUIRED_COLS, df, mo, sfxi_cfg):
    _cols = list(df.columns) if hasattr(df, "columns") else []
    required = list(REQUIRED_COLS)
    if sfxi_cfg.time_column not in required:
        required.append(sfxi_cfg.time_column)
    for _c in sfxi_cfg.design_by:
        if _c not in required:
            required.append(_c)
    missing = [c for c in required if c not in _cols]
    if missing:
        mo.stop(
            True,
            mo.md(
                "Selected dataset is not SFXI-compatible. "
                f"Missing column(s): {', '.join(missing)}. "
                "Choose the tidy+map artifact (validator/to_tidy_plus_map) and see `docs/sfxi_vec8_in_reader.md`."
            ),
        )
    return required

@app.cell(hide_code=True)
def _(df, mo, pd, pl):
    if pd is None:
        mo.stop(True, mo.md("Pandas is required for SFXI computations."))
    if pl is not None and df.__class__.__module__.startswith("polars"):
        tidy_pd = df.to_pandas()
    else:
        tidy_pd = df
    return tidy_pd

@app.cell(hide_code=True)
def _(mo, np, pd, sfxi_cfg, tidy_pd):
    label_col = sfxi_cfg.design_by[0]
    time_col = sfxi_cfg.time_column

    design_vals = sorted({str(_v) for _v in tidy_pd[label_col].dropna().unique().tolist()})
    if not design_vals:
        mo.stop(True, mo.md("No design values found for SFXI selection."))

    time_series = pd.to_numeric(tidy_pd[time_col], errors="coerce").dropna()
    time_vals = sorted({float(_v) for _v in time_series.tolist()})
    if not time_vals:
        mo.stop(True, mo.md("No numeric time values found for SFXI selection."))
    time_min = float(time_vals[0])
    time_max = float(time_vals[-1])
    if len(time_vals) > 1:
        if np is not None:
            _diffs = np.diff(np.array(time_vals, dtype=float))
            _diffs = _diffs[_diffs > 0]
            time_step = float(np.min(_diffs)) if _diffs.size else 0.25
        else:
            _diffs = [b - a for a, b in zip(time_vals[:-1], time_vals[1:]) if b > a]
            time_step = min(_diffs) if _diffs else 0.25
    else:
        time_step = 0.25

    default_time = sfxi_cfg.target_time_h if sfxi_cfg.target_time_h is not None else time_max
    if np is not None:
        try:
            if np.isnan(default_time):
                default_time = time_max
        except Exception:
            pass
    if default_time < time_min or default_time > time_max:
        default_time = time_max

    return label_col, time_col, design_vals, time_min, time_max, time_step, default_time

@app.cell(hide_code=True)
def _(pd, tidy_pd, time_col):
    induction_time_h = None
    explicit_cols = [
        "induction_time_h",
        "induction_time",
        "time_of_induction_h",
        "time_of_induction",
    ]
    for col in explicit_cols:
        if col in tidy_pd.columns:
            vals = pd.to_numeric(tidy_pd[col], errors="coerce").dropna()
            if not vals.empty:
                induction_time_h = float(vals.iloc[0])
                break

    if induction_time_h is None and "sheet_index" in tidy_pd.columns:
        sheet_vals = pd.to_numeric(tidy_pd["sheet_index"], errors="coerce").dropna()
        if not sheet_vals.empty:
            min_sheet = float(sheet_vals.min())
            sheet_series = pd.to_numeric(tidy_pd["sheet_index"], errors="coerce")
            times = pd.to_numeric(tidy_pd.loc[sheet_series > min_sheet, time_col], errors="coerce").dropna()
            if not times.empty:
                induction_time_h = float(times.min())

    return induction_time_h

@app.cell(hide_code=True)
def _(mo, np, pd, sfxi_cfg, tidy_pd, time_col):
    case_sensitive = bool(sfxi_cfg.treatment_case_sensitive)
    treatment_map = sfxi_cfg.treatment_map

    def _choose_treatment_column(df):
        candidates = [c for c in ("treatment", "treatment_alias") if c in df.columns]
        if not candidates:
            return None

        def _score(col):
            s = df[col].astype(str)
            if case_sensitive:
                want = {str(v) for v in treatment_map.values()}
                return int(s.isin(list(want)).sum())
            want = {str(v).strip().casefold() for v in treatment_map.values()}
            s = s.str.strip().str.casefold()
            return int(s.isin(list(want)).sum())

        scores = {c: _score(c) for c in candidates}
        return max(scores, key=lambda c: (scores[c], c == "treatment"))

    def _times_for_channel(channel):
        work = tidy_pd[tidy_pd["channel"] == channel].copy()
        if work.empty:
            return [], None
        treatment_col = _choose_treatment_column(work)
        if treatment_col is None:
            return [], None
        if case_sensitive:
            mapped = {str(v) for v in treatment_map.values()}
            work = work[work[treatment_col].astype(str).isin(mapped)].copy()
        else:
            mapped = {str(v).strip().casefold() for v in treatment_map.values()}
            norm = work[treatment_col].astype(str).str.strip().str.casefold()
            work = work[norm.isin(mapped)].copy()
        if work.empty:
            return [], treatment_col
        times = pd.to_numeric(work[time_col], errors="coerce").dropna()
        time_vals = sorted({float(_v) for _v in times.tolist()})
        return time_vals, treatment_col

    logic_times, logic_treatment_col = _times_for_channel(sfxi_cfg.response.logic_channel)
    intensity_times, intensity_treatment_col = _times_for_channel(sfxi_cfg.response.intensity_channel)

    if logic_treatment_col is None or intensity_treatment_col is None:
        mo.stop(True, mo.md("SFXI selection requires `treatment` or `treatment_alias` columns."))
    if not logic_times:
        mo.stop(
            True,
            mo.md(
                "No time values found for the logic channel after filtering to the configured treatments."
            ),
        )
    if not intensity_times:
        mo.stop(
            True,
            mo.md(
                "No time values found for the intensity channel after filtering to the configured treatments."
            ),
        )

    def _round_times(times):
        if np is not None:
            return [float(v) for v in np.round(np.array(times, dtype=float), 12)]
        return [round(float(v), 12) for v in times]

    common_times = sorted(set(_round_times(logic_times)) & set(_round_times(intensity_times)))
    if not common_times:
        mo.stop(
            True,
            mo.md(
                "No common time points found between the logic and intensity channels for the configured "
                "treatments. Check for missing values, adjust `treatment_map`, or choose a different dataset."
            ),
        )

    treatment_col = logic_treatment_col
    treatment_order = [sfxi_cfg.treatment_map[_k] for _k in ("00", "10", "01", "11")]
    return common_times, treatment_col, treatment_order

@app.cell(hide_code=True)
def _(default_time, design_vals, label_col, mo, sfxi_cfg, time_max, time_min, time_step):
    design_select = mo.ui.dropdown(
        options=design_vals,
        value=design_vals[0],
        label=f"Design ({label_col})",
        full_width=True,
    )
    time_mode = mo.ui.dropdown(
        options=["nearest", "last_before", "first_after", "exact"],
        value=sfxi_cfg.time_mode,
        label="Time mode",
        full_width=True,
    )
    time_slider = mo.ui.slider(
        start=time_min,
        stop=time_max,
        value=default_time,
        step=time_step,
        label="Target time (h)",
        full_width=True,
    )
    mo.hstack(
        [
            design_select,
            time_mode,
            time_slider,
        ]
    )
    return design_select, time_mode, time_slider

@app.cell(hide_code=True)
def _(common_times, mo, np, time_mode, time_slider):
    time_target_h = float(time_slider.value)
    mode = str(time_mode.value)

    def _choose_common_time(times, target, mode):
        if not times:
            return None
        time_list = sorted(float(_t) for _t in times)
        if target is None:
            return time_list[-1]
        target = float(target)
        if mode == "exact":
            if np is not None:
                for _t in time_list:
                    if np.isclose(_t, target, rtol=0, atol=1e-12):
                        return _t
            else:
                for _t in time_list:
                    if abs(_t - target) <= 1e-12:
                        return _t
            return None
        if mode == "nearest":
            return min(time_list, key=lambda _t: abs(_t - target))
        if mode == "last_before":
            candidates = [_t for _t in time_list if _t <= target]
            return max(candidates) if candidates else None
        if mode == "first_after":
            candidates = [_t for _t in time_list if _t >= target]
            return min(candidates) if candidates else None
        return None

    time_selected_h = _choose_common_time(common_times, time_target_h, mode)
    if time_selected_h is None:
        mo.stop(
            True,
            mo.md(
                f"No common time matches target {time_target_h:.3f} h (mode={mode}). "
                "Try a different time mode or adjust the slider."
            ),
        )
    return time_target_h, time_selected_h

@app.cell(hide_code=True)
def _(
    cornerize_and_aggregate,
    design_select,
    label_col,
    mo,
    sfxi_cfg,
    time_selected_h,
    tidy_pd,
):
    design_val = design_select.value
    design_mask = tidy_pd[label_col].astype(str) == str(design_val)
    subset_pd = tidy_pd[design_mask].copy()
    if subset_pd.empty:
        mo.stop(True, mo.md("No rows for the selected design."))
    selection_pd = tidy_pd.copy()

    target_time = float(time_selected_h)
    try:
        sel_logic = cornerize_and_aggregate(
            selection_pd,
            design_by=[label_col],
            treatment_map=sfxi_cfg.treatment_map,
            case_sensitive=sfxi_cfg.treatment_case_sensitive,
            time_column=sfxi_cfg.time_column,
            channel=sfxi_cfg.response.logic_channel,
            target_time_h=target_time,
            time_mode="exact",
            time_tolerance_h=sfxi_cfg.time_tolerance_h,
            require_all_corners_per_design=sfxi_cfg.require_all_corners_per_design,
        )
        sel_int = cornerize_and_aggregate(
            selection_pd,
            design_by=[label_col],
            treatment_map=sfxi_cfg.treatment_map,
            case_sensitive=sfxi_cfg.treatment_case_sensitive,
            time_column=sfxi_cfg.time_column,
            channel=sfxi_cfg.response.intensity_channel,
            target_time_h=target_time,
            time_mode="exact",
            time_tolerance_h=sfxi_cfg.time_tolerance_h,
            require_all_corners_per_design=sfxi_cfg.require_all_corners_per_design,
        )
    except Exception as exc:
        mo.stop(True, mo.md(f"Snapshot selection failed: `{exc}`"))

    chosen_time = target_time
    return subset_pd, sel_logic, sel_int, chosen_time

@app.cell(hide_code=True)
def _(mo, time_mode, time_selected_h, time_target_h):
    delta = abs(float(time_selected_h) - float(time_target_h))
    lines = [
        f"**Target time (slider): {float(time_target_h):.3f} h**",
        f"**Canonical snapshot time used: {float(time_selected_h):.3f} h**",
    ]
    if delta > 0:
        lines.append(f"Δ from target (mode={time_mode.value}): {delta:.3f} h")
    mo.md("## Snapshot selection\\n" + "\\n".join(lines))

@app.cell(hide_code=True)
def _(np, pd, subset_pd, time_col, treatment_col):
    _dfc = subset_pd[subset_pd["channel"] == "OD600"].copy()
    if _dfc.empty:
        ts_od600 = pd.DataFrame(columns=[time_col, treatment_col, "y_mean", "y_sd", "y_n", "y_lo", "y_hi"])
    else:
        _dfc[time_col] = pd.to_numeric(_dfc[time_col], errors="coerce")
        _dfc["value"] = pd.to_numeric(_dfc["value"], errors="coerce")
        _dfc = _dfc.dropna(subset=[time_col, "value", treatment_col])
        ts_od600 = (
            _dfc.groupby([time_col, treatment_col], dropna=False)["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        ts_od600 = ts_od600.rename(columns={"mean": "y_mean", "std": "y_sd", "count": "y_n"})
        ts_od600["y_sd"] = ts_od600["y_sd"].fillna(0.0)
        ts_od600["y_lo"] = ts_od600["y_mean"] - ts_od600["y_sd"]
        ts_od600["y_hi"] = ts_od600["y_mean"] + ts_od600["y_sd"]
    return ts_od600

@app.cell(hide_code=True)
def _(np, pd, subset_pd, time_col, treatment_col, sfxi_cfg, time_selected_h):
    bar_stats = pd.DataFrame(columns=[treatment_col, "y_mean", "y_sd", "y_n", "y_lo", "y_hi"])
    bar_points = pd.DataFrame(columns=[treatment_col, "value"])
    time_snap = None

    _dfc = subset_pd[subset_pd["channel"] == sfxi_cfg.response.logic_channel].copy()
    if not _dfc.empty:
        _dfc[time_col] = pd.to_numeric(_dfc[time_col], errors="coerce")
        _dfc["value"] = pd.to_numeric(_dfc["value"], errors="coerce")
        _dfc = _dfc.dropna(subset=[time_col, "value", treatment_col])
        time_snap = float(time_selected_h)
        if np is not None:
            _mask = np.isclose(_dfc[time_col], time_snap, atol=1e-9)
        else:
            _mask = (_dfc[time_col] - time_snap).abs() <= 1e-9
        _df_snap = _dfc[_mask].copy()
        if not _df_snap.empty:
            bar_stats = (
                _df_snap.groupby(treatment_col, dropna=False)["value"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            bar_stats = bar_stats.rename(columns={"mean": "y_mean", "std": "y_sd", "count": "y_n"})
            bar_stats["y_sd"] = bar_stats["y_sd"].fillna(0.0)
            bar_stats["y_lo"] = bar_stats["y_mean"] - bar_stats["y_sd"]
            bar_stats["y_hi"] = bar_stats["y_mean"] + bar_stats["y_sd"]
            bar_points = _df_snap[[treatment_col, "value"]].copy()

    return bar_stats, bar_points, time_snap

@app.cell(hide_code=True)
def _(
    alt,
    altair_err,
    bar_points,
    bar_stats,
    induction_time_h,
    mo,
    pd,
    sfxi_cfg,
    time_col,
    time_selected_h,
    treatment_col,
    treatment_order,
    ts_od600,
):
    if alt is None:
        mo.stop(True, mo.md(f"Altair is required for plotting: `{altair_err}`"))

    if ts_od600 is None or ts_od600.empty:
        mo.stop(True, mo.md("No OD600 data available for this design."))

    _snap_time = float(time_selected_h)

    _ts_tooltips = [
        alt.Tooltip(f"{time_col}:Q", title="Time (h)"),
        alt.Tooltip(f"{treatment_col}:N", title="Treatment"),
        alt.Tooltip("y_mean:Q", title="Mean"),
        alt.Tooltip("y_sd:Q", title="SD"),
        alt.Tooltip("y_n:Q", title="N"),
    ]
    _ts_width = 320
    _ts_height = 320
    _bar_width = 420
    _bar_height = 320
    _chart_spacing = 28

    _ts_base = alt.Chart(ts_od600).encode(
        x=alt.X(
            f"{time_col}:Q",
            axis=alt.Axis(labelOverlap=False),
        ),
        color=alt.Color(
            f"{treatment_col}:N",
            sort=treatment_order,
            scale=alt.Scale(domain=treatment_order),
            legend=alt.Legend(orient="bottom", title="Treatment"),
        ),
    )

    _ts_band = _ts_base.mark_area(opacity=0.2).encode(
        y=alt.Y("y_lo:Q", title="OD600"),
        y2=alt.Y2("y_hi:Q"),
        tooltip=_ts_tooltips,
    )
    _ts_line = _ts_base.mark_line().encode(
        y=alt.Y("y_mean:Q", title="OD600"),
        tooltip=_ts_tooltips,
    )

    _y_max = ts_od600["y_hi"].max()
    if pd.isna(_y_max):
        _y_max = ts_od600["y_mean"].max()
    if pd.isna(_y_max):
        _y_max = 0.0

    _rule_df = pd.DataFrame(
        {
            time_col: [_snap_time],
            "y": [float(_y_max)],
            "label": [f"t = {_snap_time:.3f} h"],
        }
    )
    _ts_rule = alt.Chart(_rule_df).mark_rule(color="black").encode(x=alt.X(f"{time_col}:Q"))
    _ts_text = alt.Chart(_rule_df).mark_text(color="black", align="left", dx=6, dy=-6).encode(
        x=alt.X(f"{time_col}:Q"),
        y=alt.Y("y:Q"),
        text="label",
    )

    _induction_time = None
    if induction_time_h is not None:
        try:
            _val = float(induction_time_h)
            if not pd.isna(_val):
                _induction_time = _val
        except Exception:
            _induction_time = None

    _ts_layers = [_ts_band, _ts_line]
    if _induction_time is not None:
        _ind_df = pd.DataFrame({time_col: [_induction_time]})
        _ts_induction = alt.Chart(_ind_df).mark_rule(color="red", strokeDash=[6, 4]).encode(
            x=alt.X(f"{time_col}:Q")
        )
        _ts_layers.append(_ts_induction)
    _ts_layers.extend([_ts_rule, _ts_text])

    ts_chart = alt.layer(*_ts_layers).properties(
        width=_ts_width,
        height=_ts_height,
    )

    if bar_stats is None or bar_stats.empty:
        mo.stop(True, mo.md("No snapshot data available at this time."))

    _bar_axis = alt.Axis(labelLimit=0, labelOverlap=False, labelAngle=-45)
    _bar_title = f"{sfxi_cfg.response.logic_channel} snapshot (mean)"
    _bar_base = alt.Chart(bar_stats).encode(
        x=alt.X(
            f"{treatment_col}:N",
            sort=treatment_order,
            axis=_bar_axis,
        ),
        y=alt.Y("y_mean:Q", title=_bar_title),
        tooltip=[
            alt.Tooltip(f"{treatment_col}:N", title="Treatment"),
            alt.Tooltip("y_mean:Q", title="Mean"),
            alt.Tooltip("y_sd:Q", title="SD"),
            alt.Tooltip("y_n:Q", title="N"),
        ],
    )

    _bar_bars = _bar_base.mark_bar().encode(
        color=alt.Color(
            f"{treatment_col}:N",
            sort=treatment_order,
            scale=alt.Scale(domain=treatment_order),
            legend=None,
        )
    )
    _bar_err_rule = _bar_base.mark_rule(color="black").encode(
        y=alt.Y("y_lo:Q"),
        y2=alt.Y2("y_hi:Q"),
    )
    _bar_err_low = _bar_base.mark_tick(color="black", orient="horizontal", size=8, thickness=1.5).encode(
        y=alt.Y("y_lo:Q"),
    )
    _bar_err_high = _bar_base.mark_tick(color="black", orient="horizontal", size=8, thickness=1.5).encode(
        y=alt.Y("y_hi:Q"),
    )

    _bar_layers = [_bar_bars, _bar_err_rule, _bar_err_low, _bar_err_high]
    if bar_points is not None and not bar_points.empty:
        _bar_points = alt.Chart(bar_points).mark_point(filled=True, strokeWidth=0, size=50).encode(
            x=alt.X(f"{treatment_col}:N", sort=treatment_order, axis=_bar_axis),
            y=alt.Y("value:Q"),
            tooltip=[
                alt.Tooltip(f"{treatment_col}:N", title="Treatment"),
                alt.Tooltip("value:Q", title="Value"),
            ],
        )
        _bar_layers.append(_bar_points)

    bar_chart = alt.layer(*_bar_layers).properties(
        width=_bar_width,
        height=_bar_height,
    )

    chart = (
        alt.hconcat(ts_chart, bar_chart, spacing=_chart_spacing)
        .resolve_scale(color="shared")
        .configure(background="white")
        .configure_view(fill="white")
        .configure_axis(
            domain=True,
            domainColor="black",
            domainWidth=1,
            tickColor="black",
            labelColor="black",
            titleColor="black",
            labelFontSize=13,
            titleFontSize=14,
        )
        .configure_legend(
            labelColor="black",
            titleColor="black",
            labelFontSize=13,
            titleFontSize=13,
        )
        .configure_title(color="black", fontSize=15)
        .configure_text(color="black", fontSize=13)
    )
    mo.ui.altair_chart(chart)

@app.cell(hide_code=True)
def _(build_vec8_from_tidy, mo, sfxi_step_cfg, time_selected_h, tidy_pd):
    cfg_payload = dict(sfxi_step_cfg)
    cfg_payload["target_time_h"] = float(time_selected_h)
    cfg_payload["time_mode"] = "exact"
    try:
        vec8_result = build_vec8_from_tidy(tidy_pd, cfg_payload)
    except Exception as exc:
        mo.stop(True, mo.md(f"8-vector computation failed: `{exc}`"))
    return vec8_result

@app.cell(hide_code=True)
def _(mo, vec8_result):
    mo.vstack(
        [
            mo.md("## 8-vector output"),
            mo.ui.table(vec8_result.vec8, page_size=10),
        ]
    )

@app.cell(hide_code=True)
def _(mo, vec8_result):
    _ref = vec8_result.log.get("reference", {}) if hasattr(vec8_result, "log") else {}
    _lines = [
        f"**reference.design_id:** `{_ref.get('design_id')}`",
        f"**reference.design_id_resolved:** `{_ref.get('design_id_resolved')}`",
        f"**reference.stat:** `{_ref.get('stat')}`",
    ]
    mo.md("## Reference anchor\\n" + "\\n".join(_lines))

@app.cell(hide_code=True)
def _(Path, exports_dir, mo, sfxi_cfg):
    export_dir = exports_dir / sfxi_cfg.output_subdir
    xlsx_name = Path(sfxi_cfg.vec8_filename).with_suffix(".xlsx").name
    export_path = export_dir / xlsx_name
    export_button = mo.ui.run_button(label="Export 8-vector (XLSX)", kind="success")
    log_name = sfxi_cfg.log_filename
    mo.vstack(
        [
            mo.md("## Export 8-vector"),
            mo.md(f"Export path: `{export_path}`"),
            export_button,
            mo.md(f"Log will be written as `{log_name}` next to the XLSX."),
        ]
    )
    return export_button, export_path, log_name

@app.cell(hide_code=True)
def _(export_button, export_path, json, log_name, mo, vec8_result):
    if not export_button.value:
        mo.stop(True)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    vec8_df = vec8_result.vec8
    try:
        vec8_df.to_excel(export_path, index=False)
    except Exception as exc:
        mo.stop(
            True,
            mo.md(
                f"XLSX export failed: `{exc}`. "
                "Ensure `openpyxl` is installed (included in reader core deps; run `uv sync` if missing)."
            ),
        )
    log_path = export_path.parent / log_name
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(vec8_result.log, fh, indent=2, sort_keys=True, default=str)
    mo.md(f"Exported 8-vector to `{export_path}` and log to `{log_path}`.")
'''
EXPERIMENT_SFXI_EDA_TEMPLATE = (
    EXPERIMENT_EDA_BASE_TEMPLATE
    + EXPERIMENT_EDA_BASE_LAYOUT_TEMPLATE
    + EXPERIMENT_SFXI_EXTENSION_TEMPLATE
    + EXPERIMENT_EDA_TEMPLATE_FOOTER
)

NOTEBOOK_PRESETS: dict[str, dict[str, str]] = {
    "notebook/eda": {
        "description": "Minimal artifact explorer (formerly notebook/plots).",
        "template": EXPERIMENT_NOTEBOOK_EDA_TEMPLATE,
    },
    "notebook/basic": {
        "description": "Minimal artifact explorer with design/treatment table and df.parquet preview.",
        "template": EXPERIMENT_EDA_BASIC_TEMPLATE,
    },
    "notebook/microplate": {
        "description": "Minimal artifact explorer (same scaffold as notebook/basic).",
        "template": EXPERIMENT_EDA_MICROPLATE_TEMPLATE,
    },
    "notebook/cytometry": {
        "description": "Cytometry EDA scaffold (FSC/SSC scatter + fluorophore histograms).",
        "template": EXPERIMENT_EDA_CYTOMETRY_TEMPLATE,
    },
    "notebook/sfxi_eda": {
        "description": "SFXI vec8 explorer (EDA scaffold + time slice → corners → vec8).",
        "template": EXPERIMENT_SFXI_EDA_TEMPLATE,
    },
}

NOTEBOOK_PRESET_ALIASES: dict[str, str] = {
    "notebook/plots": "notebook/eda",
}


def list_notebook_presets() -> list[tuple[str, str]]:
    return sorted((name, info["description"]) for name, info in NOTEBOOK_PRESETS.items())


def normalize_notebook_preset(name: str) -> str:
    return NOTEBOOK_PRESET_ALIASES.get(name, name)


def resolve_notebook_preset(name: str) -> str:
    name = normalize_notebook_preset(name)
    if name not in NOTEBOOK_PRESETS:
        opts = ", ".join(sorted(NOTEBOOK_PRESETS))
        raise ConfigError(f"Unknown notebook preset {name!r}. Available presets: {opts}")
    return NOTEBOOK_PRESETS[name]["template"]


def write_experiment_notebook(
    target: Path,
    *,
    preset: str = "notebook/eda",
    overwrite: bool = False,
    plot_specs: list[dict] | None = None,
) -> tuple[Path, bool]:
    if target.exists() and not overwrite:
        return target, False
    preset = normalize_notebook_preset(preset)
    template = resolve_notebook_preset(preset)
    if preset == "notebook/eda" and "__PLOT_SPECS__" in template:
        payload = plot_specs or []
        template = template.replace("__PLOT_SPECS__", repr(payload))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    return target, True
