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

EXPERIMENT_EDA_BASE_TEMPLATE = '''import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

@app.cell
def _():
    from pathlib import Path
    import json

    import altair as alt
    import marimo as mo
    try:
        import polars as pl
    except Exception:
        pl = None
    try:
        import pandas as pd
    except Exception:
        pd = None
    from reader.core.config_model import ReaderSpec

    return (
        Path,
        json,
        alt,
        mo,
        pl,
        pd,
        ReaderSpec,
    )

@app.cell
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
    _spec = _load_spec(exp_dir)
    outputs_dir = Path(_spec.paths.outputs).resolve()
    plots_dir = (outputs_dir / _spec.paths.plots).resolve()
    exports_dir = (outputs_dir / _spec.paths.exports).resolve()
    data_groupings = _spec.data.groupings or {}
    exp_meta = {
        "id": _spec.experiment.id,
        "title": _spec.experiment.title or "",
    }
    pipeline_step_ids = [step.id for step in _spec.pipeline.steps]
    return (
        exp_dir,
        exp_meta,
        outputs_dir,
        pipeline_step_ids,
        data_groupings,
        plots_dir,
        exports_dir,
    )

@app.cell
def _(json, outputs_dir):
    artifact_info = {}
    artifact_note = ""
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
    return artifact_info, artifact_labels, artifact_note

@app.cell
def _(artifact_info, artifact_labels, artifact_note, mo, pipeline_step_ids):
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

@app.cell
def _(artifact_dropdown, artifact_info):
    if artifact_dropdown is None:
        selected_label = None
        artifact_path = None
    else:
        selected_label = artifact_dropdown.value
        artifact_path = artifact_info.get(selected_label, {}).get("path")
    return selected_label, artifact_path

@app.cell
def _(artifact_path, pd, pl):
    df_active = None
    df_backend = None
    df_error = None
    _pl_error = None
    _pd_error = None

    if artifact_path is not None:
        if pl is not None:
            try:
                df_active = pl.read_parquet(artifact_path)
                df_backend = "polars"
            except Exception as exc:
                _pl_error = str(exc)
        if df_active is None and pd is not None:
            try:
                df_active = pd.read_parquet(artifact_path)
                df_backend = "pandas"
            except Exception as exc:
                _pd_error = str(exc)
        if df_active is None:
            if pl is None and pd is None:
                df_error = "Neither polars nor pandas is installed. Install the notebooks group."
            else:
                _details = []
                if _pl_error:
                    _details.append(f"polars: {_pl_error}")
                if _pd_error:
                    _details.append(f"pandas: {_pd_error}")
                _suffix = "; ".join(_details) if _details else "unknown error"
                df_error = f"Failed to read parquet with available backends ({_suffix})."
    return df_active, df_backend, df_error

@app.cell
def _(df_active, pl):
    df_rows = None
    df_cols = None
    if df_active is not None:
        if pl is not None and df_active.__class__.__module__.startswith("polars"):
            df_rows = df_active.height
            df_cols = df_active.width
        else:
            df_rows, df_cols = df_active.shape
    return df_rows, df_cols

@app.cell
def _(df_active, pd, pl):
    design_treatment_rows = []
    design_treatment_note = ""
    if df_active is None:
        design_treatment_note = "No dataset selected yet."
    else:
        _columns = list(df_active.columns) if hasattr(df_active, "columns") else []
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
                try:
                    if pl is not None and df.__class__.__module__.startswith("polars"):
                        series = df.select(pl.col(col).drop_nulls().unique()).to_series()
                        values = series.to_list()
                    elif pd is not None:
                        values = df[col].dropna().unique().tolist()
                    else:
                        values = []
                except Exception:
                    values = []
                values = [str(v) for v in values if v is not None]
                return sorted(values)

            _design_vals = _unique_values(df_active, _design_col)
            _treatment_vals = _unique_values(df_active, _treatment_col)
            _max_len = max(len(_design_vals), len(_treatment_vals), 1)
            for _i in range(_max_len):
                design_treatment_rows.append(
                    {
                        "Design IDs": _design_vals[_i] if _i < len(_design_vals) else None,
                        "Treatments": _treatment_vals[_i] if _i < len(_treatment_vals) else None,
                    }
                )
    return design_treatment_rows, design_treatment_note

@app.cell
def _(
    artifact_labels,
    artifact_note,
    design_treatment_note,
    design_treatment_rows,
    exp_dir,
    exp_meta,
    mo,
    outputs_dir,
    plots_dir,
    exports_dir,
):
    _exp_id = exp_meta.get("id") or exp_dir.name
    _exp_title = exp_meta.get("title") or _exp_id
    _artifacts_dir = outputs_dir / "artifacts"
    _manifests_dir = outputs_dir / "manifests"
    _manifest_paths = {
        "manifest.json": _manifests_dir / "manifest.json",
        "plots_manifest.json": _manifests_dir / "plots_manifest.json",
        "exports_manifest.json": _manifests_dir / "exports_manifest.json",
    }
    _manifest_lines = []
    for _name, _path in _manifest_paths.items():
        _status = "" if _path.exists() else " (missing)"
        _manifest_lines.append(f"  - {_name}: `{_path}`{_status}")
    _manifest_block = "\\n".join(_manifest_lines) if _manifest_lines else "  - (none found)"

    _summary_md = (
        "**Run summary**\\n"
        f"- Outputs: `{outputs_dir}`\\n"
        f"- Plots: `{plots_dir}`\\n"
        f"- Exports: `{exports_dir}`\\n"
        f"- Manifests:\\n{_manifest_block}\\n\\n"
        "**Artifacts**\\n"
        f"- Artifacts dir: `{_artifacts_dir}`\\n"
        f"- Artifact datasets: {len(artifact_labels)}"
    )
    if artifact_note:
        _summary_md += f"\\n- Note: {artifact_note}"
    _summary_panel = mo.md(_summary_md)

    if design_treatment_rows:
        _design_table = mo.ui.table(design_treatment_rows)
    else:
        _design_table = mo.md(design_treatment_note or "No design/treatment summary available.")
    mo.vstack(
        [
            mo.md(f"# {_exp_title}\\n**Experiment id:** `{_exp_id}`"),
            _summary_panel,
            mo.md("**Design IDs + treatments**"),
            _design_table,
        ]
    )

@app.cell
def _(
    artifact_dropdown,
    artifact_note,
    artifact_path,
    df_backend,
    df_cols,
    df_error,
    df_rows,
    mo,
    selected_label,
):
    _elements = [mo.md("## Dataset selection")]
    if artifact_dropdown is None:
        _elements.append(mo.md(artifact_note or "No datasets found."))
    else:
        _elements.append(artifact_dropdown)
        _status_rows = []
        if selected_label:
            _status_rows.append({"Field": "Selected dataset", "Value": selected_label})
        if artifact_path:
            _status_rows.append({"Field": "Parquet path", "Value": str(artifact_path)})
        if df_backend:
            _status_rows.append({"Field": "Backend", "Value": df_backend})
        if df_rows is not None and df_cols is not None and df_backend:
            _status_rows.append({"Field": "Rows", "Value": df_rows})
            _status_rows.append({"Field": "Columns", "Value": df_cols})
        if _status_rows:
            _elements.append(mo.ui.table(_status_rows))
        if df_error:
            _elements.append(mo.md(f"**Load error:** `{df_error}`"))
    mo.vstack(_elements)

@app.cell
def _(df_active, df_error, mo, selected_label):
    if df_error:
        mo.stop(True, mo.md(f"Failed to load `{selected_label}`: {df_error}"))
    if df_active is None:
        mo.stop(True, mo.md("Select a dataset to explore."))
    df_ready = True
    return df_ready

@app.cell
def _(df_active, df_ready, mo, pd, pl):
    _columns = list(df_active.columns) if hasattr(df_active, "columns") else []
    _elements = [mo.md("## Dataset table explorer")]
    df_preview = df_active
    if len(_columns) > 40:
        _display_cols = _columns[:40]
        if pl is not None and df_active.__class__.__module__.startswith("polars"):
            df_preview = df_active.select(_display_cols)
        elif pd is not None:
            df_preview = df_active[_display_cols]
        _elements.append(mo.md(f"Showing first 40 columns of {len(_columns)}."))
    _elements.append(mo.ui.table(df_preview, page_size=10))
    mo.vstack(_elements)

@app.cell
def _(data_groupings, df_active, df_ready, pd, pl):
    df_eda = df_active
    grouping_columns = []
    if data_groupings:
        _columns = list(df_active.columns) if hasattr(df_active, "columns") else []

        def _iter_group_items(raw):
            if isinstance(raw, dict):
                return list(raw.items())
            if isinstance(raw, list):
                items = []
                for entry in raw:
                    if isinstance(entry, dict):
                        items.extend(entry.items())
                return items
            return []

        def _coerce_values(values):
            if values is None:
                return []
            if isinstance(values, (list, tuple, set)):
                return list(values)
            return [values]

        _pd_df = None
        if pd is not None and isinstance(df_active, pd.DataFrame):
            _pd_df = df_active.copy()
            df_eda = _pd_df

        for _col, _sets in data_groupings.items():
            if _col not in _columns or not isinstance(_sets, dict):
                continue
            for _set_name, _raw in _sets.items():
                _mapping = {}
                for _label, _values in _iter_group_items(_raw):
                    for _val in _coerce_values(_values):
                        _mapping[_val] = str(_label)
                if not _mapping:
                    continue
                _new_col = f"{_col}__{_set_name}"
                grouping_columns.append(_new_col)
                if pl is not None and df_active.__class__.__module__.startswith("polars"):
                    df_eda = df_eda.with_columns(
                        pl.col(_col)
                        .map_elements(lambda v, m=_mapping: m.get(v, "Other"), return_dtype=pl.Utf8)
                        .alias(_new_col)
                    )
                elif _pd_df is not None:
                    _pd_df[_new_col] = _pd_df[_col].map(_mapping).fillna("Other")
    return df_eda, grouping_columns

@app.cell
def _(df_eda, grouping_columns, pd, pl):
    all_columns = list(df_eda.columns) if hasattr(df_eda, "columns") else []
    numeric_cols = []
    categorical_cols = []
    if pl is not None and df_eda.__class__.__module__.startswith("polars"):
        for _name, _dtype in zip(df_eda.columns, df_eda.dtypes):
            _dtype_name = str(_dtype).lower()
            if any(token in _dtype_name for token in ("int", "float", "decimal", "double")):
                numeric_cols.append(_name)
            elif (
                _dtype_name in {"bool", "boolean", "utf8", "string", "categorical", "enum"}
                or "date" in _dtype_name
                or "time" in _dtype_name
            ):
                categorical_cols.append(_name)
    elif pd is not None:
        for _name, _dtype in df_eda.dtypes.items():
            if pd.api.types.is_numeric_dtype(_dtype):
                numeric_cols.append(_name)
            elif (
                pd.api.types.is_bool_dtype(_dtype)
                or pd.api.types.is_string_dtype(_dtype)
                or pd.api.types.is_categorical_dtype(_dtype)
                or pd.api.types.is_object_dtype(_dtype)
            ):
                categorical_cols.append(_name)
    if "channel" in all_columns and "channel" not in categorical_cols:
        categorical_cols.append("channel")
    for _col in grouping_columns:
        if _col not in categorical_cols:
            categorical_cols.append(_col)

    channel_values = []
    if "channel" in all_columns:
        try:
            if pl is not None and df_eda.__class__.__module__.startswith("polars"):
                series = df_eda.select(pl.col("channel").drop_nulls().unique()).to_series()
                channel_values = series.to_list()
            elif pd is not None:
                channel_values = df_eda["channel"].dropna().unique().tolist()
        except Exception:
            channel_values = []
    channel_values = sorted([str(v) for v in channel_values if v is not None])

    return all_columns, numeric_cols, categorical_cols, channel_values

@app.cell
def _(all_columns, categorical_cols, channel_values, mo, numeric_cols):
    if not numeric_cols:
        _eda_panel = mo.md("## Ad-hoc EDA\\nNo numeric columns available for plotting.")
        plot_type = None
        x_dropdown = None
        y_dropdown = None
        hue_dropdown = None
        groupby_dropdown = None
        agg_dropdown = None
        bins_slider = None
        channel_dropdown = None
        subplot_dropdown = None
    else:
        plot_type = mo.ui.dropdown(
            options=["scatter", "line", "histogram"],
            value="scatter",
            label="Plot type",
            full_width=True,
        )
        _x_default = "time" if "time" in numeric_cols else numeric_cols[0]
        x_dropdown = mo.ui.dropdown(
            options=numeric_cols,
            value=_x_default,
            label="X",
            full_width=True,
        )
        _y_default = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        y_dropdown = mo.ui.dropdown(
            options=numeric_cols,
            value=_y_default,
            label="Y",
            full_width=True,
        )
        channel_dropdown = None
        if channel_values:
            channel_dropdown = mo.ui.dropdown(
                options=channel_values,
                value=channel_values[0],
                label="Measurement / Channel",
                full_width=True,
            )
        _cat_options = ["(none)"] + categorical_cols
        hue_dropdown = mo.ui.dropdown(
            options=_cat_options,
            value="(none)",
            label="Hue",
            full_width=True,
        )
        groupby_dropdown = mo.ui.dropdown(
            options=_cat_options,
            value="(none)",
            label="Group by",
            full_width=True,
        )
        subplot_dropdown = mo.ui.dropdown(
            options=["(none)"] + all_columns,
            value="(none)",
            label="Subplot by",
            full_width=True,
        )
        agg_dropdown = mo.ui.dropdown(
            options=["mean", "median"],
            value="mean",
            label="Aggregation",
            full_width=True,
        )
        bins_slider = mo.ui.slider(5, 100, value=30, label="Bins", full_width=True, step=1)
        _controls = [
            mo.md("## Ad-hoc EDA"),
            mo.hstack([plot_type, x_dropdown, y_dropdown]),
            mo.hstack([hue_dropdown, groupby_dropdown, agg_dropdown]),
            mo.hstack([subplot_dropdown, channel_dropdown]) if channel_dropdown else mo.hstack([subplot_dropdown]),
            bins_slider,
        ]
        _eda_panel = mo.vstack(_controls)
    _eda_panel
    return (
        plot_type,
        x_dropdown,
        y_dropdown,
        hue_dropdown,
        groupby_dropdown,
        agg_dropdown,
        bins_slider,
        channel_dropdown,
        subplot_dropdown,
    )

@app.cell
def _():
    def style_chart(chart):
        return (
            chart.properties(background="white")
            .configure_view(stroke=None)
            .configure_axis(gridColor="#E5E5E5")
            .configure_legend(strokeColor=None, fillColor="white")
        )

    return style_chart

@app.cell
def _(
    agg_dropdown,
    alt,
    bins_slider,
    channel_dropdown,
    df_eda,
    df_ready,
    groupby_dropdown,
    hue_dropdown,
    mo,
    pd,
    pl,
    plot_type,
    subplot_dropdown,
    style_chart,
    x_dropdown,
    y_dropdown,
):
    if plot_type is None or x_dropdown is None:
        mo.stop(True, mo.md("No numeric columns available for plotting."))
    plot_kind = plot_type.value
    x_col = x_dropdown.value
    hue_col = hue_dropdown.value if hue_dropdown is not None else "(none)"
    groupby_col = groupby_dropdown.value if groupby_dropdown is not None else "(none)"
    agg_value = agg_dropdown.value if agg_dropdown is not None else "mean"
    bins = bins_slider.value if bins_slider is not None else 30
    subplot_col = subplot_dropdown.value if subplot_dropdown is not None else "(none)"
    channel_value = channel_dropdown.value if channel_dropdown is not None else None

    _columns = list(df_eda.columns) if hasattr(df_eda, "columns") else []
    y_col = y_dropdown.value if y_dropdown is not None else None
    use_channel = channel_value is not None and "channel" in _columns
    if use_channel and "value" in _columns:
        y_col = "value"
    if x_col not in _columns:
        mo.stop(True, mo.md(f"Selected x column `{x_col}` is not in the dataset."))
    if plot_kind != "histogram":
        if y_col is None or y_col not in _columns:
            mo.stop(True, mo.md("Select a valid y column for this plot type."))
    if hue_col != "(none)" and hue_col not in _columns:
        mo.stop(True, mo.md(f"Hue column `{hue_col}` is not in the dataset."))
    if groupby_col != "(none)" and groupby_col not in _columns:
        mo.stop(True, mo.md(f"Group-by column `{groupby_col}` is not in the dataset."))
    if subplot_col != "(none)" and subplot_col not in _columns:
        mo.stop(True, mo.md(f"Subplot column `{subplot_col}` is not in the dataset."))

    def _is_polars(df) -> bool:
        return pl is not None and df.__class__.__module__.startswith("polars")

    def _aggregate(df, group_cols):
        if _is_polars(df):
            agg_expr = pl.col(y_col).median() if agg_value == "median" else pl.col(y_col).mean()
            return df.group_by(group_cols).agg(agg_expr.alias(y_col))
        if pd is not None:
            if agg_value == "median":
                return df.groupby(group_cols, dropna=False)[y_col].median().reset_index()
            return df.groupby(group_cols, dropna=False)[y_col].mean().reset_index()
        return df

    color_field = None
    if groupby_col != "(none)":
        color_field = groupby_col
    elif hue_col != "(none)":
        color_field = hue_col

    df_plot = df_eda
    if use_channel:
        if _is_polars(df_plot):
            df_plot = df_plot.filter(pl.col("channel") == channel_value)
        elif pd is not None:
            df_plot = df_plot[df_plot["channel"] == channel_value]

    if plot_kind == "histogram":
        if _is_polars(df_plot):
            df_plot = df_plot.select([x_col]).to_pandas()
        elif pd is not None and not isinstance(df_plot, pd.DataFrame):
            try:
                df_plot = pd.DataFrame(df_plot)
            except Exception:
                pass
        chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X(f"{x_col}:Q", bin=alt.Bin(maxbins=bins)),
                y=alt.Y("count()", title="Count"),
            )
            .properties(height=320)
        )
    else:
        if plot_kind == "line":
            group_cols = [x_col]
            if color_field:
                group_cols.append(color_field)
            if subplot_col != "(none)" and subplot_col not in group_cols:
                group_cols.append(subplot_col)
            do_agg = groupby_col != "(none)"
            if not do_agg:
                try:
                    if _is_polars(df_plot):
                        _unique = df_plot.select(pl.col(x_col).n_unique()).item()
                        do_agg = _unique < df_plot.height
                    elif pd is not None:
                        do_agg = df_plot[x_col].nunique(dropna=False) < len(df_plot)
                except Exception:
                    do_agg = False
            if do_agg:
                df_plot = _aggregate(df_plot, group_cols)

        if _is_polars(df_plot):
            df_plot = df_plot.to_pandas()
        elif pd is not None and not isinstance(df_plot, pd.DataFrame):
            try:
                df_plot = pd.DataFrame(df_plot)
            except Exception:
                pass

        base = alt.Chart(df_plot).properties(height=320)
        if plot_kind == "line":
            base = base.mark_line()
        else:
            base = base.mark_point()

        enc = {
            "x": alt.X(f"{x_col}:Q", title=x_col),
            "y": alt.Y(f"{y_col}:Q", title=y_col),
        }
        if color_field:
            enc["color"] = alt.Color(f"{color_field}:N", title=color_field)
        chart = base.encode(**enc)

    if subplot_col != "(none)":
        chart = chart.facet(column=alt.Column(f"{subplot_col}:N", title=subplot_col), columns=3)
    chart = style_chart(chart)
    mo.ui.altair_chart(chart)

'''

EXPERIMENT_EDA_TEMPLATE_FOOTER = '''
if __name__ == "__main__":
    app.run()
'''

EXPERIMENT_EDA_BASIC_TEMPLATE = EXPERIMENT_EDA_BASE_TEMPLATE + EXPERIMENT_EDA_TEMPLATE_FOOTER
EXPERIMENT_EDA_MICROPLATE_TEMPLATE = EXPERIMENT_EDA_BASIC_TEMPLATE
EXPERIMENT_EDA_CYTOMETRY_TEMPLATE = EXPERIMENT_EDA_BASIC_TEMPLATE
EXPERIMENT_NOTEBOOK_PLOT_TEMPLATE = EXPERIMENT_EDA_BASIC_TEMPLATE

NOTEBOOK_PRESETS: dict[str, dict[str, str]] = {
    "notebook/basic": {
        "description": "Minimal artifact explorer with run summary, design/treatment table, and df.parquet preview.",
        "template": EXPERIMENT_EDA_BASIC_TEMPLATE,
    },
    "notebook/microplate": {
        "description": "Minimal artifact explorer (same scaffold as notebook/basic).",
        "template": EXPERIMENT_EDA_MICROPLATE_TEMPLATE,
    },
    "notebook/cytometry": {
        "description": "Minimal artifact explorer (same scaffold as notebook/basic).",
        "template": EXPERIMENT_EDA_CYTOMETRY_TEMPLATE,
    },
    "notebook/plots": {
        "description": "Minimal artifact explorer (same scaffold as notebook/basic).",
        "template": EXPERIMENT_NOTEBOOK_PLOT_TEMPLATE,
    },
}


def list_notebook_presets() -> list[tuple[str, str]]:
    return sorted((name, info["description"]) for name, info in NOTEBOOK_PRESETS.items())


def resolve_notebook_preset(name: str) -> str:
    if name not in NOTEBOOK_PRESETS:
        opts = ", ".join(sorted(NOTEBOOK_PRESETS))
        raise ConfigError(f"Unknown notebook preset {name!r}. Available presets: {opts}")
    return NOTEBOOK_PRESETS[name]["template"]


def write_experiment_notebook(
    target: Path,
    *,
    preset: str = "notebook/basic",
    overwrite: bool = False,
    plot_specs: list[dict] | None = None,
) -> tuple[Path, bool]:
    if target.exists() and not overwrite:
        return target, False
    template = resolve_notebook_preset(preset)
    if preset == "notebook/plots" and "__PLOT_SPECS__" in template:
        payload = plot_specs or []
        template = template.replace("__PLOT_SPECS__", repr(payload))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    return target, True
