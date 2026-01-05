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
    from reader.core.config_model import ReaderSpec

    return (
        Path,
        json,
        alt,
        mo,
        pl,
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
    data_aliases = _spec.data.aliases or {}
    design_id_aliases = {}
    if isinstance(data_aliases, dict):
        raw_aliases = data_aliases.get("design_id", {})
        if isinstance(raw_aliases, dict):
            design_id_aliases = {str(k): str(v) for k, v in raw_aliases.items()}
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
        design_id_aliases,
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
def _(artifact_path, pl):
    df_active = None
    df_backend = None
    df_error = None
    _pl_error = None

    if artifact_path is not None:
        if pl is None:
            df_error = "Polars is required to read parquet. Install the notebooks group."
        else:
            try:
                df_active = pl.read_parquet(artifact_path)
                df_backend = "polars"
            except Exception as exc:
                _pl_error = str(exc)
        if df_active is None and df_error is None:
            _suffix = _pl_error or "unknown error"
            df_error = f"Failed to read parquet with polars ({_suffix})."
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
def _(df_active, pl):
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
                values = []
                try:
                    if pl is not None and df.__class__.__module__.startswith("polars"):
                        _series = df.select(pl.col(col).drop_nulls().unique()).to_series()
                        values = _series.to_list()
                except Exception:
                    values = []
                values = [str(_v) for _v in values if _v is not None]
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

    if design_treatment_rows:
        _design_table = mo.ui.table(design_treatment_rows)
    else:
        _design_table = mo.md(design_treatment_note or "No design/treatment summary available.")
    mo.vstack(
        [
            mo.md(f"# {_exp_title}\\n**Experiment id:** `{_exp_id}`"),
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
def _(df_active, df_ready, mo, pl):
    _columns = list(df_active.columns) if hasattr(df_active, "columns") else []
    _elements = [mo.md("## Dataset table explorer")]
    df_preview = df_active
    if len(_columns) > 40:
        _display_cols = _columns[:40]
        if pl is not None and df_active.__class__.__module__.startswith("polars"):
            df_preview = df_active.select(_display_cols)
        _elements.append(mo.md(f"Showing first 40 columns of {len(_columns)}."))
    _elements.append(mo.ui.table(df_preview, page_size=10))
    mo.vstack(_elements)

@app.cell
def _(data_groupings, df_active, df_ready, pl):
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
    return df_eda, grouping_columns

@app.cell
def _(df_eda, grouping_columns, pl):
    all_columns = list(df_eda.columns) if hasattr(df_eda, "columns") else []
    base_columns = [col for col in all_columns if col not in grouping_columns]
    numeric_cols = []
    categorical_cols = []
    temporal_cols = []
    if pl is not None and df_eda.__class__.__module__.startswith("polars"):
        for _name, _dtype in zip(df_eda.columns, df_eda.dtypes):
            _dtype_name = str(_dtype).lower()
            if any(token in _dtype_name for token in ("int", "float", "decimal", "double")):
                numeric_cols.append(_name)
            elif "date" in _dtype_name or "time" in _dtype_name:
                temporal_cols.append(_name)
            elif _dtype_name in {"bool", "boolean", "utf8", "string", "categorical", "enum"}:
                categorical_cols.append(_name)
    if "channel" in base_columns and "channel" not in categorical_cols:
        categorical_cols.append("channel")
    for _col in grouping_columns:
        if _col not in categorical_cols:
            categorical_cols.append(_col)

    return base_columns, all_columns, numeric_cols, categorical_cols, temporal_cols

@app.cell
def _(all_columns, categorical_cols, mo, numeric_cols, temporal_cols):
    if not all_columns:
        plot_type = None
        x_dropdown = None
        y_dropdown = None
        hue_dropdown = None
        facet_row_dropdown = None
        facet_col_dropdown = None
    else:
        plot_type = mo.ui.dropdown(
            options=["line", "scatter", "histogram"],
            value="line",
            label="Plot type",
            full_width=True,
        )
        if "time" in all_columns:
            _x_default = "time"
        elif numeric_cols:
            _x_default = numeric_cols[0]
        else:
            _x_default = all_columns[0]
        x_dropdown = mo.ui.dropdown(
            options=all_columns,
            value=_x_default,
            label="X",
            full_width=True,
        )
        if "value" in all_columns:
            _y_default = "value"
        elif len(numeric_cols) > 1:
            _y_default = numeric_cols[1]
        elif numeric_cols:
            _y_default = numeric_cols[0]
        else:
            _y_default = all_columns[0]
        y_dropdown = mo.ui.dropdown(
            options=all_columns,
            value=_y_default,
            label="Y",
            full_width=True,
        )
        _cat_options = ["(none)"] + categorical_cols
        hue_dropdown = mo.ui.dropdown(
            options=_cat_options,
            value="(none)",
            label="Hue",
            full_width=True,
        )
        _facet_candidates = []
        for _col in all_columns:
            if _col in categorical_cols or _col in temporal_cols:
                if _col not in _facet_candidates:
                    _facet_candidates.append(_col)
        _facet_options = ["(none)"] + _facet_candidates
        facet_row_dropdown = mo.ui.dropdown(
            options=_facet_options,
            value="(none)",
            label="Facet row",
            full_width=True,
        )
        facet_col_dropdown = mo.ui.dropdown(
            options=_facet_options,
            value="(none)",
            label="Facet col",
            full_width=True,
        )
    return plot_type, x_dropdown, y_dropdown, hue_dropdown, facet_row_dropdown, facet_col_dropdown

@app.cell
def _():
    def humanize_label(value):
        if value is None:
            return ""
        text = str(value)
        if "/" in text:
            return text
        if any(ch.isupper() for ch in text) and "_" not in text:
            return text
        if text.isupper():
            return text
        text = text.replace("__", " ").replace("_", " ")
        parts = []
        for part in text.split():
            if part.lower() == "id":
                parts.append("ID")
            else:
                parts.append(part.capitalize())
        return " ".join(parts)

    return humanize_label

@app.cell
def _(pl):
    def sorted_unique_values(df, col):
        if pl is None or not df.__class__.__module__.startswith("polars"):
            return []
        try:
            _series = df.select(pl.col(col).drop_nulls().unique()).to_series()
            values = _series.to_list()
        except Exception:
            values = []
        values = [str(_v) for _v in values if _v is not None]
        return sorted(values)

    return sorted_unique_values

@app.cell
def _(categorical_cols, df_eda, humanize_label, mo, sorted_unique_values, x_dropdown):
    if x_dropdown is None:
        x_filter_dropdown = None
    elif x_dropdown.value in categorical_cols:
        _values = sorted_unique_values(df_eda, x_dropdown.value)
        x_filter_dropdown = mo.ui.dropdown(
            options=["(all)"] + _values,
            value="(all)",
            label=f"Filter: {humanize_label(x_dropdown.value)}",
            full_width=True,
        )
    else:
        x_filter_dropdown = None
    return x_filter_dropdown

@app.cell
def _(categorical_cols, df_eda, humanize_label, mo, sorted_unique_values, y_dropdown):
    if y_dropdown is None:
        y_filter_dropdown = None
    elif y_dropdown.value in categorical_cols:
        _values = sorted_unique_values(df_eda, y_dropdown.value)
        y_filter_dropdown = mo.ui.dropdown(
            options=["(all)"] + _values,
            value="(all)",
            label=f"Filter: {humanize_label(y_dropdown.value)}",
            full_width=True,
        )
    else:
        y_filter_dropdown = None
    return y_filter_dropdown

@app.cell
def _(mo, plot_type):
    if plot_type is None:
        bins_input = None
    elif plot_type.value == "histogram":
        bins_input = mo.ui.number(value=30, label="Bins", full_width=True)
    else:
        bins_input = None
    return bins_input

@app.cell
def _(
    bins_input,
    facet_col_dropdown,
    facet_row_dropdown,
    hue_dropdown,
    mo,
    plot_type,
    x_dropdown,
    x_filter_dropdown,
    y_dropdown,
    y_filter_dropdown,
):
    if plot_type is None or x_dropdown is None:
        _eda_panel = mo.md("## Ad-hoc EDA\\nNo columns available for plotting.")
    else:
        x_controls = mo.vstack([x_dropdown, x_filter_dropdown]) if x_filter_dropdown else x_dropdown
        y_controls = mo.vstack([y_dropdown, y_filter_dropdown]) if y_filter_dropdown else y_dropdown
        _controls = [
            mo.md("## Ad-hoc EDA"),
            mo.hstack([plot_type, x_controls, y_controls]),
            mo.hstack([hue_dropdown, facet_row_dropdown, facet_col_dropdown]),
        ]
        if bins_input is not None:
            _controls.append(bins_input)
        _eda_panel = mo.vstack(_controls)
    _eda_panel
    return

@app.cell
def _():
    def style_chart(chart):
        return (
            chart.properties(background="white")
            .configure_view(stroke=None)
            .configure_axis(
                grid=True,
                gridColor="#E6E6E6",
                domain=True,
                domainColor="#444444",
                domainWidth=1,
                ticks=True,
                tickColor="#444444",
                labels=True,
                labelColor="#333333",
                titleColor="#333333",
            )
            .configure_legend(strokeColor=None, fillColor="white", labelColor="#333333", titleColor="#333333")
            .configure_header(labelColor="#333333", labelFontWeight="bold", titleColor="#333333")
        )

    return style_chart

@app.cell
def _(
    alt,
    categorical_cols,
    design_id_aliases,
    df_eda,
    df_ready,
    grouping_columns,
    facet_col_dropdown,
    facet_row_dropdown,
    hue_dropdown,
    humanize_label,
    mo,
    numeric_cols,
    pl,
    plot_type,
    style_chart,
    temporal_cols,
    bins_input,
    x_dropdown,
    x_filter_dropdown,
    y_dropdown,
    y_filter_dropdown,
):
    if plot_type is None or x_dropdown is None:
        mo.stop(True, mo.md("No columns available for plotting."))
    if pl is None or not df_eda.__class__.__module__.startswith("polars"):
        mo.stop(True, mo.md("Polars is required for plotting. Install the notebooks group."))

    plot_kind = plot_type.value
    x_choice = x_dropdown.value
    y_choice = y_dropdown.value if y_dropdown is not None else None
    hue_choice = hue_dropdown.value if hue_dropdown is not None else "(none)"
    facet_row_choice = facet_row_dropdown.value if facet_row_dropdown is not None else "(none)"
    facet_col_choice = facet_col_dropdown.value if facet_col_dropdown is not None else "(none)"
    x_filter_value = x_filter_dropdown.value if x_filter_dropdown is not None else "(all)"
    y_filter_value = y_filter_dropdown.value if y_filter_dropdown is not None else "(all)"
    bins = int(bins_input.value) if bins_input is not None else 30

    df_plot = df_eda
    _columns = list(df_plot.columns) if hasattr(df_plot, "columns") else []
    if x_choice not in _columns:
        mo.stop(True, mo.md(f"Selected x column `{x_choice}` is not in the dataset."))
    if plot_kind != "histogram" and (y_choice is None or y_choice not in _columns):
        mo.stop(True, mo.md("Select a valid y column for this plot type."))
    if hue_choice != "(none)" and hue_choice not in _columns:
        mo.stop(True, mo.md(f"Hue column `{hue_choice}` is not in the dataset."))
    if facet_row_choice != "(none)" and facet_row_choice not in _columns:
        mo.stop(True, mo.md(f"Facet row column `{facet_row_choice}` is not in the dataset."))
    if facet_col_choice != "(none)" and facet_col_choice not in _columns:
        mo.stop(True, mo.md(f"Facet col column `{facet_col_choice}` is not in the dataset."))

    x_col = x_choice
    y_col = y_choice
    x_title = humanize_label(x_col)
    y_title = humanize_label(y_col) if y_col is not None else ""

    channel_filter = None
    if x_choice == "channel" or y_choice == "channel":
        if "channel" not in _columns:
            mo.stop(True, mo.md("Column `channel` is required for channel plotting."))
        if x_choice == "channel" and x_filter_value != "(all)":
            channel_filter = x_filter_value
        if y_choice == "channel" and y_filter_value != "(all)":
            if channel_filter is not None and channel_filter != y_filter_value:
                mo.stop(True, mo.md("Select the same channel measurement for both axes."))
            channel_filter = channel_filter or y_filter_value
        if channel_filter is not None:
            if "value" not in _columns:
                mo.stop(True, mo.md("Column `value` is required when filtering channel measurements."))
            df_plot = df_plot.filter(pl.col("channel") == channel_filter)
        if x_choice == "channel" and x_filter_value != "(all)":
            x_col = "value"
            x_title = x_filter_value
        if y_choice == "channel" and y_filter_value != "(all)":
            y_col = "value"
            y_title = y_filter_value

    if x_choice in categorical_cols and x_choice != "channel" and x_filter_value != "(all)":
        df_plot = df_plot.filter(pl.col(x_choice) == x_filter_value)
    if y_choice in categorical_cols and y_choice != "channel" and y_filter_value != "(all)":
        df_plot = df_plot.filter(pl.col(y_choice) == y_filter_value)

    def _infer_type(col):
        if col in temporal_cols:
            return "T"
        if col in numeric_cols:
            return "Q"
        return "N"

    x_type = _infer_type(x_col)
    y_type = _infer_type(y_col) if y_col is not None else "N"

    if plot_kind == "histogram" and x_type != "Q":
        mo.stop(True, mo.md("Histogram requires a numeric X column."))
    if plot_kind == "line" and y_type != "Q":
        mo.stop(True, mo.md("Line plots require a numeric Y column."))

    design_id_display_col = None
    if design_id_aliases and "design_id" in _columns:
        design_id_display_col = "design_id__label"
        if design_id_display_col not in _columns:
            df_plot = df_plot.with_columns(
                pl.col("design_id")
                .map_elements(
                    lambda v, m=design_id_aliases: m.get(str(v), str(v)) if v is not None else None,
                    return_dtype=pl.Utf8,
                )
                .alias(design_id_display_col)
            )
            _columns.append(design_id_display_col)

    hue_field = hue_choice if hue_choice != "(none)" else None
    if hue_choice == "design_id" and design_id_display_col is not None:
        hue_field = design_id_display_col

    facet_row_field = facet_row_choice if facet_row_choice != "(none)" else None
    if facet_row_choice == "design_id" and design_id_display_col is not None:
        facet_row_field = design_id_display_col
    facet_col_field = facet_col_choice if facet_col_choice != "(none)" else None
    if facet_col_choice == "design_id" and design_id_display_col is not None:
        facet_col_field = design_id_display_col

    def _as_chart_data(df):
        try:
            return df.to_arrow()
        except Exception as exc:
            mo.stop(True, mo.md(f"Failed to prepare data for plotting: {exc}"))
        return df

    data = _as_chart_data(df_plot)
    base = alt.Chart(data).properties(height=320)

    axis_x = alt.Axis(title=humanize_label(x_title), domain=True, ticks=True, labels=True)
    axis_y = alt.Axis(title=humanize_label(y_title), domain=True, ticks=True, labels=True)

    tooltip_fields = [alt.Tooltip(f"{x_col}:{x_type}", title=humanize_label(x_title))]
    if plot_kind == "line" and y_col is not None:
        tooltip_fields.append(alt.Tooltip(f"mean({y_col}):Q", title=humanize_label(y_title)))
    elif plot_kind != "histogram" and y_col is not None:
        tooltip_fields.append(alt.Tooltip(f"{y_col}:{y_type}", title=humanize_label(y_title)))
    if hue_field is not None:
        tooltip_fields.append(
            alt.Tooltip(f"{hue_field}:N", title=humanize_label(hue_choice))
        )
    for _col in ("design_id", "treatment", "sample_id", "replicate", "position"):
        display_col = _col
        if _col == "design_id" and design_id_display_col is not None:
            display_col = design_id_display_col
        if display_col in _columns and display_col not in {x_col, y_col, hue_field, facet_row_field, facet_col_field}:
            tooltip_fields.append(
                alt.Tooltip(f"{display_col}:N", title=humanize_label(_col))
            )

    if plot_kind == "histogram":
        chart = (
            base.mark_bar()
            .encode(
                x=alt.X(f"{x_col}:Q", bin=alt.Bin(maxbins=bins), axis=axis_x),
                y=alt.Y("count()", title="Count"),
                tooltip=tooltip_fields + [alt.Tooltip("count()", title="Count")],
            )
        )
    else:
        color_kwargs = {}
        if hue_field is not None:
            color_kwargs["color"] = alt.Color(
                f"{hue_field}:N",
                title=humanize_label(hue_choice),
                scale=alt.Scale(scheme="tableau10"),
            )
        if plot_kind == "scatter":
            point_enc = {
                "x": alt.X(f"{x_col}:{x_type}", axis=axis_x),
                "y": alt.Y(f"{y_col}:{y_type}", axis=axis_y),
                "tooltip": tooltip_fields,
            }
            point_enc.update(color_kwargs)
            if hue_field is not None:
                point_enc["shape"] = alt.Shape(f"{hue_field}:N", legend=None)
            chart = base.mark_point(filled=True, stroke=None, strokeWidth=0, size=60).encode(
                **point_enc
            )
        else:
            band_enc = {
                "x": alt.X(f"{x_col}:{x_type}", axis=axis_x),
                "y": alt.Y(f"{y_col}:Q", axis=axis_y),
            }
            band_enc.update(color_kwargs)
            band = base.mark_errorband(extent="ci", opacity=0.2).encode(**band_enc)

            line_enc = {
                "x": alt.X(f"{x_col}:{x_type}", axis=axis_x),
                "y": alt.Y(f"mean({y_col}):Q", axis=axis_y),
                "tooltip": tooltip_fields,
            }
            line_enc.update(color_kwargs)
            line = base.mark_line().encode(**line_enc)

            point_enc = {
                "x": alt.X(f"{x_col}:{x_type}", axis=axis_x),
                "y": alt.Y(f"mean({y_col}):Q", axis=axis_y),
                "tooltip": tooltip_fields,
            }
            point_enc.update(color_kwargs)
            if hue_field is not None:
                point_enc["shape"] = alt.Shape(f"{hue_field}:N", legend=None)
            points = base.mark_point(filled=True, stroke=None, strokeWidth=0, size=60).encode(
                **point_enc
            )
            chart = alt.layer(band, line, points)

    design_labels = []
    if "design_id" in _columns:
        try:
            _series = df_plot.select(pl.col("design_id").drop_nulls().unique()).to_series()
            raw_vals = [str(_v) for _v in _series.to_list() if _v is not None]
            raw_vals = sorted(raw_vals)
            if design_id_aliases:
                design_labels = [design_id_aliases.get(_v, _v) for _v in raw_vals]
            else:
                design_labels = raw_vals
        except Exception:
            design_labels = []
    if design_labels:
        if len(design_labels) > 6:
            summary = ", ".join(design_labels[:5])
            summary = f"{summary}, +{len(design_labels) - 5} more"
        else:
            summary = ", ".join(design_labels)
        title_prefix = "Design IDs"
        facet_group_label = None
        if facet_row_choice in grouping_columns:
            facet_group_label = humanize_label(facet_row_choice)
        elif facet_col_choice in grouping_columns:
            facet_group_label = humanize_label(facet_col_choice)
        if facet_group_label:
            title_prefix = f"Design IDs ({facet_group_label})"
        title = f"{title_prefix}: {summary}"
    else:
        title = "Design IDs"

    def _facet_type(col_name):
        return "T" if col_name in temporal_cols else "N"

    if facet_row_field is not None or facet_col_field is not None:
        if facet_row_field is not None and facet_col_field is not None:
            chart = chart.facet(
                row=alt.Row(
                    f"{facet_row_field}:{_facet_type(facet_row_choice)}",
                    title=humanize_label(facet_row_choice),
                    header=alt.Header(labelFontWeight="bold"),
                ),
                column=alt.Column(
                    f"{facet_col_field}:{_facet_type(facet_col_choice)}",
                    title=humanize_label(facet_col_choice),
                    header=alt.Header(labelFontWeight="bold"),
                ),
            )
        else:
            _facet_field = facet_row_field or facet_col_field
            _facet_choice = facet_row_choice if facet_row_field is not None else facet_col_choice
            chart = chart.facet(
                column=alt.Column(
                    f"{_facet_field}:{_facet_type(_facet_choice)}",
                    title=humanize_label(_facet_choice),
                    header=alt.Header(labelFontWeight="bold"),
                ),
                columns=5,
            )
        chart = chart.resolve_scale(x="shared", y="shared").resolve_axis(x="independent", y="independent")

    chart = chart.properties(title=title)
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
        "description": "Minimal artifact explorer with design/treatment table and df.parquet preview.",
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
