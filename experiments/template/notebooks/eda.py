import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

@app.cell
def _():
    import json
    from pathlib import Path

    import duckdb
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from reader.core.config_model import ReaderSpec

    altair_err = None
    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()
    except Exception as exc:
        alt = None
        altair_err = exc

    return mo, json, Path, alt, altair_err, duckdb, pd, plt, ReaderSpec

@app.cell
def _(Path, ReaderSpec, mo):
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

    def _resolve_outputs(spec) -> tuple[Path, Path, Path]:
        outputs_path = Path(spec.paths.outputs).resolve()
        plots_cfg = spec.paths.plots
        exports_cfg = spec.paths.exports
        plots_path = outputs_path if plots_cfg in ("", ".", "./") else outputs_path / str(plots_cfg)
        exports_path = outputs_path if exports_cfg in ("", ".", "./") else outputs_path / str(exports_cfg)
        return outputs_path, plots_path, exports_path

    exp_dir = _find_experiment_root(Path(__file__).resolve())
    spec = _load_spec(exp_dir)
    outputs_dir, plots_dir, exports_dir = _resolve_outputs(spec)
    exp_meta = {"name": spec.experiment.title or spec.experiment.id, "id": spec.experiment.id}
    ingest_steps_cfg = [
        s.model_dump(by_alias=True) for s in spec.pipeline.steps if str(s.uses).startswith("ingest/")
    ]
    from reader.core.specs import resolve_export_specs, resolve_plot_specs

    plot_steps_cfg = [
        {"id": s.id, "uses": s.uses, "reads": s.reads, "with": s.with_, "writes": s.writes}
        for s in resolve_plot_specs(spec)
    ]
    export_steps_cfg = [
        {"id": s.id, "uses": s.uses, "reads": s.reads, "with": s.with_, "writes": s.writes}
        for s in resolve_export_specs(spec)
    ]
    manifest_path = outputs_dir / "manifests" / "manifest.json"
    plots_manifest_path = outputs_dir / "manifests" / "plots_manifest.json"
    exports_manifest_path = outputs_dir / "manifests" / "exports_manifest.json"
    return (
        spec,
        exp_dir,
        exp_meta,
        ingest_steps_cfg,
        plot_steps_cfg,
        export_steps_cfg,
        outputs_dir,
        manifest_path,
        plots_dir,
        exports_dir,
        plots_manifest_path,
        exports_manifest_path,
    )

@app.cell
def _(exp_dir, exp_meta, mo):
    _exp_name = exp_meta.get("name") or exp_dir.name
    title = mo.ui.text(value=_exp_name, label="Title")
    design_keys = mo.ui.text(value="design_id", label="Design keys (comma-separated)")
    treatment_keys = mo.ui.text(value="treatment", label="Treatment keys (comma-separated)")
    notes = mo.ui.text_area(value="", label="Notes", full_width=True)
    return design_keys, notes, title, treatment_keys

@app.cell
def _(
    design_keys,
    exp_dir,
    exp_meta,
    ingest_steps_cfg,
    meta_summary_md,
    mo,
    notes,
    plot_steps_cfg,
    export_steps_cfg,
    title,
    treatment_keys,
):
    _notes_text = (notes.value or "").strip()
    _exp_name = exp_meta.get("name")
    _exp_id = exp_meta.get("id")
    _ingest_uses = sorted({str(step.get("uses", "")) for step in ingest_steps_cfg if isinstance(step, dict)})
    _ingest_text = ", ".join(_ingest_uses) if _ingest_uses else "—"
    _header_lines = [f"# {title.value or exp_dir.name}", ""]
    _header_lines.append(f"**Experiment directory:** `{exp_dir.name}`")
    if _exp_name:
        _header_lines.append(f"**Experiment name:** `{_exp_name}`")
    if _exp_id:
        _header_lines.append(f"**Experiment id:** `{_exp_id}`")
    _header_lines.append(f"**Design keys:** `{design_keys.value}`")
    _header_lines.append(f"**Treatment keys:** `{treatment_keys.value}`")
    _header_lines.append(f"**Ingest:** {_ingest_text}")
    if _notes_text:
        _header_lines.append(f"**Notes:** {_notes_text}")
    _header_md = mo.md("\n".join(_header_lines))
    mo.vstack([_header_md, meta_summary_md])
    return

@app.cell
def _(
    exports_manifest_path,
    json,
    manifest_path,
    mo,
    outputs_dir,
    plots_dir,
    exports_dir,
    plots_manifest_path,
):
    if not outputs_dir.exists():
        mo.stop(True, mo.md("No outputs/ directory found. Run `reader run` first."))
    if not manifest_path.exists():
        mo.stop(True, mo.md("No outputs/manifests/manifest.json found. Run `reader run` first."))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    labels = sorted(artifacts.keys())
    plot_files = sorted([p.name for p in plots_dir.glob("*") if p.is_file()]) if plots_dir.exists() else []
    export_files = sorted([p.name for p in exports_dir.glob("*") if p.is_file()]) if exports_dir.exists() else []
    deliverable_entries = []
    if plots_manifest_path.exists():
        plots_manifest = json.loads(plots_manifest_path.read_text(encoding="utf-8"))
        deliverable_entries.extend(plots_manifest.get("plots", []) or [])
    if exports_manifest_path.exists():
        exports_manifest = json.loads(exports_manifest_path.read_text(encoding="utf-8"))
        deliverable_entries.extend(exports_manifest.get("exports", []) or [])
    if not labels:
        mo.stop(True, mo.md("No artifacts listed in manifest.json."))
    return manifest, artifacts, labels, plot_files, export_files, deliverable_entries

@app.cell
def _(labels, mo):
    _preferred = ["final/df", "merged/df", "raw/df"]
    _default_label = None
    for _name in _preferred:
        if _name in labels:
            _default_label = _name
            break
    if _default_label is None and labels:
        _default_label = labels[0]
    meta_source = mo.ui.dropdown(
        options=labels,
        value=_default_label,
        label="Metadata source (artifact)",
        full_width=True,
    )
    meta_limit = mo.ui.number(label="Max unique values", value=12, start=5, stop=200, step=1)
    return meta_limit, meta_source

@app.cell
def _(
    artifacts,
    design_keys,
    duckdb,
    meta_limit,
    meta_source,
    mo,
    outputs_dir,
    treatment_keys,
):
    meta_summary_md = mo.md("")
    _label = meta_source.value
    if not _label:
        mo.stop(True, mo.md("## Metadata summary\nSelect a metadata source to summarize."))
    _entry = artifacts.get(_label)
    if not _entry:
        mo.stop(True, mo.md(f"## Metadata summary\nArtifact not found: `{_label}`"))
    _artifact_path = outputs_dir / "artifacts" / _entry.get("step_dir", "") / _entry.get("filename", "")
    if not _artifact_path.exists():
        mo.stop(True, mo.md(f"## Metadata summary\nArtifact file not found: `{_artifact_path}`"))

    def _split_keys(text: str) -> list[str]:
        return [part.strip() for part in (text or "").split(",") if part.strip()]

    def _qcol(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    try:
        _schema_df = duckdb.query(
            f"SELECT * FROM read_parquet('{_artifact_path.as_posix()}') LIMIT 0"
        ).df()
    except Exception as exc:
        mo.stop(True, mo.md(f"## Metadata summary\nSchema read failed: `{exc}`"))
    _columns = list(_schema_df.columns)
    _limit = int(meta_limit.value)

    def _summarize_column(col: str) -> str:
        if col not in _columns:
            return f"- **{col}**: *(missing)*"
        col_q = _qcol(col)
        try:
            count_df = duckdb.query(
                f"SELECT COUNT(DISTINCT {col_q}) AS n FROM read_parquet('{_artifact_path.as_posix()}')"
            ).df()
            n = int(count_df.loc[0, "n"]) if not count_df.empty else 0
            values_df = duckdb.query(
                f"""
                SELECT DISTINCT {col_q} AS value
                FROM read_parquet('{_artifact_path.as_posix()}')
                WHERE {col_q} IS NOT NULL
                LIMIT {_limit + 1}
                """
            ).df()
        except Exception as exc:
            return f"- **{col}**: *(query failed: {exc})*"
        values = [str(v) for v in values_df["value"].tolist()] if "value" in values_df.columns else []
        preview = ", ".join(f"`{v}`" for v in values[:_limit]) if values else "—"
        extra = f" (+{n - _limit} more)" if n > _limit else ""
        return f"- **{col}** (n={n}): {preview}{extra}"

    _keys = _split_keys(design_keys.value) + _split_keys(treatment_keys.value)
    _key_lines = [_summarize_column(k) for k in _keys] if _keys else ["- *(no keys specified)*"]

    _time_line = ""
    if "time" in _columns:
        try:
            _time_df = duckdb.query(
                f"""
                SELECT
                    min(time) AS time_min,
                    max(time) AS time_max,
                    COUNT(DISTINCT time) AS time_n
                FROM read_parquet('{_artifact_path.as_posix()}')
                """
            ).df()
            _time_min = _time_df.loc[0, "time_min"] if not _time_df.empty else None
            _time_max = _time_df.loc[0, "time_max"] if not _time_df.empty else None
            _time_n = int(_time_df.loc[0, "time_n"]) if not _time_df.empty else 0
            if _time_n <= 1:
                _time_line = f"- **Time:** single time point (`{_time_min}`)"
            else:
                _time_line = f"- **Time:** {_time_n} time points (`{_time_min}` → `{_time_max}`)"
        except Exception as exc:
            _time_line = f"- **Time:** *(summary failed: {exc})*"

    _header = [
        "## Metadata summary",
        f"**Source artifact:** `{_label}`",
    ]
    _detail_lines = []
    if _time_line:
        _detail_lines.append(_time_line)
    _detail_lines.append("**Keys:**")
    _detail_lines.extend(_key_lines)
    meta_summary_md = mo.md("\n".join(_header + [""] + _detail_lines))
    return meta_summary_md

@app.cell
def _(artifacts, mo):
    mo.md("## Available artifacts")
    artifact_rows = [
        {
            "label": _label,
            "step_dir": _entry.get("step_dir", ""),
            "file": _entry.get("filename", ""),
            "contract": _entry.get("contract", ""),
        }
        for _label, _entry in artifacts.items()
    ]
    mo.ui.table(artifact_rows)
    return artifact_rows

@app.cell
def _(deliverable_entries, mo):
    mo.md("## Plots/exports")
    if not deliverable_entries:
        mo.stop(True, mo.md("No plot/export entries yet."))
    deliverable_rows = [
        {
            "step_id": _entry.get("step_id", ""),
            "plugin": _entry.get("plugin", ""),
            "files": len(_entry.get("files", []) or []),
        }
        for _entry in deliverable_entries
    ]
    mo.ui.table(deliverable_rows)
    return deliverable_rows

@app.cell
def _(mo):
    import pkgutil

    import reader.plugins.plot as _plot_pkg

    available_plot_modules = sorted({_mod.name.split(".")[-1] for _mod in pkgutil.iter_modules(_plot_pkg.__path__)})
    return available_plot_modules

@app.cell
def _(available_plot_modules, mo, plot_steps_cfg):
    return

@app.cell
def _(labels, mo):
    _default_label = labels[0] if labels else None
    artifact_select = mo.ui.dropdown(options=labels, value=_default_label, label="Artifact")
    preview_limit = mo.ui.number(label="Preview rows", value=200, start=10, stop=5000, step=10)
    return artifact_select, preview_limit

@app.cell
def _(artifact_select, artifacts, mo, outputs_dir):
    label = artifact_select.value
    if label is None:
        mo.stop(True, mo.md("Select an artifact to preview."))
    entry = artifacts[label]
    artifact_path = outputs_dir / "artifacts" / entry["step_dir"] / entry["filename"]
    return label, entry, artifact_path

@app.cell
def _(artifact_path, entry, mo):
    contract = entry.get("contract", "")
    size_bytes = artifact_path.stat().st_size if artifact_path.exists() else 0
    size_mb = round(size_bytes / (1024 * 1024), 2)
    mo.md("## Selected artifact")
    mo.ui.table(
        [
            {
                "label": entry.get("label", ""),
                "contract": contract,
                "file": artifact_path.name,
                "size_mb": size_mb,
            }
        ]
    )
    return contract

@app.cell
def _(duckdb, mo, preview_limit, artifact_path):
    _limit = int(preview_limit.value)
    query = f"SELECT * FROM read_parquet('{artifact_path.as_posix()}') LIMIT {_limit}"
    try:
        preview_df = duckdb.query(query).df()
    except Exception as exc:
        mo.stop(True, mo.md(f"Preview query failed: `{exc}`"))
    return preview_df

@app.cell
def _(mo, pd, preview_df):
    mo.md("## Quick plot")
    if not isinstance(preview_df, pd.DataFrame):
        mo.stop(True, mo.md("No preview data available."))
    if preview_df.empty:
        mo.stop(True, mo.md("No preview data available."))
    _numeric_cols = [_c for _c in preview_df.columns if pd.api.types.is_numeric_dtype(preview_df[_c])]
    if not _numeric_cols:
        mo.stop(True, mo.md("No numeric columns found for plotting."))
    default_x = "time" if "time" in _numeric_cols else _numeric_cols[0]
    default_y = _numeric_cols[1] if len(_numeric_cols) > 1 and default_x == _numeric_cols[0] else _numeric_cols[0]
    plot_type_default = "line" if default_x == "time" else "scatter"
    plot_type = mo.ui.dropdown(options=["line", "scatter", "hist"], value=plot_type_default, label="Plot type")
    x_col = mo.ui.dropdown(options=_numeric_cols, value=default_x, label="x")
    y_col = mo.ui.dropdown(options=_numeric_cols, value=default_y, label="y")
    _cat_cols = [_c for _c in preview_df.columns if _c not in _numeric_cols][:30]
    _hue_default = "treatment" if "treatment" in _cat_cols else "(none)"
    hue_col = mo.ui.dropdown(options=["(none)"] + _cat_cols, value=_hue_default, label="hue")
    return plot_type, x_col, y_col, hue_col

@app.cell
def _(hue_col, mo, plot_type, plt, preview_df, x_col, y_col):
    if preview_df is None or getattr(preview_df, "empty", False):
        mo.stop(True)
    x = x_col.value
    y = y_col.value
    _hue = hue_col.value
    kind = plot_type.value

    _fig, _ax = plt.subplots()
    if kind == "hist":
        _ax.hist(preview_df[x].dropna(), bins=30, alpha=0.8)
        _ax.set_xlabel(x)
        _ax.set_ylabel("count")
        _ax.set_title(f"hist: {x}")
    else:
        if _hue != "(none)" and _hue in preview_df.columns:
            for _label, _grp in preview_df.groupby(_hue):
                if kind == "line":
                    _ax.plot(_grp[x], _grp[y], label=str(_label))
                else:
                    _ax.scatter(_grp[x], _grp[y], label=str(_label), s=12, alpha=0.8)
            _ax.legend()
        else:
            if kind == "line":
                _ax.plot(preview_df[x], preview_df[y], linewidth=1.2)
            else:
                _ax.scatter(preview_df[x], preview_df[y], s=12, alpha=0.8)
        _ax.set_xlabel(x)
        _ax.set_ylabel(y)
        _ax.set_title(f"{kind}: {y} vs {x}")
    return _ax

@app.cell
def _(mo, pd, preview_df):
    mo.md("## Quick plot (Altair)")
    if not isinstance(preview_df, pd.DataFrame):
        mo.stop(True, mo.md("No preview data available."))
    if preview_df.empty:
        mo.stop(True, mo.md("No preview data available."))
    _numeric_cols = [_c for _c in preview_df.columns if pd.api.types.is_numeric_dtype(preview_df[_c])]
    if not _numeric_cols:
        mo.stop(True, mo.md("No numeric columns found for plotting."))
    _default_x = "time" if "time" in _numeric_cols else _numeric_cols[0]
    _default_y = _numeric_cols[1] if len(_numeric_cols) > 1 and _default_x == _numeric_cols[0] else _numeric_cols[0]
    _plot_type_default = "line" if _default_x == "time" else "scatter"
    alt_plot_type = mo.ui.dropdown(options=["line", "scatter", "hist"], value=_plot_type_default, label="Plot type")
    alt_x_col = mo.ui.dropdown(options=_numeric_cols, value=_default_x, label="x")
    alt_y_col = mo.ui.dropdown(options=_numeric_cols, value=_default_y, label="y")
    _cat_cols = [_c for _c in preview_df.columns if _c not in _numeric_cols][:30]
    _hue_default = "treatment" if "treatment" in _cat_cols else "(none)"
    alt_hue_col = mo.ui.dropdown(options=["(none)"] + _cat_cols, value=_hue_default, label="hue")
    alt_agg = mo.ui.dropdown(options=["raw", "mean", "median"], value="mean", label="aggregate")
    return alt_agg, alt_hue_col, alt_plot_type, alt_x_col, alt_y_col

@app.cell
def _(alt, altair_err, alt_agg, alt_hue_col, alt_plot_type, alt_x_col, alt_y_col, mo, preview_df):
    mo.stop(altair_err is not None, mo.md("Altair is required here. Run `uv sync --locked --group notebooks` (or `uv add altair`)."))
    if preview_df is None or getattr(preview_df, "empty", False):
        mo.stop(True)
    _kind = alt_plot_type.value
    _x = alt_x_col.value
    _y = alt_y_col.value
    _hue = alt_hue_col.value
    _agg = alt_agg.value

    _df_plot = preview_df.copy()
    _color_enc = alt.Undefined
    if _hue != "(none)" and _hue in _df_plot.columns:
        _color_enc = alt.Color(f"{_hue}:N", title=_hue)

    if _kind == "hist":
        _chart = (
            alt.Chart(_df_plot)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X(f"{_x}:Q", bin=alt.Bin(maxbins=40), title=_x),
                y=alt.Y("count()", title="count"),
                color=_color_enc,
                tooltip=[alt.Tooltip(f"{_x}:Q", title=_x), alt.Tooltip("count()", title="count")],
            )
        )
    else:
        if _agg != "raw":
            _group_cols = [_x]
            if _hue != "(none)" and _hue in _df_plot.columns:
                _group_cols.append(_hue)
            _df_plot = (
                _df_plot.groupby(_group_cols, dropna=False)[_y]
                .agg(_agg)
                .reset_index()
                .rename(columns={_y: f"{_y}__{_agg}"})
            )
            _y_field = f"{_y}__{_agg}"
        else:
            _y_field = _y
        _mark = alt.MarkDef(type="line" if _kind == "line" else "circle", opacity=0.8)
        _chart = (
            alt.Chart(_df_plot)
            .mark(_mark)
            .encode(
                x=alt.X(f"{_x}:Q", title=_x),
                y=alt.Y(f"{_y_field}:Q", title=_y_field),
                color=_color_enc,
                tooltip=[c for c in [_x, _y_field, _hue] if c in _df_plot.columns],
            )
        )
    mo.ui.altair_chart(_chart)
    return

@app.cell
def _(artifact_path, contract, duckdb, mo, preview_df):
    mo.md("## Tidy explorer")
    is_tidy = str(contract).startswith("tidy")
    if not is_tidy:
        mo.stop(True, mo.md("Selected artifact is not a tidy table."))
    try:
        meta_df = duckdb.query(
            f"""
            SELECT
                min(time) AS time_min,
                max(time) AS time_max,
                count(*) AS rows
            FROM read_parquet('{artifact_path.as_posix()}')
            """
        ).df()
        channel_df = duckdb.query(
            f"""
            SELECT DISTINCT channel
            FROM read_parquet('{artifact_path.as_posix()}')
            ORDER BY channel
            """
        ).df()
    except Exception as exc:
        mo.stop(True, mo.md(f"Tidy summary failed: `{exc}`"))
    time_min = float(meta_df.loc[0, "time_min"]) if not meta_df.empty else 0.0
    time_max = float(meta_df.loc[0, "time_max"]) if not meta_df.empty else 0.0
    channels = channel_df["channel"].astype(str).tolist() if "channel" in channel_df.columns else []
    mo.ui.table(
        [
            {
                "rows": int(meta_df.loc[0, "rows"]) if not meta_df.empty else 0,
                "time_min": time_min,
                "time_max": time_max,
                "channels": len(channels),
            }
        ]
    )
    return channels, time_min, time_max

@app.cell
def _(channels, mo, preview_df, time_max, time_min):
    if not channels:
        mo.stop(True, mo.md("No channels found in tidy data."))
    channel_select = mo.ui.dropdown(options=channels, value=channels[0], label="Channel")
    time_low = mo.ui.number(label="Time min", value=time_min)
    time_high = mo.ui.number(label="Time max", value=time_max)
    _numeric_cols = [_c for _c in preview_df.columns if _c not in {"time", "value"}]
    _cat_cols = [_c for _c in preview_df.columns if _c not in _numeric_cols][:30]
    _hue_default = "treatment" if "treatment" in _cat_cols else "(none)"
    hue_select = mo.ui.dropdown(options=["(none)"] + _cat_cols, value=_hue_default, label="Group")
    max_rows = mo.ui.number(label="Max rows", value=5000, start=100, stop=200000, step=100)
    apply_filters = mo.ui.run_button(label="Apply filters")
    return apply_filters, channel_select, hue_select, max_rows, time_high, time_low

@app.cell
def _(apply_filters, artifact_path, channel_select, duckdb, hue_select, mo, max_rows, time_high, time_low):
    mo.stop(not apply_filters.value)
    channel = channel_select.value
    tmin = float(time_low.value)
    tmax = float(time_high.value)
    _limit = int(max_rows.value)
    if tmin > tmax:
        mo.stop(True, mo.md("Time min must be <= time max."))
    _sql = (
        f"SELECT * FROM read_parquet('{artifact_path.as_posix()}') "
        f"WHERE channel = '{channel}' AND time BETWEEN {tmin} AND {tmax} "
        f"LIMIT {_limit}"
    )
    try:
        tidy_filtered = duckdb.query(_sql).df()
    except Exception as exc:
        mo.stop(True, mo.md(f"Tidy query failed: `{exc}`"))
    return tidy_filtered

@app.cell
def _(mo, tidy_filtered):
    mo.md("### Filtered rows")
    if tidy_filtered is None or getattr(tidy_filtered, "empty", False):
        mo.stop(True, mo.md("No rows after filters."))
    mo.ui.data_explorer(tidy_filtered)
    return

@app.cell
def _(alt, altair_err, hue_select, mo, plt, tidy_filtered):
    mo.md("### Tidy time series (Altair mean)")
    mo.stop(altair_err is not None, mo.md("Altair is required here. Run `uv sync --locked --group notebooks` (or `uv add altair`)."))
    if tidy_filtered is None or getattr(tidy_filtered, "empty", False):
        mo.stop(True, mo.md("No rows after filters."))
    if "time" not in tidy_filtered.columns or "value" not in tidy_filtered.columns:
        mo.stop(True, mo.md("Filtered data is missing time/value columns."))

    _hue = hue_select.value
    if _hue != "(none)" and _hue in tidy_filtered.columns:
        _df_plot = (
            tidy_filtered.groupby([_hue, "time"], dropna=False)["value"]
            .mean()
            .reset_index()
        )
        _color_enc = alt.Color(f"{_hue}:N", title=_hue)
        _tooltip = ["time", "value", _hue]
    else:
        _df_plot = tidy_filtered.groupby("time", dropna=False)["value"].mean().reset_index()
        _color_enc = alt.value("#4C78A8")
        _tooltip = ["time", "value"]

    _chart = (
        alt.Chart(_df_plot)
        .mark_line(point=True, opacity=0.9)
        .encode(
            x=alt.X("time:Q", title="time"),
            y=alt.Y("value:Q", title="value"),
            color=_color_enc,
            tooltip=_tooltip,
        )
    )
    mo.ui.altair_chart(_chart)

@app.cell
def _(artifact_path, mo):
    default_sql = f"SELECT * FROM read_parquet('{artifact_path.as_posix()}') LIMIT 500"
    sql_input = mo.ui.text_area(value=default_sql, label="DuckDB SQL", full_width=True)
    run_sql = mo.ui.run_button(label="Run SQL")
    return default_sql, sql_input, run_sql

@app.cell
def _(duckdb, mo, run_sql, sql_input):
    mo.stop(not run_sql.value)
    _sql = (sql_input.value or "").strip()
    if not _sql:
        mo.stop(True, mo.md("SQL query is empty."))
    try:
        sql_df = duckdb.query(_sql).df()
    except Exception as exc:
        mo.stop(True, mo.md(f"SQL error: `{exc}`"))
    return sql_df

@app.cell
def _(Path, deliverable_entries, mo, plot_files, export_files, plots_dir, exports_dir, outputs_dir):
    mo.md("## Plot + export outputs")
    deliverable_files = []
    for _entry in deliverable_entries:
        for _name in _entry.get("files", []) or []:
            deliverable_files.append(_name)
    file_rows = []
    if deliverable_files:
        for _name in sorted(set(deliverable_files)):
            path = Path(_name)
            if not path.is_absolute():
                path = outputs_dir / path
            file_rows.append({"file": path.name, "path": str(path)})
    elif plot_files or export_files:
        file_rows = [{"file": _name, "path": str(plots_dir / _name)} for _name in plot_files]
        file_rows.extend([{"file": _name, "path": str(exports_dir / _name)} for _name in export_files])
    else:
        mo.md("No saved plot/export files yet. Use `reader plot` or `reader export` if you want files on disk.")
    plot_rows = file_rows
    if plot_rows:
        mo.ui.table(plot_rows)
    return plot_rows

if __name__ == "__main__":
    app.run()
