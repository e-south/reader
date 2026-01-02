import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import duckdb
    return mo, json, Path, duckdb

@app.cell
def _(Path, mo):
    def _find_experiment_root(start: Path) -> Path:
        for base in [start] + list(start.parents):
            if (base / "config.yaml").exists():
                return base
        raise RuntimeError(
            "No config.yaml found. Place this notebook under an experiment directory "
            "or set exp_dir manually."
        )

    exp_dir = _find_experiment_root(Path(__file__).resolve())
    outputs_dir = exp_dir / "outputs"
    manifest_path = outputs_dir / "manifest.json"
    plots_dir = outputs_dir / "plots"

    mo.md(
        f"""# Reader EDA

**Experiment:** `{exp_dir.name}`

Artifacts and plots are loaded from `{outputs_dir}`.
"""
    )
    return exp_dir, outputs_dir, manifest_path, plots_dir

@app.cell
def _(json, manifest_path, mo, outputs_dir, plots_dir):
    if not outputs_dir.exists():
        mo.stop(True, mo.md("No outputs/ directory found. Run `reader run` first."))
    if not manifest_path.exists():
        mo.stop(True, mo.md("No outputs/manifest.json found. Run `reader run` first."))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    labels = sorted(artifacts.keys())
    plot_files = sorted([p.name for p in plots_dir.glob("*") if p.is_file()]) if plots_dir.exists() else []
    report_manifest_path = outputs_dir / "report_manifest.json"
    report_entries = []
    if report_manifest_path.exists():
        report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
        report_entries = report_manifest.get("reports", []) or []
    if not labels:
        mo.stop(True, mo.md("No artifacts listed in manifest.json."))
    return manifest, artifacts, labels, plot_files, report_entries

@app.cell
def _(artifacts, mo):
    mo.md("## Available artifacts")
    artifact_rows = [
        {
            "label": label,
            "step_dir": entry.get("step_dir", ""),
            "file": entry.get("filename", ""),
            "contract": entry.get("contract", ""),
        }
        for label, entry in artifacts.items()
    ]
    mo.ui.table(artifact_rows)
    return artifact_rows

@app.cell
def _(mo, report_entries):
    mo.md("## Reports")
    if not report_entries:
        mo.md("No report_manifest.json entries yet.")
        return
    report_rows = [
        {
            "step_id": entry.get("step_id", ""),
            "plugin": entry.get("plugin", ""),
            "files": len(entry.get("files", []) or []),
        }
        for entry in report_entries
    ]
    mo.ui.table(report_rows)
    return report_rows

@app.cell
def _(labels, mo):
    default_label = labels[0] if labels else None
    artifact_select = mo.ui.dropdown(options=labels, value=default_label, label="Artifact")
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
    limit = int(preview_limit.value)
    query = f"SELECT * FROM read_parquet('{artifact_path.as_posix()}') LIMIT {limit}"
    try:
        preview_df = duckdb.query(query).df()
    except Exception as exc:
        mo.stop(True, mo.md(f"Preview query failed: `{exc}`"))
    preview_df

@app.cell
def _(mo, preview_df):
    import pandas as pd

    mo.md("## Quick plot")
    if preview_df is None or getattr(preview_df, "empty", False):
        mo.stop(True, mo.md("No preview data available."))
    numeric_cols = [c for c in preview_df.columns if pd.api.types.is_numeric_dtype(preview_df[c])]
    if not numeric_cols:
        mo.stop(True, mo.md("No numeric columns found for plotting."))
    default_x = "time" if "time" in numeric_cols else numeric_cols[0]
    default_y = numeric_cols[1] if len(numeric_cols) > 1 and default_x == numeric_cols[0] else numeric_cols[0]
    plot_type_default = "line" if default_x == "time" else "scatter"
    plot_type = mo.ui.dropdown(options=["line", "scatter", "hist"], value=plot_type_default, label="Plot type")
    x_col = mo.ui.dropdown(options=numeric_cols, value=default_x, label="x")
    y_col = mo.ui.dropdown(options=numeric_cols, value=default_y, label="y")
    cat_cols = [c for c in preview_df.columns if c not in numeric_cols][:30]
    hue_default = "treatment" if "treatment" in cat_cols else "(none)"
    hue_col = mo.ui.dropdown(options=["(none)"] + cat_cols, value=hue_default, label="hue")
    return plot_type, x_col, y_col, hue_col

@app.cell
def _(hue_col, mo, plot_type, preview_df, x_col, y_col):
    import matplotlib.pyplot as plt

    if preview_df is None or getattr(preview_df, "empty", False):
        mo.stop(True)
    x = x_col.value
    y = y_col.value
    hue = hue_col.value
    kind = plot_type.value

    fig, ax = plt.subplots()
    if kind == "hist":
        ax.hist(preview_df[x].dropna(), bins=30, alpha=0.8)
        ax.set_xlabel(x)
        ax.set_ylabel("count")
        ax.set_title(f"hist: {x}")
    else:
        if hue != "(none)" and hue in preview_df.columns:
            for _label, grp in preview_df.groupby(hue):
                if kind == "line":
                    ax.plot(grp[x], grp[y], label=str(_label))
                else:
                    ax.scatter(grp[x], grp[y], label=str(_label), s=12, alpha=0.8)
            ax.legend()
        else:
            if kind == "line":
                ax.plot(preview_df[x], preview_df[y], linewidth=1.2)
            else:
                ax.scatter(preview_df[x], preview_df[y], s=12, alpha=0.8)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{kind}: {y} vs {x}")
    ax

@app.cell
def _(artifact_path, contract, duckdb, mo, preview_df):
    mo.md("## Tidy explorer")
    is_tidy = str(contract).startswith("tidy")
    if not is_tidy:
        mo.stop(True, mo.md("Selected artifact is not a tidy table."))
    try:
        meta_df = duckdb.query(
            f\"\"\"
            SELECT
                min(time) AS time_min,
                max(time) AS time_max,
                count(*) AS rows
            FROM read_parquet('{artifact_path.as_posix()}')
            \"\"\"
        ).df()
        channel_df = duckdb.query(
            f\"\"\"
            SELECT DISTINCT channel
            FROM read_parquet('{artifact_path.as_posix()}')
            ORDER BY channel
            \"\"\"
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
    numeric_cols = [c for c in preview_df.columns if c not in {"time", "value"}]
    cat_cols = [c for c in preview_df.columns if c not in numeric_cols][:30]
    hue_default = "treatment" if "treatment" in cat_cols else "(none)"
    hue_select = mo.ui.dropdown(options=["(none)"] + cat_cols, value=hue_default, label="Group")
    max_rows = mo.ui.number(label="Max rows", value=5000, start=100, stop=200000, step=100)
    apply_filters = mo.ui.run_button(label="Apply filters")
    return apply_filters, channel_select, hue_select, max_rows, time_high, time_low

@app.cell
def _(apply_filters, artifact_path, channel_select, duckdb, hue_select, mo, max_rows, time_high, time_low):
    mo.stop(not apply_filters.value)
    channel = channel_select.value
    tmin = float(time_low.value)
    tmax = float(time_high.value)
    limit = int(max_rows.value)
    if tmin > tmax:
        mo.stop(True, mo.md("Time min must be <= time max."))
    sql = (
        "SELECT * FROM read_parquet('{path}') "
        "WHERE channel = '{channel}' AND time BETWEEN {tmin} AND {tmax} "
        "LIMIT {limit}"
    ).format(path=artifact_path.as_posix(), channel=str(channel), tmin=tmin, tmax=tmax, limit=limit)
    try:
        tidy_filtered = duckdb.query(sql).df()
    except Exception as exc:
        mo.stop(True, mo.md(f"Tidy query failed: `{exc}`"))
    tidy_filtered

@app.cell
def _(mo, tidy_filtered):
    mo.md("### Filtered rows")
    if tidy_filtered is None or getattr(tidy_filtered, "empty", False):
        mo.stop(True, mo.md("No rows after filters."))
    mo.ui.data_explorer(tidy_filtered)
    return

@app.cell
def _(hue_select, mo, tidy_filtered):
    import matplotlib.pyplot as plt

    mo.md("### Tidy time series (mean)")
    if tidy_filtered is None or getattr(tidy_filtered, "empty", False):
        mo.stop(True, mo.md("No rows after filters."))
    if "time" not in tidy_filtered.columns or "value" not in tidy_filtered.columns:
        mo.stop(True, mo.md("Filtered data is missing time/value columns."))
    hue = hue_select.value
    fig, ax = plt.subplots()
    if hue != "(none)" and hue in tidy_filtered.columns:
        grouped = (
            tidy_filtered.groupby([hue, "time"], dropna=False)["value"]
            .mean()
            .reset_index()
        )
        for _label, grp in grouped.groupby(hue):
            ax.plot(grp["time"], grp["value"], label=str(_label))
        ax.legend()
    else:
        grouped = tidy_filtered.groupby("time", dropna=False)["value"].mean().reset_index()
        ax.plot(grouped["time"], grouped["value"])
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax

@app.cell
def _(artifact_path, mo):
    default_sql = f"SELECT * FROM read_parquet('{artifact_path.as_posix()}') LIMIT 500"
    sql_input = mo.ui.text_area(value=default_sql, label="DuckDB SQL", full_width=True)
    run_sql = mo.ui.run_button(label="Run SQL")
    return default_sql, sql_input, run_sql

@app.cell
def _(duckdb, mo, run_sql, sql_input):
    mo.stop(not run_sql.value)
    sql = (sql_input.value or "").strip()
    if not sql:
        mo.stop(True, mo.md("SQL query is empty."))
    try:
        sql_df = duckdb.query(sql).df()
    except Exception as exc:
        mo.stop(True, mo.md(f"SQL error: `{exc}`"))
    sql_df

@app.cell
def _(Path, mo, plot_files, plots_dir, report_entries, outputs_dir):
    mo.md("## Plot outputs")
    report_files = []
    for entry in report_entries:
        for name in entry.get("files", []) or []:
            report_files.append(name)
    file_rows = []
    if report_files:
        for name in sorted(set(report_files)):
            path = Path(name)
            if not path.is_absolute():
                path = outputs_dir / path
            file_rows.append({"file": path.name, "path": str(path)})
    elif plot_files:
        file_rows = [{"file": name, "path": str(plots_dir / name)} for name in plot_files]
    else:
        mo.stop(True, mo.md("No plots found yet. Run `reader run` with reports enabled."))
    plot_rows = file_rows
    mo.ui.table(plot_rows)
    return plot_rows

if __name__ == "__main__":
    app.run()
