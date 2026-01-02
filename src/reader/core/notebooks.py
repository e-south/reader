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

EXPERIMENT_EDA_BASIC_TEMPLATE = '''import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, json, Path, duckdb, pd, plt

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
    deliverables_manifest_path = outputs_dir / "deliverables_manifest.json"
    return exp_dir, outputs_dir, manifest_path, plots_dir, deliverables_manifest_path

@app.cell
def _(exp_dir, mo):
    title = mo.ui.text(value=exp_dir.name, label="Title")
    design_keys = mo.ui.text(value="design_id", label="Design keys (comma-separated)")
    treatment_keys = mo.ui.text(value="treatment", label="Treatment keys (comma-separated)")
    notes = mo.ui.text_area(value="", label="Notes", full_width=True)
    return design_keys, notes, title, treatment_keys

@app.cell
def _(design_keys, exp_dir, mo, notes, title, treatment_keys):
    notes_text = (notes.value or "").strip()
    notes_block = f"\\n\\n**Notes:** {notes_text}" if notes_text else ""
    mo.md(
        f"""# Reader EDA

**Experiment:** `{exp_dir.name}`
**Design keys:** `{design_keys.value}`
**Treatment keys:** `{treatment_keys.value}`{notes_block}
"""
    )
    return

@app.cell
def _(deliverables_manifest_path, json, manifest_path, mo, outputs_dir, plots_dir):
    if not outputs_dir.exists():
        mo.stop(True, mo.md("No outputs/ directory found. Run `reader run` first."))
    if not manifest_path.exists():
        mo.stop(True, mo.md("No outputs/manifest.json found. Run `reader run` first."))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    labels = sorted(artifacts.keys())
    plot_files = sorted([p.name for p in plots_dir.glob("*") if p.is_file()]) if plots_dir.exists() else []
    deliverable_entries = []
    if deliverables_manifest_path.exists():
        deliverables_manifest = json.loads(deliverables_manifest_path.read_text(encoding="utf-8"))
        deliverable_entries = deliverables_manifest.get("deliverables", []) or []
    if not labels:
        mo.stop(True, mo.md("No artifacts listed in manifest.json."))
    return manifest, artifacts, labels, plot_files, deliverable_entries

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
def _(deliverable_entries, mo):
    mo.md("## Deliverables")
    if not deliverable_entries:
        mo.md("No deliverables_manifest.json entries yet.")
        return
    deliverable_rows = [
        {
            "step_id": entry.get("step_id", ""),
            "plugin": entry.get("plugin", ""),
            "files": len(entry.get("files", []) or []),
        }
        for entry in deliverable_entries
    ]
    mo.ui.table(deliverable_rows)
    return deliverable_rows

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
def _(mo, pd, preview_df):
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
def _(hue_col, mo, plot_type, plt, preview_df, x_col, y_col):
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
def _(hue_select, mo, plt, tidy_filtered):
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
def _(Path, deliverable_entries, mo, plot_files, plots_dir, outputs_dir):
    mo.md("## Plot outputs")
    deliverable_files = []
    for entry in deliverable_entries:
        for name in entry.get("files", []) or []:
            deliverable_files.append(name)
    file_rows = []
    if deliverable_files:
        for name in sorted(set(deliverable_files)):
            path = Path(name)
            if not path.is_absolute():
                path = outputs_dir / path
            file_rows.append({"file": path.name, "path": str(path)})
    elif plot_files:
        file_rows = [{"file": name, "path": str(plots_dir / name)} for name in plot_files]
    else:
        mo.stop(True, mo.md("No plots found yet. Run `reader run` with deliverables enabled."))
    plot_rows = file_rows
    mo.ui.table(plot_rows)
    return plot_rows

if __name__ == "__main__":
    app.run()
'''

EXPERIMENT_EDA_MICROPLATE_TEMPLATE = '''import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import duckdb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from reader.lib.microplates.base import nearest_time_per_key
    return mo, json, Path, duckdb, pd, np, plt, nearest_time_per_key

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
    deliverables_manifest_path = outputs_dir / "deliverables_manifest.json"
    return exp_dir, outputs_dir, manifest_path, plots_dir, deliverables_manifest_path

@app.cell
def _(exp_dir, mo):
    title = mo.ui.text(value=exp_dir.name, label="Title")
    design_keys = mo.ui.text(value="design_id", label="Design keys (comma-separated)")
    treatment_keys = mo.ui.text(value="treatment", label="Treatment keys (comma-separated)")
    notes = mo.ui.text_area(value="", label="Notes", full_width=True)
    return design_keys, notes, title, treatment_keys

@app.cell
def _(design_keys, exp_dir, mo, notes, title, treatment_keys):
    notes_text = (notes.value or "").strip()
    notes_block = f"\\n\\n**Notes:** {notes_text}" if notes_text else ""
    mo.md(
        f"""# Reader EDA (Microplate)

**Experiment:** `{exp_dir.name}`
**Design keys:** `{design_keys.value}`
**Treatment keys:** `{treatment_keys.value}`{notes_block}
"""
    )
    return

@app.cell
def _(deliverables_manifest_path, json, manifest_path, mo, outputs_dir, plots_dir):
    if not outputs_dir.exists():
        mo.stop(True, mo.md("No outputs/ directory found. Run `reader run` first."))
    if not manifest_path.exists():
        mo.stop(True, mo.md("No outputs/manifest.json found. Run `reader run` first."))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    labels = sorted(artifacts.keys())
    plot_files = sorted([p.name for p in plots_dir.glob("*") if p.is_file()]) if plots_dir.exists() else []
    deliverable_entries = []
    if deliverables_manifest_path.exists():
        deliverables_manifest = json.loads(deliverables_manifest_path.read_text(encoding="utf-8"))
        deliverable_entries = deliverables_manifest.get("deliverables", []) or []
    if not labels:
        mo.stop(True, mo.md("No artifacts listed in manifest.json."))
    return manifest, artifacts, labels, plot_files, deliverable_entries

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
def _(deliverable_entries, mo):
    mo.md("## Deliverables")
    if not deliverable_entries:
        mo.md("No deliverables_manifest.json entries yet.")
        return
    deliverable_rows = [
        {
            "step_id": entry.get("step_id", ""),
            "plugin": entry.get("plugin", ""),
            "files": len(entry.get("files", []) or []),
        }
        for entry in deliverable_entries
    ]
    mo.ui.table(deliverable_rows)
    return deliverable_rows

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
def _(mo, pd, preview_df):
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
def _(hue_col, mo, plot_type, plt, preview_df, x_col, y_col):
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
def _(hue_select, mo, plt, tidy_filtered):
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
def _(mo, tidy_filtered):
    mo.md("### Snapshot at time")
    if tidy_filtered is None or getattr(tidy_filtered, "empty", False):
        mo.stop(True, mo.md("No rows after filters."))
    if "time" not in tidy_filtered.columns or "value" not in tidy_filtered.columns:
        mo.stop(True, mo.md("Filtered data is missing time/value columns."))
    cat_cols = [c for c in tidy_filtered.columns if c not in {"time", "value"}]
    if not cat_cols:
        mo.stop(True, mo.md("No categorical columns available for grouping."))
    default_group = "design_id" if "design_id" in cat_cols else cat_cols[0]
    default_hue = "treatment" if "treatment" in cat_cols else "(none)"
    snap_group = mo.ui.dropdown(options=cat_cols, value=default_group, label="Group")
    snap_hue = mo.ui.dropdown(options=["(none)"] + cat_cols, value=default_hue, label="Hue")
    snap_time = mo.ui.number(label="Target time", value=float(tidy_filtered["time"].median()))
    snap_tol = mo.ui.number(label="Time tolerance", value=0.5)
    return snap_group, snap_hue, snap_time, snap_tol

@app.cell
def _(mo, nearest_time_per_key, np, plt, snap_group, snap_hue, snap_time, snap_tol, tidy_filtered):
    if tidy_filtered is None or getattr(tidy_filtered, "empty", False):
        mo.stop(True)
    group_key = snap_group.value
    hue_key = snap_hue.value
    keys = [group_key] if group_key else []
    if hue_key != "(none)":
        keys.append(hue_key)
    if not keys:
        mo.stop(True, mo.md("Select a group or hue column."))
    snap_df = nearest_time_per_key(
        tidy_filtered,
        target_time=float(snap_time.value),
        tol=float(snap_tol.value),
        keys=keys,
        time_col="time",
    )
    if snap_df.empty:
        mo.stop(True, mo.md("No rows within the requested time tolerance."))
    summary = snap_df.groupby(keys, dropna=False)["value"].mean().reset_index()
    fig, ax = plt.subplots()
    if hue_key != "(none)" and hue_key in summary.columns:
        groups = summary[group_key].astype(str).unique().tolist()
        hues = summary[hue_key].astype(str).unique().tolist()
        x = np.arange(len(groups))
        width = 0.8 / max(1, len(hues))
        for i, h in enumerate(hues):
            vals = []
            for g in groups:
                val = summary.loc[
                    (summary[group_key].astype(str) == str(g)) & (summary[hue_key].astype(str) == str(h)),
                    "value",
                ]
                vals.append(float(val.mean()) if not val.empty else 0.0)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=str(h))
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.legend()
    else:
        groups = summary[group_key].astype(str).unique().tolist()
        vals = [float(summary.loc[summary[group_key].astype(str) == str(g), "value"].mean()) for g in groups]
        ax.bar(groups, vals)
        ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel(group_key)
    ax.set_ylabel("value")
    ax.set_title(f"snapshot at ~{float(snap_time.value):.3g}")
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
def _(Path, deliverable_entries, mo, plot_files, plots_dir, outputs_dir):
    mo.md("## Plot outputs")
    deliverable_files = []
    for entry in deliverable_entries:
        for name in entry.get("files", []) or []:
            deliverable_files.append(name)
    file_rows = []
    if deliverable_files:
        for name in sorted(set(deliverable_files)):
            path = Path(name)
            if not path.is_absolute():
                path = outputs_dir / path
            file_rows.append({"file": path.name, "path": str(path)})
    elif plot_files:
        file_rows = [{"file": name, "path": str(plots_dir / name)} for name in plot_files]
    else:
        mo.stop(True, mo.md("No plots found yet. Run `reader run` with deliverables enabled."))
    plot_rows = file_rows
    mo.ui.table(plot_rows)
    return plot_rows

if __name__ == "__main__":
    app.run()
'''

EXPERIMENT_EDA_CYTOMETRY_TEMPLATE = '''import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, json, Path, duckdb, pd, plt

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
    deliverables_manifest_path = outputs_dir / "deliverables_manifest.json"
    return exp_dir, outputs_dir, manifest_path, plots_dir, deliverables_manifest_path

@app.cell
def _(exp_dir, mo):
    title = mo.ui.text(value=exp_dir.name, label="Title")
    design_keys = mo.ui.text(value="sample_id", label="Design keys (comma-separated)")
    treatment_keys = mo.ui.text(value="treatment", label="Treatment keys (comma-separated)")
    notes = mo.ui.text_area(value="", label="Notes", full_width=True)
    return design_keys, notes, title, treatment_keys

@app.cell
def _(design_keys, exp_dir, mo, notes, title, treatment_keys):
    notes_text = (notes.value or "").strip()
    notes_block = f"\\n\\n**Notes:** {notes_text}" if notes_text else ""
    mo.md(
        f"""# Reader EDA (Cytometry)

**Experiment:** `{exp_dir.name}`
**Design keys:** `{design_keys.value}`
**Treatment keys:** `{treatment_keys.value}`{notes_block}
"""
    )
    return

@app.cell
def _(deliverables_manifest_path, json, manifest_path, mo, outputs_dir, plots_dir):
    if not outputs_dir.exists():
        mo.stop(True, mo.md("No outputs/ directory found. Run `reader run` first."))
    if not manifest_path.exists():
        mo.stop(True, mo.md("No outputs/manifest.json found. Run `reader run` first."))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    labels = sorted(artifacts.keys())
    plot_files = sorted([p.name for p in plots_dir.glob("*") if p.is_file()]) if plots_dir.exists() else []
    deliverable_entries = []
    if deliverables_manifest_path.exists():
        deliverables_manifest = json.loads(deliverables_manifest_path.read_text(encoding="utf-8"))
        deliverable_entries = deliverables_manifest.get("deliverables", []) or []
    if not labels:
        mo.stop(True, mo.md("No artifacts listed in manifest.json."))
    return manifest, artifacts, labels, plot_files, deliverable_entries

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
def _(deliverable_entries, mo):
    mo.md("## Deliverables")
    if not deliverable_entries:
        mo.md("No deliverables_manifest.json entries yet.")
        return
    deliverable_rows = [
        {
            "step_id": entry.get("step_id", ""),
            "plugin": entry.get("plugin", ""),
            "files": len(entry.get("files", []) or []),
        }
        for entry in deliverable_entries
    ]
    mo.ui.table(deliverable_rows)
    return deliverable_rows

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
def _(mo, pd, preview_df):
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
def _(hue_col, mo, plot_type, plt, preview_df, x_col, y_col):
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
def _(mo, pd, preview_df):
    mo.md("## Population distributions")
    if preview_df is None or getattr(preview_df, "empty", False):
        mo.stop(True, mo.md("No preview data available."))
    numeric_cols = [c for c in preview_df.columns if pd.api.types.is_numeric_dtype(preview_df[c])]
    if not numeric_cols:
        mo.stop(True, mo.md("No numeric columns found for distributions."))
    value_col = mo.ui.dropdown(options=numeric_cols, value=numeric_cols[0], label="Value column")
    cat_cols = [c for c in preview_df.columns if c not in numeric_cols][:30]
    group_default = "sample" if "sample" in cat_cols else "(none)"
    group_col = mo.ui.dropdown(options=["(none)"] + cat_cols, value=group_default, label="Group")
    bins = mo.ui.number(label="Bins", value=50, start=10, stop=200, step=5)
    density = mo.ui.checkbox(label="Normalize", value=True)
    return bins, density, group_col, value_col

@app.cell
def _(bins, density, group_col, mo, plt, preview_df, value_col):
    if preview_df is None or getattr(preview_df, "empty", False):
        mo.stop(True)
    col = value_col.value
    if col not in preview_df.columns:
        mo.stop(True, mo.md("Selected value column is missing."))
    fig, ax = plt.subplots()
    if group_col.value != "(none)" and group_col.value in preview_df.columns:
        for label, grp in preview_df.groupby(group_col.value):
            ax.hist(
                grp[col].dropna(),
                bins=int(bins.value),
                alpha=0.5,
                density=bool(density.value),
                label=str(label),
            )
        ax.legend()
    else:
        ax.hist(
            preview_df[col].dropna(),
            bins=int(bins.value),
            alpha=0.8,
            density=bool(density.value),
        )
    ax.set_xlabel(col)
    ax.set_ylabel("density" if density.value else "count")
    ax.set_title("population distribution")
    ax

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
def _(hue_select, mo, plt, tidy_filtered):
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
def _(Path, deliverable_entries, mo, plot_files, plots_dir, outputs_dir):
    mo.md("## Plot outputs")
    deliverable_files = []
    for entry in deliverable_entries:
        for name in entry.get("files", []) or []:
            deliverable_files.append(name)
    file_rows = []
    if deliverable_files:
        for name in sorted(set(deliverable_files)):
            path = Path(name)
            if not path.is_absolute():
                path = outputs_dir / path
            file_rows.append({"file": path.name, "path": str(path)})
    elif plot_files:
        file_rows = [{"file": name, "path": str(plots_dir / name)} for name in plot_files]
    else:
        mo.stop(True, mo.md("No plots found yet. Run `reader run` with deliverables enabled."))
    plot_rows = file_rows
    mo.ui.table(plot_rows)
    return plot_rows

if __name__ == "__main__":
    app.run()
'''

NOTEBOOK_PRESETS: dict[str, dict[str, str]] = {
    "eda/basic": {
        "description": "Generic EDA notebook with artifact preview, quick plots, and a tidy explorer.",
        "template": EXPERIMENT_EDA_BASIC_TEMPLATE,
    },
    "eda/microplate": {
        "description": "Microplate-focused EDA with time-series and snapshot cross-sections.",
        "template": EXPERIMENT_EDA_MICROPLATE_TEMPLATE,
    },
    "eda/cytometry": {
        "description": "Cytometry-focused EDA with population distribution plots.",
        "template": EXPERIMENT_EDA_CYTOMETRY_TEMPLATE,
    },
}


def list_notebook_presets() -> list[tuple[str, str]]:
    return sorted((name, info["description"]) for name, info in NOTEBOOK_PRESETS.items())


def resolve_notebook_preset(name: str) -> str:
    if name not in NOTEBOOK_PRESETS:
        opts = ", ".join(sorted(NOTEBOOK_PRESETS))
        raise ConfigError(f"Unknown notebook preset {name!r}. Available presets: {opts}")
    return NOTEBOOK_PRESETS[name]["template"]


def write_experiment_notebook(target: Path, *, preset: str = "eda/basic", overwrite: bool = False) -> Path:
    if target.exists() and not overwrite:
        raise ConfigError(f"Notebook already exists: {target}")
    template = resolve_notebook_preset(preset)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    return target
