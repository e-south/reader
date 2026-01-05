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
def _(design_treatment_note, design_treatment_rows, exp_dir, exp_meta, mo):
    _exp_id = exp_meta.get("id") or exp_dir.name
    _exp_title = exp_meta.get("title") or _exp_id
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
def _():
    return

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
