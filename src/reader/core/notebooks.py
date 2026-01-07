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
    mo.vstack(
        [
            mo.md(f"# {_exp_title}\\n**Experiment id:** `{_exp_id}`"),
            mo.md("**Design IDs + treatments**"),
            _design_table,
        ]
    )

@app.cell(hide_code=True)
def _(artifact_dropdown, artifact_note, df_error, mo):
    _elements = [mo.md("## Dataset selection")]
    if artifact_dropdown is None:
        _elements.append(mo.md(artifact_note or "No datasets found."))
    else:
        _elements.append(artifact_dropdown)
        if df_error:
            _elements.append(mo.md(f"**Load error:** `{df_error}`"))
    mo.vstack(_elements)

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
        "description": "Minimal artifact explorer (same scaffold as notebook/basic).",
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
