"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/engine.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import warnings
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from reader.core.artifacts import ArtifactStore
from reader.core.config_model import ReaderSpec
from reader.core.context import RunContext
from reader.core.contracts import BUILTIN, validate_df
from reader.core.errors import ConfigError, ContractError, ExecutionError, ReaderError
from reader.core.mpl import ensure_mpl_cache_dir
from reader.core.registry import Plugin, load_entry_points
from reader.core.specs import ensure_unique_spec_ids, resolve_export_specs, resolve_plot_specs


def _digest_cfg(plugin_cfg: Any) -> str:
    # stable hash of pydantic model or dict
    if hasattr(plugin_cfg, "model_dump"):
        payload = plugin_cfg.model_dump(mode="json")
    elif isinstance(plugin_cfg, dict):
        payload = plugin_cfg
    else:
        payload = json.loads(json.dumps(plugin_cfg, default=str))
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _needs_plot_palette(steps: list[Any], palette: str | None) -> bool:
    if palette is None:
        return False
    return any(getattr(step, "uses", "").startswith("plot/") for step in steps)


def _collect_categories(steps: list[Any]) -> set[str]:
    cats: set[str] = set()
    for step in steps:
        uses = getattr(step, "uses", "")
        if "/" in uses:
            cats.add(uses.split("/", 1)[0])
    return cats


def _snapshot_dir(root: Path) -> dict[Path, float]:
    if not root.exists():
        return {}
    return {p: p.stat().st_mtime for p in root.rglob("*") if p.is_file()}


def _diff_files(before: dict[Path, float], after: dict[Path, float]) -> list[Path]:
    changed: list[Path] = []
    for path, mtime in after.items():
        prev = before.get(path)
        if prev is None or mtime > prev + 1e-6:
            changed.append(path)
    return changed


def _resolve_inputs(store: ArtifactStore, reads: dict[str, str]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for label, target in reads.items():
        if isinstance(target, str) and target.startswith("file:"):
            # pass through file path; plugin validates
            path = Path(target.split("file:", 1)[1])
            if not path.exists():
                raise ExecutionError(f"Input file missing for '{label}': {path}")
            if path.is_dir():
                raise ExecutionError(f"Input file path is a directory for '{label}': {path}")
            inputs[label] = path
        else:
            art = store.read(target)
            inputs[label] = art
    return inputs


def _assert_input_contracts(
    plugin: Plugin,
    inputs: dict[str, Any],
    *,
    where: str,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    req = plugin.input_contracts()
    allowed: set[str] = set()
    for raw_name, contract_id in req.items():
        optional = raw_name.endswith("?")
        name = raw_name[:-1] if optional else raw_name
        allowed.add(name)
        if name not in inputs:
            if optional:
                continue
            raise ExecutionError(f"[{where}] input '{name}' is required by plugin but not provided in 'reads'")
        if contract_id == "none":
            continue
        inp = inputs[name]
        if getattr(inp, "contract_id", None) != contract_id:
            msg = (
                f"[{where}] input '{name}' must be contract {contract_id} "
                f"but got {getattr(inp, 'contract_id', None)}"
            )
            if strict:
                raise ExecutionError(msg)
            if logger is not None:
                logger.warning("contract relaxed • %s", msg)
            else:
                warnings.warn(msg, stacklevel=2)
    extra = sorted(set(inputs) - allowed)
    if extra:
        raise ExecutionError(f"[{where}] unexpected inputs provided: {extra} (allowed: {sorted(allowed)})")


def _assert_output_contracts(
    plugin: Plugin,
    outputs: dict[str, Any],
    *,
    where: str,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    exp = plugin.output_contracts()
    # Exactly the declared outputs must be present
    if set(outputs.keys()) != set(exp.keys()):
        raise ExecutionError(f"[{where}] plugin must emit outputs {sorted(exp)} but emitted {sorted(outputs)}")
    for name, cid in exp.items():
        val = outputs[name]
        if cid == "none":
            if isinstance(val, pd.DataFrame):
                msg = f"[{where}] output '{name}' is declared as contract 'none' but returned a DataFrame"
                if strict:
                    raise ExecutionError(msg)
                if logger is not None:
                    logger.warning("contract relaxed • %s", msg)
                else:
                    warnings.warn(msg, stacklevel=2)
            continue
        if isinstance(val, pd.DataFrame):
            contract = BUILTIN.get(cid)
            if not contract:
                msg = f"[{where}] unknown contract id {cid}"
                if strict:
                    raise ExecutionError(msg)
                if logger is not None:
                    logger.warning("contract relaxed • %s", msg)
                else:
                    warnings.warn(msg, stacklevel=2)
                continue
            try:
                validate_df(val, contract, where=where)
            except ContractError as e:
                msg = str(e)
                if strict:
                    raise ExecutionError(msg) from e
                if logger is not None:
                    logger.warning("contract relaxed • %s", msg)
                else:
                    warnings.warn(msg, stacklevel=2)


def _resolve_output_labels(
    *, step_id: str, output_contracts: dict[str, str], writes: dict[str, str]
) -> dict[str, str]:
    unknown = sorted(set(writes) - set(output_contracts))
    if unknown:
        raise ExecutionError(
            f"[{step_id}] writes includes unknown outputs: {unknown} (expected: {sorted(output_contracts)})"
        )
    labels: dict[str, str] = {}
    for out_name, cid in output_contracts.items():
        if out_name in writes:
            if cid == "none":
                raise ExecutionError(f"[{step_id}] writes cannot target output '{out_name}' (contract is 'none').")
            label = writes[out_name]
        else:
            label = f"{step_id}/{out_name}"
        if not isinstance(label, str) or not label.strip():
            raise ExecutionError(f"[{step_id}] writes for '{out_name}' must be a non-empty string.")
        labels[out_name] = label
    if len(set(labels.values())) != len(labels):
        raise ExecutionError(f"[{step_id}] writes produce duplicate output labels: {labels}")
    return labels


def _plan_table(steps: list[Any], registry: Any, *, title: str) -> Table:
    table = Table(
        title=title,
        title_justify="left",
        title_style="bold cyan",
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
        show_lines=False,
    )
    table.add_column("#", justify="right", style="muted")
    table.add_column("Step ID", style="accent")
    table.add_column("Plugin")
    table.add_column("Inputs")
    table.add_column("Outputs")
    for i, step in enumerate(steps, 1):
        P = registry.resolve(step.uses)
        in_lines: list[str] = []
        for raw_name, contract in P.input_contracts().items():
            optional = raw_name.endswith("?")
            name = raw_name[:-1] if optional else raw_name
            contract_label = contract
            target = (step.reads or {}).get(name)
            if contract == "none" and isinstance(target, str) and target.startswith("file:"):
                contract_label = "file"
            if optional:
                in_lines.append(f"{name} ({contract_label}, optional)")
            else:
                in_lines.append(f"{name} ({contract_label})")

        out_lines: list[str] = []
        output_contracts = P.output_contracts()
        for out_name, contract in output_contracts.items():
            if contract == "none" and out_name == "files":
                if title.lower().startswith("plot"):
                    out_lines.append("files → outputs/plots/")
                    continue
                if title.lower().startswith("export"):
                    out_lines.append("files → outputs/exports/")
                    continue
            label = (step.writes or {}).get(out_name, f"{step.id}/{out_name}") if hasattr(step, "writes") else out_name
            out_lines.append(f"{label} ({contract})")
        inp = "\n".join(in_lines) if in_lines else "—"
        out = "\n".join(out_lines) if out_lines else "—"
        table.add_row(str(i), step.id, step.uses, inp, out)
    return table


def build_next_steps(spec: ReaderSpec, *, job_label: str | None = None) -> list[tuple[str, str]]:
    label = (job_label or "").strip()

    def _cmd(base: str, tail: str = "") -> str:
        return f"{base} {label}{tail}" if label else f"{base}{tail}"

    steps: list[tuple[str, str]] = []
    plot_specs = resolve_plot_specs(spec)
    export_specs = resolve_export_specs(spec)
    notebook_preset = None
    if getattr(spec, "notebook", None) and getattr(spec.notebook, "preset", None):
        notebook_preset = spec.notebook.preset
    if not notebook_preset:
        notebook_preset = "notebook/plots" if plot_specs else "notebook/basic"
    steps.append((_cmd("reader artifacts"), "Review generated artifacts (QC)"))
    if plot_specs:
        steps.append((_cmd("reader plot"), "Save plot files to outputs/plots"))
    if export_specs:
        steps.append((_cmd("reader export"), "Write export files to outputs/exports"))
    steps.append((_cmd("reader notebook"), f"Open a notebook (preset {notebook_preset})"))
    return steps


def explain(
    spec: ReaderSpec,
    *,
    console: Console,
    registry=None,
    plot_specs=None,
    export_specs=None,
) -> None:
    pipeline_steps = list(spec.pipeline.steps)
    plot_specs = list(plot_specs) if plot_specs is not None else resolve_plot_specs(spec)
    export_specs = list(export_specs) if export_specs is not None else resolve_export_specs(spec)
    ensure_unique_spec_ids(pipeline_steps, plot_specs, export_specs)
    categories = _collect_categories(pipeline_steps + plot_specs + export_specs)
    if "plot" in categories:
        out_dir = Path(spec.paths.outputs)
        ensure_mpl_cache_dir(base_dir=out_dir)
    registry = registry or load_entry_points(categories=categories)
    if pipeline_steps:
        pipeline = _plan_table(pipeline_steps, registry, title="Pipeline")
        console.print(
            Panel(
                pipeline,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(pipeline_steps)} steps", style="dim"),
            )
        )
    if plot_specs:
        plots_table = _plan_table(plot_specs, registry, title="Plots")
        console.print(
            Panel(
                plots_table,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(plot_specs)} specs", style="dim"),
            )
        )
    if export_specs:
        exports_table = _plan_table(export_specs, registry, title="Exports")
        console.print(
            Panel(
                exports_table,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(export_specs)} specs", style="dim"),
            )
        )


def validate(spec: ReaderSpec, *, console: Console, check_files: bool = False, exp_root: Path | None = None) -> None:
    # schema already validated by pydantic; here we ensure contracts exist
    pipeline_steps = list(spec.pipeline.steps)
    plot_specs = resolve_plot_specs(spec)
    export_specs = resolve_export_specs(spec)
    ensure_unique_spec_ids(pipeline_steps, plot_specs, export_specs)
    categories = _collect_categories(pipeline_steps + plot_specs + export_specs)
    if "plot" in categories:
        out_dir = Path(spec.paths.outputs)
        ensure_mpl_cache_dir(base_dir=out_dir)
    registry = load_entry_points(categories=categories)

    def _validate_steps(items: list[Any], label: str, *, available_labels: set[str] | None = None) -> set[str]:
        available = set(available_labels or set())
        produced: set[str] = set()
        for step in items:
            uses = step.uses
            try:
                plugin_cls = registry.resolve(uses)
            except Exception as e:
                raise ConfigError(f"{label} {step.id}: {e}") from e
            # check reads against declared inputs
            try:
                req = plugin_cls.input_contracts()
                expected = set()
                required = set()
                for raw_name in req:
                    optional = raw_name.endswith("?")
                    name = raw_name[:-1] if optional else raw_name
                    expected.add(name)
                    if not optional:
                        required.add(name)
                provided = set((step.reads or {}).keys())
                missing = sorted(required - provided)
                extra = sorted(provided - expected)
                if missing or extra:
                    parts = []
                    if missing:
                        parts.append(f"missing inputs: {missing}")
                    if extra:
                        parts.append(f"unexpected inputs: {extra}")
                    raise ConfigError(f"{label} {step.id}: reads do not match plugin inputs ({'; '.join(parts)})")
            except ConfigError:
                raise
            except Exception as e:
                raise ConfigError(f"{label} {step.id}: could not validate reads: {e}") from e
            # check reads reference earlier outputs (or file:)
            for key, target in (step.reads or {}).items():
                if isinstance(target, str) and target.startswith("file:"):
                    continue
                if target not in (available | produced):
                    preview = sorted(available | produced)
                    shown = ", ".join(preview[:12]) if preview else "—"
                    tail = " …" if len(preview) > 12 else ""
                    raise ConfigError(
                        f"{label} {step.id}: reads '{key}' → '{target}', which is not produced by any prior step. "
                        f"Known labels so far: {shown}{tail}. "
                        "Check writes/reads aliases or use file: paths."
                    )
            # validate config model
            try:
                plugin_cls.ConfigModel.model_validate(step.with_ or {})
            except Exception as e:
                raise ConfigError(f"{label} {step.id}: invalid config for {uses}: {e}") from e
            try:
                output_contracts = plugin_cls.output_contracts()
                output_labels = _resolve_output_labels(
                    step_id=step.id,
                    output_contracts=output_contracts,
                    writes=(step.writes or {}),
                )
            except ExecutionError as e:
                raise ConfigError(f"{label} {step.id}: {e}") from e
            for out_name, out_label in output_labels.items():
                if output_contracts[out_name] == "none":
                    continue
                if out_label in available or out_label in produced:
                    raise ConfigError(
                        f"{label} {step.id}: output label '{out_label}' is already produced by another step. "
                        "Use a unique writes mapping to avoid clobbering artifacts."
                    )
                produced.add(out_label)

        return available | produced

    pipeline_labels = _validate_steps(pipeline_steps, "pipeline")

    def _validate_specs(items: list[Any], label: str, *, available_labels: set[str]) -> None:
        for spec_item in items:
            uses = spec_item.uses
            try:
                plugin_cls = registry.resolve(uses)
            except Exception as e:
                raise ConfigError(f"{label} {spec_item.id}: {e}") from e
            # check reads against declared inputs
            try:
                req = plugin_cls.input_contracts()
                expected = set()
                required = set()
                for raw_name in req:
                    optional = raw_name.endswith("?")
                    name = raw_name[:-1] if optional else raw_name
                    expected.add(name)
                    if not optional:
                        required.add(name)
                provided = set((spec_item.reads or {}).keys())
                missing = sorted(required - provided)
                extra = sorted(provided - expected)
                if missing or extra:
                    parts = []
                    if missing:
                        parts.append(f"missing inputs: {missing}")
                    if extra:
                        parts.append(f"unexpected inputs: {extra}")
                    raise ConfigError(f"{label} {spec_item.id}: reads do not match plugin inputs ({'; '.join(parts)})")
            except ConfigError:
                raise
            except Exception as e:
                raise ConfigError(f"{label} {spec_item.id}: could not validate reads: {e}") from e
            # check reads reference pipeline outputs (or file:)
            for key, target in (spec_item.reads or {}).items():
                if isinstance(target, str) and target.startswith("file:"):
                    continue
                if target not in available_labels:
                    preview = sorted(available_labels)
                    shown = ", ".join(preview[:12]) if preview else "—"
                    tail = " …" if len(preview) > 12 else ""
                    raise ConfigError(
                        f"{label} {spec_item.id}: reads '{key}' → '{target}', which is not produced by pipeline. "
                        f"Known labels: {shown}{tail}. "
                        "Check writes/reads aliases or use file: paths."
                    )
            # validate config model
            try:
                plugin_cls.ConfigModel.model_validate(spec_item.with_ or {})
            except Exception as e:
                raise ConfigError(f"{label} {spec_item.id}: invalid config for {uses}: {e}") from e
            # enforce that plot/export plugins do not emit data artifacts
            output_contracts = plugin_cls.output_contracts()
            data_outputs = {k: v for k, v in output_contracts.items() if v != "none"}
            if data_outputs:
                raise ConfigError(
                    f"{label} {spec_item.id}: plugins must not declare data outputs (got {data_outputs}). "
                    "Move data outputs into the pipeline."
                )

    if plot_specs:
        _validate_specs(plot_specs, "plot", available_labels=pipeline_labels)
    if export_specs:
        _validate_specs(export_specs, "export", available_labels=pipeline_labels)

    files_checked = None
    missing_files: list[tuple[str, str, str, Path]] = []
    missing_roots: list[tuple[str, str, Path]] = []
    if check_files:
        entries: list[tuple[str, str, str, Path]] = []
        roots: list[tuple[str, str, Path]] = []
        for label, items in (
            ("pipeline", pipeline_steps),
            ("plot", plot_specs),
            ("export", export_specs),
        ):
            for step in items:
                for key, target in (step.reads or {}).items():
                    if isinstance(target, str) and target.startswith("file:"):
                        path = Path(target.split("file:", 1)[1])
                        entries.append((label, step.id, key, path))
                auto_roots = (step.with_ or {}).get("auto_roots") if hasattr(step, "with_") else None
                if isinstance(auto_roots, list):
                    for root in auto_roots:
                        roots.append((label, step.id, Path(root)))
        files_checked = (len(entries), len(roots))
        for label, step_id, key, path in entries:
            if not path.exists():
                missing_files.append((label, step_id, key, path))
        for label, step_id, root in roots:
            if not root.exists():
                missing_roots.append((label, step_id, root))
        if missing_files or missing_roots:
            lines = ["Missing input files:"]
            for label, step_id, key, path in missing_files:
                rel = path
                if exp_root is not None:
                    try:
                        rel = path.relative_to(exp_root)
                    except Exception:
                        rel = path
                lines.append(f"- {label}:{step_id} • {key} → {rel}")
            for label, step_id, root in missing_roots:
                rel = root
                if exp_root is not None:
                    try:
                        rel = root.relative_to(exp_root)
                    except Exception:
                        rel = root
                lines.append(f"- {label}:{step_id} • auto_roots → {rel}")
            raise ConfigError("\n".join(lines))

    lines = [
        "[green]✓ Config validated[/green]",
        f"[dim]pipeline[/dim]: {len(pipeline_steps)}",
        f"[dim]plots[/dim]: {len(plot_specs)}",
        f"[dim]exports[/dim]: {len(export_specs)}",
        "[dim]checks[/dim]: schema, plugin availability, reads, output labels, plugin config",
    ]
    if check_files:
        file_total, root_total = files_checked or (0, 0)
        if file_total == 0 and root_total == 0:
            lines.append("[dim]files[/dim]: none declared")
        else:
            parts = []
            if file_total:
                parts.append(f"{file_total} file input(s)")
            if root_total:
                parts.append(f"{root_total} auto_root(s)")
            lines.append(f"[dim]files[/dim]: ok ({', '.join(parts)})")
    else:
        lines.append("[dim]files[/dim]: skipped (--no-files)")
    lines.append("[dim]tip[/dim]: use 'reader explain' to see inputs/outputs")
    console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))


def run_spec(
    spec: ReaderSpec,
    *,
    resume_from: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    verbose: bool = True,
    console: Console | None = None,
    include_pipeline: bool = True,
    include_plots: bool = True,
    include_exports: bool = True,
    plot_specs=None,
    export_specs=None,
    job_label: str | None = None,
    show_next_steps: bool = False,
) -> None:
    os.environ.setdefault("ARROW_LOG_LEVEL", "FATAL")
    exp = spec.experiment
    out_dir = Path(spec.paths.outputs).resolve()
    if out_dir.exists() and not out_dir.is_dir():
        raise ConfigError(f"paths.outputs points to a file: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # logging
    level_name = str(log_level).upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level_name not in valid_levels:
        raise ConfigError(f"Invalid log level {log_level!r}. Choose one of: {sorted(valid_levels)}")
    logger = logging.getLogger("reader")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()
    logger.setLevel(getattr(logging, level_name))
    try:
        fh = logging.FileHandler(out_dir / "reader.log", encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Cannot write reader.log in outputs directory: {out_dir}") from e
    fmt_file = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt_file)
    # Pretty console logs with Rich (integrates with progress bars)
    console = console or Console()
    sh = RichHandler(
        console=console, markup=True, rich_tracebacks=True, show_level=True, show_time=False, show_path=False
    )
    sh.setLevel(getattr(logging, level_name) if verbose else logging.WARNING)
    logger.addHandler(fh)
    logger.addHandler(sh)

    pipeline_steps = list(spec.pipeline.steps)
    plot_steps = list(plot_specs) if plot_specs is not None else resolve_plot_specs(spec)
    export_steps = list(export_specs) if export_specs is not None else resolve_export_specs(spec)
    ensure_unique_spec_ids(pipeline_steps, plot_steps, export_steps)

    if not include_pipeline:
        if resume_from or until:
            raise ConfigError("--from/--until require pipeline execution; use reader run for sliced runs.")
        pipeline_steps = []
    else:
        # resume/until slicing (pipeline only)
        if resume_from:
            try:
                idx = next(i for i, s in enumerate(pipeline_steps) if s.id == resume_from)
                pipeline_steps = pipeline_steps[idx:]
            except StopIteration:
                raise ConfigError(f"--from: step id '{resume_from}' not found") from None
        if until:
            try:
                idx = next(i for i, s in enumerate(pipeline_steps) if s.id == until)
                pipeline_steps = pipeline_steps[: idx + 1]
            except StopIteration:
                raise ConfigError(f"--until: step id '{until}' not found") from None

    if not include_plots:
        plot_steps = []
    if not include_exports:
        export_steps = []

    all_steps = pipeline_steps + plot_steps + export_steps

    if plot_steps:
        ensure_mpl_cache_dir(base_dir=out_dir)

    palette = spec.plotting.palette if spec.plotting else None
    if palette is not None:
        if not isinstance(palette, str) or not palette.strip():
            raise ConfigError("plotting.palette must be a non-empty string or null")
        palette = palette.strip()

    PaletteBook = None
    available_palettes = None
    if (not dry_run) and _needs_plot_palette(all_steps, palette):
        try:
            mod = import_module("reader.lib.microplates.style")
            PaletteBook = getattr(mod, "PaletteBook", None)
            available_palettes = getattr(mod, "available_palettes", None)
            if PaletteBook is None or available_palettes is None:
                raise ImportError("PaletteBook or available_palettes not found in style module")
        except Exception as e:
            raise ConfigError(
                "Plot palettes require matplotlib; install plotting dependencies or set plotting.palette: null."
            ) from e
        if palette not in available_palettes():
            raise ConfigError(
                f"Unknown palette {palette!r}. Available: {available_palettes()} (or set plotting.palette: null)."
            )

    plots_cfg = spec.paths.plots
    exports_cfg = spec.paths.exports
    store = ArtifactStore(
        out_dir,
        plots_subdir=(plots_cfg if plots_cfg not in ("", ".", "./") else None),
        exports_subdir=(exports_cfg if exports_cfg not in ("", ".", "./") else None),
    )
    plots_dir = store.plots_dir
    exports_dir = store.exports_dir

    ctx = RunContext(
        exp_dir=Path(exp.root or out_dir.parent),
        outputs_dir=out_dir,
        artifacts_dir=store.artifacts_dir,
        plots_dir=plots_dir,
        exports_dir=exports_dir,
        manifest_path=store.manifest_path,
        logger=logger,
        palette_book=(None if (palette is None or PaletteBook is None) else PaletteBook(palette)),
        strict=bool(spec.pipeline.runtime.get("strict", True)) if isinstance(spec.pipeline.runtime, dict) else True,
        groupings=(spec.data.groupings or {}),
        aliases=(spec.data.aliases or {}),
    )

    registry = load_entry_points(categories=_collect_categories(all_steps))

    if dry_run:
        console.print(Panel.fit("DRY RUN — printing plan", border_style="yellow", box=box.ROUNDED))
        explain(spec, console=console, registry=registry, plot_specs=plot_steps, export_specs=export_steps)
        return

    def _execute_step(step: Any, *, phase: str) -> None:
        plugin_cls = registry.resolve(step.uses)
        cfg = plugin_cls.ConfigModel.model_validate(step.with_ or {})
        plugin = plugin_cls()
        output_contracts = plugin.output_contracts()
        output_labels = _resolve_output_labels(
            step_id=step.id,
            output_contracts=output_contracts,
            writes=(step.writes or {}),
        )
        pre_state = {}
        if phase == "plots":
            pre_state = _snapshot_dir(ctx.plots_dir)
        elif phase == "exports":
            pre_state = _snapshot_dir(ctx.exports_dir)

        # collect & check inputs
        inputs = _resolve_inputs(store, step.reads or {})
        _assert_input_contracts(plugin, inputs, where=f"{step.id}", strict=ctx.strict, logger=ctx.logger)

        # adapt artifacts -> dataframes/files for plugin
        plug_inputs: dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "load_dataframe"):
                plug_inputs[k] = v.load_dataframe()
            else:
                plug_inputs[k] = v  # Path for "file:" pseudo artifact

        # run
        try:
            outputs = plugin.run(ctx, plug_inputs, cfg)
        except ReaderError:
            raise
        except Exception as e:
            raise ExecutionError(f"{phase} {step.id} crashed: {e}") from e

        # check outputs vs declared contracts
        _assert_output_contracts(plugin, outputs, where=f"{step.id}", strict=ctx.strict, logger=ctx.logger)

        # persist outputs (pipeline only)
        for out_name, obj in outputs.items():
            cid = output_contracts[out_name]
            if isinstance(obj, pd.DataFrame) and cid != "none":
                store.persist_dataframe(
                    step_id=step.id,
                    plugin_key=plugin.key if hasattr(plugin, "key") else plugin_cls.__name__,
                    out_name=out_name,
                    label=output_labels[out_name],
                    df=obj,
                    contract_id=cid,
                    inputs=[inputs[n].label if hasattr(inputs[n], "label") else str(inputs[n]) for n in (step.reads or {})],
                    config_digest=_digest_cfg(cfg),
                    validate_contract=ctx.strict,
                )
            elif cid == "none":
                continue
            else:
                raise ExecutionError(f"{phase} {step.id}: unsupported output type for {out_name}")

        if phase in {"plots", "exports"}:
            base_dir = ctx.plots_dir if phase == "plots" else ctx.exports_dir
            post_state = _snapshot_dir(base_dir)
            changed = _diff_files(pre_state, post_state)
            file_outputs = outputs.get("files")
            extra_files: list[Path] = []
            if file_outputs:
                if isinstance(file_outputs, str | Path):
                    extra_files = [Path(file_outputs)]
                elif isinstance(file_outputs, list):
                    extra_files = [Path(p) for p in file_outputs if p]
            combined = list({*changed, *extra_files})
            rel_files: list[str] = []
            for p in combined:
                try:
                    rel_files.append(str(p.relative_to(ctx.outputs_dir)))
                except Exception:
                    rel_files.append(str(p))
            entry = {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "step_id": step.id,
                "plugin": plugin.key if hasattr(plugin, "key") else plugin_cls.__name__,
                "config_digest": _digest_cfg(cfg),
                "inputs": [inputs[n].label if hasattr(inputs[n], "label") else str(inputs[n]) for n in (step.reads or {})],
                "files": sorted(set(rel_files)),
            }
            if phase == "plots":
                store.append_plot_entry(entry)
            else:
                store.append_export_entry(entry)

    def _run_steps(items: list[Any], *, phase: str) -> None:
        if not items:
            return
        if verbose:
            for ordinal, step in enumerate(items, 1):
                ctx.logger.info("→ %s %s [%d/%d] uses=%s", phase, step.id, ordinal, len(items), step.uses)
                _execute_step(step, phase=phase)
            return

        with Progress(
            SpinnerColumn(style="accent"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"{phase.title()} ({len(items)} steps)", total=len(items))
            for step in items:
                progress.update(task, description=f"{phase}: {step.id}")
                _execute_step(step, phase=phase)
                progress.advance(task)

    ctx.logger.info(
        "run • pipeline=%d step(s)%s • plots=%d spec(s) • exports=%d spec(s)",
        len(pipeline_steps),
        " (skipped)" if (not include_pipeline) else "",
        len(plot_steps),
        len(export_steps),
    )
    if not verbose:
        console.print(
            f"[accent]run[/accent] • pipeline={len(pipeline_steps)} step(s){' (skipped)' if (not include_pipeline) else ''} "
            f"• plots={len(plot_steps)} spec(s) • exports={len(export_steps)} spec(s)"
        )
    _run_steps(pipeline_steps, phase="pipeline")
    if plot_steps:
        _run_steps(plot_steps, phase="plots")
    if export_steps:
        _run_steps(export_steps, phase="exports")

    if show_next_steps:
        next_steps = build_next_steps(spec, job_label=job_label)
        table = Table(show_header=True, header_style="bold", box=box.ROUNDED, expand=False)
        table.add_column("Command", style="accent")
        table.add_column("What it does")
        for cmd, desc in next_steps:
            table.add_row(cmd, desc)
        console.print(
            Panel(
                table,
                border_style="green",
                box=box.ROUNDED,
                title=f"Artifacts generated in [path]{out_dir}[/path]",
                title_align="left",
            )
        )
    else:
        console.print(Panel.fit(f"✓ Done — outputs in {str(out_dir)}", border_style="green", box=box.ROUNDED))


def run_job(
    spec_path: Path,
    *,
    resume_from: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    verbose: bool = True,
    console: Console | None = None,
    include_pipeline: bool = True,
    include_plots: bool = True,
    include_exports: bool = True,
    job_label: str | None = None,
    show_next_steps: bool = False,
) -> None:
    spec = ReaderSpec.load(spec_path)
    run_spec(
        spec,
        resume_from=resume_from,
        until=until,
        dry_run=dry_run,
        log_level=log_level,
        verbose=verbose,
        console=console,
        include_pipeline=include_pipeline,
        include_plots=include_plots,
        include_exports=include_exports,
        job_label=job_label,
        show_next_steps=show_next_steps,
    )
