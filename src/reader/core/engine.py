"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/engine.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from importlib import import_module
from pathlib import Path
from datetime import UTC, datetime
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
from reader.core.errors import ConfigError, ExecutionError, ReaderError
from reader.core.registry import Plugin, load_entry_points


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


def _ensure_mpl_cache_dir(base_dir: Path | None = None) -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    override = os.environ.get("READER_MPLCONFIGDIR")
    if override:
        cache_dir = Path(override).expanduser()
    else:
        root = base_dir if base_dir is not None else Path.cwd()
        cache_dir = root / ".cache" / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


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
            inputs[label] = path
        else:
            art = store.read(target)
            inputs[label] = art
    return inputs


def _assert_input_contracts(plugin: Plugin, inputs: dict[str, Any], *, where: str) -> None:
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
            raise ExecutionError(
                f"[{where}] input '{name}' must be contract {contract_id} but got {getattr(inp, 'contract_id', None)}"
            )
    extra = sorted(set(inputs) - allowed)
    if extra:
        raise ExecutionError(f"[{where}] unexpected inputs provided: {extra} (allowed: {sorted(allowed)})")


def _assert_output_contracts(plugin: Plugin, outputs: dict[str, Any], *, where: str) -> None:
    exp = plugin.output_contracts()
    # Exactly the declared outputs must be present
    if set(outputs.keys()) != set(exp.keys()):
        raise ExecutionError(f"[{where}] plugin must emit outputs {sorted(exp)} but emitted {sorted(outputs)}")
    for name, cid in exp.items():
        val = outputs[name]
        if isinstance(val, pd.DataFrame) and cid != "none":
            contract = BUILTIN.get(cid)
            if not contract:
                raise ExecutionError(f"[{where}] unknown contract id {cid}")
            validate_df(val, contract, where=where)


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
    table.add_column("Inputs (label:contract)")
    table.add_column("Outputs (label:contract)")
    for i, step in enumerate(steps, 1):
        P = registry.resolve(step.uses)
        inp = ", ".join(f"{k}:{v}" for k, v in P.input_contracts().items())
        out = ", ".join(
            f"{step.writes.get(k, f'{step.id}/{k}')}:{v}" for k, v in P.output_contracts().items()
        )
        table.add_row(str(i), step.id, step.uses, inp, out)
    return table


def explain(spec: ReaderSpec, *, console: Console, registry=None) -> None:
    steps = list(spec.steps)
    deliverables = list(spec.deliverables or [])
    registry = registry or load_entry_points(categories=_collect_categories(steps + deliverables))
    if steps:
        pipeline = _plan_table(steps, registry, title="Pipeline")
        console.print(
            Panel(
                pipeline,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(steps)} steps", style="dim"),
            )
        )
    if deliverables:
        deliverables_table = _plan_table(deliverables, registry, title="Deliverables")
        console.print(
            Panel(
                deliverables_table,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(deliverables)} steps", style="dim"),
            )
        )


def validate(spec: ReaderSpec, *, console: Console) -> None:
    # schema already validated by pydantic; here we ensure contracts exist
    steps = list(spec.steps)
    deliverables = list(spec.deliverables or [])
    registry = load_entry_points(categories=_collect_categories(steps + deliverables))

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

    pipeline_labels = _validate_steps(steps, "step")
    if deliverables:
        _validate_steps(deliverables, "deliverable", available_labels=pipeline_labels)

    lines = [
        "[green]✓ Config validated[/green]",
        f"[dim]steps[/dim]: {len(steps)}",
        f"[dim]deliverables[/dim]: {len(deliverables)}",
        "[dim]checks[/dim]: schema, plugin availability, reads, output labels, plugin config",
        "[dim]not checked[/dim]: input files, data contents",
        "[dim]tip[/dim]: use 'reader explain' to see inputs/outputs",
    ]
    console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))


def run_job(
    spec_path: Path,
    *,
    resume_from: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    console: Console | None = None,
    include_pipeline: bool = True,
    include_deliverables: bool = True,
) -> None:
    os.environ.setdefault("ARROW_LOG_LEVEL", "FATAL")
    spec = ReaderSpec.load(spec_path)
    exp = spec.experiment
    out_dir = Path(exp["outputs"]).resolve()
    # Normalize outputs relative to the experiment root if still relative
    out_dir = Path(exp["outputs"])
    if not out_dir.is_absolute():
        out_dir = Path(exp["root"]) / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # logging
    logger = logging.getLogger("reader")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    fh = logging.FileHandler(out_dir / "reader.log", encoding="utf-8")
    fmt_file = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt_file)
    # Pretty console logs with Rich (integrates with progress bars)
    console = console or Console()
    sh = RichHandler(
        console=console, markup=True, rich_tracebacks=True, show_level=True, show_time=False, show_path=False
    )
    logger.addHandler(fh)
    logger.addHandler(sh)

    steps = list(spec.steps)
    if not include_pipeline:
        if resume_from or until:
            raise ConfigError("--resume-from/--until require pipeline execution; use reader run for sliced runs.")
        steps = []
    else:
        # resume/until slicing (pipeline only)
        if resume_from:
            try:
                idx = next(i for i, s in enumerate(steps) if s.id == resume_from)
                steps = steps[idx:]
            except StopIteration:
                raise ConfigError(f"--resume-from: step id '{resume_from}' not found") from None
        if until:
            try:
                idx = next(i for i, s in enumerate(steps) if s.id == until)
                steps = steps[: idx + 1]
            except StopIteration:
                raise ConfigError(f"--until: step id '{until}' not found") from None

    deliverables = list(spec.deliverables or [])
    if not include_deliverables:
        deliverables = []
    all_steps = steps + deliverables

    if any(getattr(step, "uses", "").startswith("plot/") for step in all_steps):
        _ensure_mpl_cache_dir(base_dir=out_dir)

    palette = exp.get("palette", "colorblind")

    PaletteBook = None
    if (not dry_run) and _needs_plot_palette(all_steps, palette):
        try:
            mod = import_module("reader.lib.microplates.style")
            PaletteBook = getattr(mod, "PaletteBook", None)
            if PaletteBook is None:
                raise ImportError("PaletteBook not found in style module")
        except Exception as e:
            logger.warning(
                "Plot palette disabled: could not import reader.lib.microplates.style (%s). "
                "Continuing without a palette; plots will use Matplotlib rcParams. "
                "To re‑enable, install plotting extras or set experiment.palette: null. ",
                e.__class__.__name__,
            )
            PaletteBook = None

    # Where should plots go?
    plots_cfg = exp.get("plots_dir", "plots")  # set to None to flatten into outputs/
    plots_dir = out_dir if plots_cfg in (None, "", ".", "./") else out_dir / plots_cfg

    ctx = RunContext(
        exp_dir=Path(exp["root"]),
        outputs_dir=out_dir,
        artifacts_dir=out_dir / "artifacts",
        plots_dir=plots_dir,
        manifest_path=out_dir / "manifest.json",
        logger=logger,
        palette_book=(None if (palette is None or PaletteBook is None) else PaletteBook(palette)),
        strict=bool(spec.runtime.get("strict", True)) if isinstance(spec.runtime, dict) else True,
        collections=(spec.collections or {}),
    )

    store = ArtifactStore(out_dir, plots_subdir=(plots_cfg if plots_cfg not in (None, "", ".", "./") else None))

    registry = load_entry_points(categories=_collect_categories(all_steps))

    if dry_run:
        console.print(Panel.fit("DRY RUN — printing plan", border_style="yellow", box=box.ROUNDED))
        spec_slice = spec.model_copy(update={"steps": steps, "deliverables": deliverables})
        explain(spec_slice, console=console, registry=registry)
        return

    def _run_steps(items: list[Any], *, phase: str) -> None:
        if not items:
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
            task = progress.add_task(f"Running {phase}", total=len(items))

            for ordinal, step in enumerate(items, 1):
                ctx.logger.info("→ %s %s [%d/%d] uses=%s", phase[:-1], step.id, ordinal, len(items), step.uses)
                plugin_cls = registry.resolve(step.uses)
                cfg = plugin_cls.ConfigModel.model_validate(step.with_ or {})
                plugin = plugin_cls()
                output_contracts = plugin.output_contracts()
                output_labels = _resolve_output_labels(
                    step_id=step.id,
                    output_contracts=output_contracts,
                    writes=(step.writes or {}),
                )
                pre_plot_state = _snapshot_dir(ctx.plots_dir) if phase == "deliverables" else {}

                # collect & check inputs
                inputs = _resolve_inputs(store, step.reads or {})
                _assert_input_contracts(plugin, inputs, where=f"{step.id}")

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
                    raise ExecutionError(f"{phase[:-1]} {step.id} crashed: {e}") from e

                # check outputs vs declared contracts
                _assert_output_contracts(plugin, outputs, where=f"{step.id}")

                # persist outputs
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
                            inputs=[
                                inputs[n].label if hasattr(inputs[n], "label") else str(inputs[n])
                                for n in (step.reads or {})
                            ],
                            config_digest=_digest_cfg(cfg),
                        )
                    elif cid == "none":
                        # e.g., plots: plugin must have already written files into ctx.plots_dir
                        continue
                    else:
                        raise ExecutionError(f"{phase[:-1]} {step.id}: unsupported output type for {out_name}")

                if phase == "deliverables":
                    post_plot_state = _snapshot_dir(ctx.plots_dir)
                    plot_files = _diff_files(pre_plot_state, post_plot_state)
                    file_outputs = outputs.get("files")
                    extra_files: list[Path] = []
                    if file_outputs:
                        if isinstance(file_outputs, (str, Path)):
                            extra_files = [Path(file_outputs)]
                        elif isinstance(file_outputs, list):
                            extra_files = [Path(p) for p in file_outputs if p]
                    combined = list({*plot_files, *extra_files})
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
                        "inputs": [
                            inputs[n].label if hasattr(inputs[n], "label") else str(inputs[n])
                            for n in (step.reads or {})
                        ],
                        "files": sorted(rel_files),
                    }
                    store.append_deliverable_entry(entry)

                progress.advance(task)

    ctx.logger.info(
        "run • pipeline=%d step(s)%s • deliverables=%d step(s)",
        len(steps),
        " (skipped)" if (not include_pipeline) else "",
        len(deliverables),
    )
    if include_pipeline and (not include_deliverables) and spec.deliverables:
        ctx.logger.info("deliverables skipped • use `reader deliverables` or `reader run --deliverables`")
    _run_steps(steps, phase="steps")
    if deliverables:
        _run_steps(deliverables, phase="deliverables")

    console.print(Panel.fit(f"✓ Done — outputs in {str(out_dir)}", border_style="green", box=box.ROUNDED))
