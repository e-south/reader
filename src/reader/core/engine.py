"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/engine.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import shutil
import tempfile
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

from reader.core.artifacts import ArtifactStore, ReportStore
from reader.core.config_model import ReaderSpec
from reader.core.context import RunContext
from reader.core.contracts import BUILTIN, validate_df
from reader.core.errors import (
    ConfigError,
    ContractError,
    ExecutionError,
    MergeError,
    ParseError,
    PlotError,
    ReaderError,
    SFXIError,
    TransformError,
)
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


def _resolve_inputs(store: ArtifactStore, reads: dict[str, str]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for label, target in reads.items():
        if isinstance(target, str) and target.startswith("file:"):
            # pass through file path; plugin validates
            inputs[label] = Path(target.split("file:", 1)[1])
        else:
            art = store.read(target)
            inputs[label] = art
    return inputs


def _assert_input_contracts(plugin: Plugin, inputs: dict[str, Any], *, where: str) -> None:
    req = plugin.input_contracts()
    for raw_name, contract_id in req.items():
        optional = raw_name.endswith("?")
        name = raw_name[:-1] if optional else raw_name
        if name not in inputs:
            if optional:
                continue
            raise ExecutionError(f"[{where}] input '{name}' is required by plugin but not provided in 'reads'")
        if contract_id in {"none", "any"}:
            continue
        inp = inputs[name]
        if getattr(inp, "contract_id", None) != contract_id:
            raise ExecutionError(
                f"[{where}] input '{name}' must be contract {contract_id} but got {getattr(inp, 'contract_id', None)}"
            )


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


def _wrap_plugin_error(step_id: str, plugin: Plugin, err: Exception) -> ReaderError:
    msg = f"step {step_id}: {err}"
    if isinstance(err, ValueError):
        category = getattr(plugin, "category", "")
        if category == "ingest":
            return ParseError(msg)
        if category == "merge":
            return MergeError(msg)
        if category == "transform":
            if getattr(plugin, "key", "") == "sfxi":
                return SFXIError(msg)
            return TransformError(msg)
        if category == "plot":
            return PlotError(msg)
        if category == "validator":
            return ContractError(msg)
    return ExecutionError(f"{msg}")


def explain_steps(*, steps: list, console: Console, registry=None, title: str = "Plan") -> None:
    registry = registry or load_entry_points()
    table = Table(
        title=f"[title]{title}[/title]",
        title_justify="left",
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
        out = ", ".join(f"{k}:{v}" for k, v in P.output_contracts().items())
        table.add_row(str(i), step.id, step.uses, inp, out)
    console.print(Panel(table, border_style="accent", box=box.ROUNDED, subtitle=f"[muted]{len(steps)} steps[/muted]"))


def explain(spec: ReaderSpec, *, console: Console, registry=None) -> None:
    explain_steps(steps=spec.steps, console=console, registry=registry, title="Pipeline plan")


def validate(spec: ReaderSpec, *, console: Console) -> None:
    # schema already validated by pydantic; here we ensure contracts exist + reads are well-formed
    registry = load_entry_points()
    seen_ids: set[str] = set()
    available_outputs: dict[str, set[str]] = {}

    def _assert_contract_exists(contract_id: str, *, where: str) -> None:
        if contract_id in {"none", "any"}:
            return
        if contract_id not in BUILTIN:
            raise ConfigError(f"{where}: unknown contract id '{contract_id}' (not in built-ins)")

    for step in spec.steps:
        if step.id in seen_ids:
            raise ConfigError(f"duplicate step id '{step.id}'")
        seen_ids.add(step.id)

        uses = step.uses
        try:
            plugin_cls = registry.resolve(uses)
        except Exception as e:
            raise ConfigError(f"step {step.id}: {e}") from e

        if getattr(plugin_cls, "category", None) in {"plot", "export"}:
            raise ConfigError(
                f"step {step.id}: plugin '{uses}' is category '{plugin_cls.category}'. "
                "Plots/exports must be declared under reports: (use report_presets/report_overrides)."
            )

        # validate config model
        try:
            plugin_cls.ConfigModel.model_validate(step.with_ or {})
        except Exception as e:
            raise ConfigError(f"step {step.id}: invalid config for {uses}: {e}") from e

        # validate plugin contract ids exist (unless 'none')
        for name, cid in plugin_cls.input_contracts().items():
            _assert_contract_exists(cid, where=f"step {step.id} input '{name}'")
        for name, cid in plugin_cls.output_contracts().items():
            _assert_contract_exists(cid, where=f"step {step.id} output '{name}'")

        # validate reads: keys + references
        req_inputs = plugin_cls.input_contracts()
        allowed_input_names = {k.rstrip("?") for k in req_inputs}
        for name in step.reads:
            if name not in allowed_input_names:
                raise ConfigError(f"step {step.id}: reads.{name} is not a valid input for {uses}")

        for raw_name, contract_id in req_inputs.items():
            optional = raw_name.endswith("?")
            name = raw_name[:-1] if optional else raw_name
            if name not in step.reads:
                if optional:
                    continue
                raise ConfigError(f"step {step.id}: missing required input '{name}' for {uses}")
            target = step.reads.get(name)
            if isinstance(target, str) and target.startswith("file:"):
                if contract_id != "none":
                    # Allow file: only for 'none' inputs (explicit file paths)
                    raise ConfigError(
                        f"step {step.id}: reads.{name} is a file: reference but {uses} expects {contract_id}"
                    )
                continue
            if not isinstance(target, str) or "/" not in target:
                raise ConfigError(f"step {step.id}: reads.{name} must be '<step_id>/<output>' or file:<path>")
            src_id, out_name = target.split("/", 1)
            if src_id not in available_outputs:
                raise ConfigError(f"step {step.id}: reads.{name} references '{src_id}', which has not run yet")
            if out_name not in available_outputs[src_id]:
                raise ConfigError(
                    f"step {step.id}: reads.{name} references '{src_id}/{out_name}', "
                    f"but {src_id} outputs {sorted(available_outputs[src_id])}"
                )

        available_outputs[step.id] = set(plugin_cls.output_contracts().keys())

    # Validate report steps (optional, must be plot/export and read from pipeline artifacts)
    for step in spec.reports or []:
        if step.id in seen_ids:
            raise ConfigError(f"duplicate step id '{step.id}' (reports)")
        seen_ids.add(step.id)

        uses = step.uses
        try:
            plugin_cls = registry.resolve(uses)
        except Exception as e:
            raise ConfigError(f"report step {step.id}: {e}") from e

        if getattr(plugin_cls, "category", None) not in {"plot", "export"}:
            raise ConfigError(
                f"report step {step.id}: plugin '{uses}' is category '{plugin_cls.category}', "
                "but reports only allow plot/export plugins."
            )

        try:
            plugin_cls.ConfigModel.model_validate(step.with_ or {})
        except Exception as e:
            raise ConfigError(f"report step {step.id}: invalid config for {uses}: {e}") from e

        for name, cid in plugin_cls.input_contracts().items():
            _assert_contract_exists(cid, where=f"report step {step.id} input '{name}'")
        for name, cid in plugin_cls.output_contracts().items():
            _assert_contract_exists(cid, where=f"report step {step.id} output '{name}'")

        req_inputs = plugin_cls.input_contracts()
        allowed_input_names = {k.rstrip("?") for k in req_inputs}
        for name in step.reads:
            if name not in allowed_input_names:
                raise ConfigError(f"report step {step.id}: reads.{name} is not a valid input for {uses}")

        for raw_name, _contract_id in req_inputs.items():
            optional = raw_name.endswith("?")
            name = raw_name[:-1] if optional else raw_name
            if name not in step.reads:
                if optional:
                    continue
                raise ConfigError(f"report step {step.id}: missing required input '{name}' for {uses}")
            target = step.reads.get(name)
            if isinstance(target, str) and target.startswith("file:"):
                raise ConfigError(
                    f"report step {step.id}: file: inputs are not allowed in reports; use pipeline artifacts instead."
                )
            if not isinstance(target, str) or "/" not in target:
                raise ConfigError(
                    f"report step {step.id}: reads.{name} must be '<step_id>/<output>' (pipeline artifact)."
                )
            src_id, out_name = target.split("/", 1)
            if src_id not in available_outputs:
                raise ConfigError(
                    f"report step {step.id}: reads.{name} references '{src_id}', which is not a pipeline step."
                )
            if out_name not in available_outputs[src_id]:
                raise ConfigError(
                    f"report step {step.id}: reads.{name} references '{src_id}/{out_name}', "
                    f"but {src_id} outputs {sorted(available_outputs[src_id])}"
                )
    console.print(Panel.fit("✓ Config validated", border_style="green", box=box.ROUNDED))


def _init_context(
    spec: ReaderSpec,
    *,
    log_level: str,
    console: Console | None,
    include_plot_steps: bool,
) -> tuple[RunContext, ArtifactStore, object, Path, Console]:
    exp = spec.experiment
    out_dir = Path(exp.outputs)
    if not out_dir.is_absolute():
        out_dir = Path(exp.root or ".") / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("reader")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    fh = logging.FileHandler(out_dir / "reader.log", encoding="utf-8")
    fmt_file = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt_file)
    console = console or Console()
    sh = RichHandler(
        console=console, markup=True, rich_tracebacks=True, show_level=True, show_time=False, show_path=False
    )
    logger.addHandler(fh)
    logger.addHandler(sh)

    if include_plot_steps:
        _ensure_mplconfigdir()

    palette = exp.palette
    palette_book = None
    if include_plot_steps and palette is not None:
        try:
            mod = import_module("reader.plotting.microplates.style")
            palettes = getattr(mod, "_PALETTES", None)
            if isinstance(palettes, dict) and palette not in palettes:
                raise PlotError(
                    f"Unknown palette '{palette}'. Available: {', '.join(sorted(palettes))}. "
                    "Set experiment.palette: null to disable palette selection."
                )
            PaletteBook = getattr(mod, "PaletteBook", None)
            if PaletteBook is None:
                raise PlotError("PaletteBook not found in style module.")
            palette_book = PaletteBook(palette)
        except PlotError:
            raise
        except Exception as e:
            raise PlotError(
                "Plot palette requested but plotting style module could not be loaded. "
                "Install plotting dependencies or set experiment.palette: null."
            ) from e

    plots_cfg = exp.plots_dir if exp.plots_dir is not None else None
    plots_dir = out_dir if plots_cfg in (None, "", ".", "./") else out_dir / plots_cfg

    ctx = RunContext(
        exp_dir=Path(exp.root or "."),
        outputs_dir=out_dir,
        artifacts_dir=out_dir / "artifacts",
        plots_dir=plots_dir,
        manifest_path=out_dir / "manifest.json",
        logger=logger,
        palette_book=palette_book,
        collections=(spec.collections.root if spec.collections is not None else {}),
    )

    store = ArtifactStore(out_dir, plots_subdir=(plots_cfg if plots_cfg not in (None, "", ".", "./") else None))
    registry = load_entry_points()
    return ctx, store, registry, out_dir, console


def _ensure_mplconfigdir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    tmp = tempfile.mkdtemp(prefix="reader-mpl-")
    os.environ["MPLCONFIGDIR"] = tmp
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)


def run_job(
    spec_path: Path,
    *,
    resume_from: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    console: Console | None = None,
) -> None:
    spec = ReaderSpec.load(spec_path)
    has_plot_steps = any(str(step.uses).split("/", 1)[0] == "plot" for step in spec.steps)
    ctx, store, registry, out_dir, console = _init_context(
        spec, log_level=log_level, console=console, include_plot_steps=has_plot_steps
    )
    # Enforce pipeline/report separation even if user skips `reader validate`.
    for step in spec.steps:
        plugin_cls = registry.resolve(step.uses)
        if getattr(plugin_cls, "category", None) in {"plot", "export"}:
            raise ConfigError(
                f"step {step.id}: plugin '{step.uses}' is category '{plugin_cls.category}'. "
                "Plots/exports must be declared under reports: (use report_presets/report_overrides)."
            )

    # resume/until slicing
    steps = spec.steps
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

    if dry_run:
        console.print(Panel.fit("DRY RUN — printing plan", border_style="warn", box=box.ROUNDED))
        explain_steps(steps=spec.steps, console=console, registry=registry, title="Pipeline plan")
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
        task = progress.add_task("Running steps", total=len(steps))

        for ordinal, step in enumerate(steps, 1):
            ctx.logger.info("→ step %s [%d/%d] uses=%s", step.id, ordinal, len(steps), step.uses)
            plugin_cls = registry.resolve(step.uses)
            cfg = plugin_cls.ConfigModel.model_validate(step.with_ or {})
            plugin = plugin_cls()

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
                raise _wrap_plugin_error(step.id, plugin, e) from e

            # check outputs vs declared contracts
            _assert_output_contracts(plugin, outputs, where=f"{step.id}")

            # persist outputs
            for out_name, obj in outputs.items():
                cid = plugin.output_contracts()[out_name]
                if isinstance(obj, pd.DataFrame) and cid != "none":
                    store.persist_dataframe(
                        step_id=step.id,
                        plugin_key=plugin.key if hasattr(plugin, "key") else plugin_cls.__name__,
                        out_name=out_name,
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
                    raise ExecutionError(f"step {step.id}: unsupported output type for {out_name}")

            progress.advance(task)

    console.print(Panel.fit(f"✓ Done — outputs in [path]{str(out_dir)}[/path]", border_style="green", box=box.ROUNDED))


def run_reports(
    spec_path: Path,
    *,
    step_id: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    console: Console | None = None,
) -> None:
    spec = ReaderSpec.load(spec_path)
    report_steps = list(spec.reports or [])
    if not report_steps:
        raise ConfigError("No reports defined in config.yaml (use reports: to add plot/export steps).")

    if step_id:
        report_steps = [s for s in report_steps if s.id == step_id]
        if not report_steps:
            raise ConfigError(f"--step: report step id '{step_id}' not found")

    has_plot_steps = any(str(step.uses).split("/", 1)[0] == "plot" for step in report_steps)
    ctx, store, registry, out_dir, console = _init_context(
        spec, log_level=log_level, console=console, include_plot_steps=has_plot_steps
    )
    report_store = ReportStore(out_dir)

    if dry_run:
        console.print(Panel.fit("DRY RUN — printing report plan", border_style="warn", box=box.ROUNDED))
        explain_steps(steps=report_steps, console=console, registry=registry, title="Report plan")
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
        task = progress.add_task("Running reports", total=len(report_steps))

        for ordinal, step in enumerate(report_steps, 1):
            ctx.logger.info("→ report %s [%d/%d] uses=%s", step.id, ordinal, len(report_steps), step.uses)
            plugin_cls = registry.resolve(step.uses)
            if getattr(plugin_cls, "category", None) not in {"plot", "export"}:
                raise ConfigError(
                    f"report step {step.id}: plugin '{step.uses}' is category '{plugin_cls.category}', "
                    "but reports only allow plot/export plugins."
                )
            cfg = plugin_cls.ConfigModel.model_validate(step.with_ or {})
            plugin = plugin_cls()

            # file: reads are not allowed in reports
            for name, target in (step.reads or {}).items():
                if isinstance(target, str) and target.startswith("file:"):
                    raise ConfigError(f"report step {step.id}: reads.{name} is file:... but reports must use artifacts")

            inputs = _resolve_inputs(store, step.reads or {})
            _assert_input_contracts(plugin, inputs, where=f"{step.id}")

            plug_inputs: dict[str, Any] = {}
            for k, v in inputs.items():
                if hasattr(v, "load_dataframe"):
                    plug_inputs[k] = v.load_dataframe()
                else:
                    plug_inputs[k] = v

            try:
                outputs = plugin.run(ctx, plug_inputs, cfg)
            except ReaderError:
                raise
            except Exception as e:
                raise _wrap_plugin_error(step.id, plugin, e) from e

            _assert_output_contracts(plugin, outputs, where=f"{step.id}")

            # Persist dataframe outputs (report tables) when declared
            for out_name, obj in outputs.items():
                cid = plugin.output_contracts()[out_name]
                if isinstance(obj, pd.DataFrame) and cid != "none":
                    store.persist_dataframe(
                        step_id=step.id,
                        plugin_key=plugin.key if hasattr(plugin, "key") else plugin_cls.__name__,
                        out_name=out_name,
                        df=obj,
                        contract_id=cid,
                        inputs=[
                            inputs[n].label if hasattr(inputs[n], "label") else str(inputs[n])
                            for n in (step.reads or {})
                        ],
                        config_digest=_digest_cfg(cfg),
                    )

            # Record report outputs (files)
            files = outputs.get("files") if isinstance(outputs, dict) else None
            meta = outputs.get("meta") if isinstance(outputs, dict) else None
            report_store.persist_files(
                step_id=step.id,
                plugin_key=plugin.key if hasattr(plugin, "key") else plugin_cls.__name__,
                inputs=[inputs[n].label if hasattr(inputs[n], "label") else str(inputs[n]) for n in (step.reads or {})],
                files=files,
                config_digest=_digest_cfg(cfg),
                meta=meta,
            )

            progress.advance(task)

    console.print(
        Panel.fit(f"✓ Reports done — outputs in [path]{str(out_dir)}[/path]", border_style="green", box=box.ROUNDED)
    )
