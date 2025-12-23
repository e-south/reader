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
        if contract_id == "none":
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


def explain(spec: ReaderSpec, *, console: Console, registry=None) -> None:
    registry = registry or load_entry_points()
    table = Table(
        title="[title]Plan[/title]",
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
    for i, step in enumerate(spec.steps, 1):
        P = registry.resolve(step.uses)
        inp = ", ".join(f"{k}:{v}" for k, v in P.input_contracts().items())
        out = ", ".join(f"{k}:{v}" for k, v in P.output_contracts().items())
        table.add_row(str(i), step.id, step.uses, inp, out)
    console.print(
        Panel(table, border_style="accent", box=box.ROUNDED, subtitle=f"[muted]{len(spec.steps)} steps[/muted]")
    )


def validate(spec: ReaderSpec, *, console: Console) -> None:
    # schema already validated by pydantic; here we ensure contracts exist
    for step in spec.steps:
        uses = step.uses
        try:
            plugin_cls = load_entry_points().resolve(uses)
        except Exception as e:
            raise ConfigError(f"step {step.id}: {e}") from e
        # validate config model
        try:
            plugin_cls.ConfigModel.model_validate(step.with_ or {})
        except Exception as e:
            raise ConfigError(f"step {step.id}: invalid config for {uses}: {e}") from e
    console.print(Panel.fit("✓ Config validated", border_style="ok", box=box.ROUNDED))


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

    palette = exp.get("palette", "colorblind")
    from importlib import import_module

    PaletteBook = None
    if palette is not None:
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
    registry = load_entry_points()

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
        explain(spec, console=console, registry=registry)
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
                raise ExecutionError(f"step {step.id} crashed: {e}") from e

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

    console.print(Panel.fit(f"✓ Done — outputs in [path]{str(out_dir)}[/path]", border_style="ok", box=box.ROUNDED))
