from __future__ import annotations

import contextlib
import logging
from importlib import import_module
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from reader.errors import ConfigError
from reader.runtime import ReaderRuntime
from reader.workbench.context import RunContext
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.records import RecordStore

from ._shared import needs_plot_palette


def configure_logger(*, out_dir: Path, log_level: str, verbose: bool, console: Console) -> logging.Logger:
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
        file_handler = logging.FileHandler(out_dir / "reader.log", encoding="utf-8")
    except OSError as err:
        raise ConfigError(f"Cannot write reader.log in outputs directory: {out_dir}") from err
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    rich_handler = RichHandler(
        console=console, markup=True, rich_tracebacks=True, show_level=True, show_time=False, show_path=False
    )
    rich_handler.setLevel(getattr(logging, level_name) if verbose else logging.WARNING)

    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)
    return logger


def resolve_palette_book(*, decl: WorkbenchDecl, steps: list[Any], dry_run: bool) -> Any:
    palette = decl.plotting_palette
    if palette is not None:
        if not isinstance(palette, str) or not palette.strip():
            raise ConfigError("plotting.palette must be a non-empty string or null")
        palette = palette.strip()

    if dry_run or not needs_plot_palette(steps, palette):
        return None

    try:
        mod = import_module("reader.plotting.style")
    except ModuleNotFoundError as err:
        missing = str(err.name or "")
        if missing == "matplotlib" or missing.startswith("matplotlib."):
            raise ConfigError(
                "Plot palettes require matplotlib; install plotting dependencies or set plotting.palette: null."
            ) from err
        raise ConfigError(f"Failed to import plot palette support: {err}") from err
    except Exception as err:
        raise ConfigError(f"Failed to initialize plot palette support: {err}") from err

    palette_book_cls = getattr(mod, "PaletteBook", None)
    available_palettes = getattr(mod, "available_palettes", None)
    if palette_book_cls is None or available_palettes is None:
        raise ConfigError(
            "reader.plotting.style must expose PaletteBook and available_palettes for plotting.palette support."
        )

    if palette not in available_palettes():
        raise ConfigError(
            f"Unknown palette {palette!r}. Available: {available_palettes()} (or set plotting.palette: null)."
        )
    return palette_book_cls(palette)


def build_run_context(
    *,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
    out_dir: Path,
    store: RecordStore,
    logger: logging.Logger,
    palette_book: Any,
) -> RunContext:
    return RunContext(
        exp_dir=decl.experiment.root,
        outputs_dir=out_dir,
        artifacts_dir=store.artifacts_dir,
        plots_dir=store.plots_dir,
        exports_dir=store.exports_dir,
        records_path=store.records_path,
        logger=logger,
        palette_book=palette_book,
        experiment=decl.experiment_semantics,
        protocol=runtime.bind_protocol(decl.experiment_semantics.protocol),
    )


def slice_pipeline_steps(steps: list[Any], *, resume_from: str | None, until: str | None) -> list[Any]:
    selected = list(steps)
    start_index = 0
    if resume_from:
        try:
            start_index = next(i for i, step in enumerate(selected) if step.id == resume_from)
            selected = selected[start_index:]
        except StopIteration:
            raise ConfigError(f"--from: step id '{resume_from}' not found") from None
    if until:
        try:
            until_index = next(i for i, step in enumerate(steps) if step.id == until)
        except StopIteration:
            raise ConfigError(f"--until: step id '{until}' not found") from None
        if resume_from and start_index > until_index:
            raise ConfigError(f"--from '{resume_from}' comes after --until '{until}' in pipeline order")
        relative_index = until_index - start_index
        selected = selected[: relative_index + 1]
    return selected
