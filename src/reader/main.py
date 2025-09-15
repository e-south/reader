"""
--------------------------------------------------------------------------------
<reader project>
reader/main.py

Top-level CLI for the reader pipeline with experiment shortcuts.

    1.  YAML → ReaderCfg (pydantic validation)
    2.  parser → raw DataFrame
    3.  merge sample-map (metadata)
    4.  ordered transformations / feature-engineering
    5.  optional fold-change
    6.  plotting

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from reader.utils.plot_style import PaletteBook
from reader.utils.prune_config import _prune_empty_config_elements 
from reader.config import ReaderCfg, CustomParameter, normalize_plot_params
from reader.parsers.raw import ensure_all_parsers_imported, get_raw_parser
from reader.parsers import parse_plate_map
from reader.processors.custom_params import apply_custom_parameters
from reader.processors.fold_change import apply_fold_change
from reader.utils.logging import setup_logging

LOG = logging.getLogger(__name__)


# CONFIG LOADING
def _abs_path(base: Path, p: str | Path) -> str:
    """Return *absolute* string for a path relative to *base*."""
    return str((base / Path(p)).resolve())


def _resolve_paths_inplace(data: Dict[str, Any], base: Path) -> None:
    """Convert all path strings inside the raw dict to absolute paths."""
    raw = data["data"]["raw"]
    if isinstance(raw, list):
        data["data"]["raw"] = [_abs_path(base, p) for p in raw]
    else:
        data["data"]["raw"] = _abs_path(base, raw)

    if "sample_map" in data["data"]:
        data["data"]["sample_map"] = _abs_path(base, data["data"]["sample_map"])

    data["output"]["dir"] = _abs_path(base, data["output"]["dir"])


def load_cfg(path: Path) -> ReaderCfg:
    cfg_dict = yaml.safe_load(path.read_text())
    _resolve_paths_inplace(cfg_dict, path.parent)
    cfg = ReaderCfg.model_validate(cfg_dict)
    LOG.info("✓ loaded config from %s", path)
    return cfg


# RAW PARSING & MERGE
def parse_raw(cfg: ReaderCfg):
    ensure_all_parsers_imported()
    Parser = get_raw_parser(cfg.parsing.parser)

    parser = Parser(
        path=cfg.raw_files[0],
        channels=cfg.channels,
        sheet_names=cfg.parsing.sheet.names,
        add_sheet=cfg.parsing.sheet.add_column,
        **cfg.parsing.extra,
    )
    df = parser.parse()
    LOG.info("✓ parsed %d rows", len(df))
    return df, parser


def merge_sample_map(raw_df, parser, smap: Optional[Path]):
    if smap is None:
        return raw_df
    meta = parse_plate_map(smap)
    merged = parser.merge(raw_df, meta)
    LOG.info("✓ merged with sample map → %d rows", len(merged))
    return merged


# TRANSFORMATIONS
def _ratios_as_dicts(cfg: ReaderCfg) -> List[CustomParameter]:
    """Create `CustomParameter` instances for ratio transforms."""
    return [
        CustomParameter(                                   # ← construct objects
            name=xf.name,
            type="ratio",
            parameters=[xf.numerator, xf.denominator],     # type: ignore[attr-defined]
        )
        for xf in cfg.transformations
        if xf.type == "ratio"
    ]


def apply_transforms(df, cfg: ReaderCfg):
    blank = cfg.xform("blank_correction")
    overflow = cfg.xform("overflow_handling")

    tidy, blanks = apply_custom_parameters(
        df,
        blank_correction=getattr(blank, "method", "disregard") if blank else "disregard",
        overflow_action=getattr(overflow, "action", "max") if overflow else "max",
        custom_parameters=_ratios_as_dicts(cfg),
        outlier_filter=False,
    )
    LOG.info("✓ transformations complete (%d rows tidy)", len(tidy))
    return tidy, blanks


# COSMETIC RENAMES
def rename_genotypes(df, cfg: ReaderCfg):
    return df.replace({"genotype": cfg.naming.map}) if cfg.naming.use_short else df


# PLOTTING
def _merge_fig(base: Dict[str, Any], override: Optional[Dict[str, Any]]):
    merged = deepcopy(base)
    if override:
        merged.update(override)
    return merged


def generate_plots(df, blanks, cfg: ReaderCfg):
    dft = cfg.plotting.defaults
    out = cfg.output.dir
    palette_book = PaletteBook(dft.fig.get("palette", "colorblind"))

    for spec in cfg.plotting.plots:
        LOG.info("→ plotting %s", spec.name)
        mod = import_module(f"reader.plotters.{spec.module}")
        plot_fn = getattr(mod, f"plot_{spec.module}")

        # Validate/normalize module params (typed for logic_symmetry)
        kws = normalize_plot_params(spec)

        # Only attach common defaults when the plot actually expects them.
        # (Many plots do, but 'logic_symmetry' does not need channels/groups.)
        if "channels" not in kws:
            kws.setdefault("channels", dft.channels)
        if "groups" not in kws:
            kws.setdefault("groups",   spec.groups or dft.groups)

        kws["palette_book"] = palette_book

        plot_fn(
            df=df,
            blanks=blanks,
            output_dir=out,
            **kws,
            fig_kwargs=_merge_fig(dft.fig, spec.fig),
            subplots=spec.subplots,
            filename=spec.filename,
            iterate_genotypes=spec.iterate_genotypes,
        )
    LOG.info("✓ all plots written")


# PIPELINE
def run_pipeline(cfg_path: Path):
    cfg = load_cfg(cfg_path)
    setup_logging(cfg.output.dir)

    raw, parser   = parse_raw(cfg)
    merged        = merge_sample_map(raw, parser, cfg.data.sample_map)
    
    renamed       = rename_genotypes(merged, cfg)
    tidy, blanks  = apply_transforms(renamed, cfg)
    
    _prune_empty_config_elements(tidy, cfg)

    tidy.to_csv(cfg.output.dir / "tidy_data.csv", index=False)
    
    # (Optional) sfxi_ingest_prep (if configured in yaml)
    sfxi_cfg = cfg.xform("sfxi_ingest_prep")
    if sfxi_cfg:
        from reader.processors.sfxi_ingest_prep import run_sfxi_ingest_prep
        run_sfxi_ingest_prep(tidy, sfxi_cfg, cfg.output.dir)
    
    fc_cfg = cfg.xform("fold_change")
    if fc_cfg:
        tidy = apply_fold_change(tidy, fc_cfg, out_dir=cfg.output.dir)
        
    generate_plots(tidy, blanks, cfg)

    print(f"🎉 done → outputs in {cfg.output.dir}")


# CLI
def _cli() -> argparse.Namespace:
    """
    Parse CLI arguments.

    • 0 positional args  → interactive prompt if >1 experiment present  
    • 1 positional arg   → treat as prefix **or** explicit config path  
    • --cfg              → explicit path overrides everything
    """
    parser = argparse.ArgumentParser(prog="reader")
    parser.add_argument(
        "id",
        nargs="?",
        default=None,
        help="experiment prefix (e.g. 1, 001) or path/to/config.yaml",
    )
    parser.add_argument(
        "--exp-dir",
        default="experiments",
        help="root experiments folder (default: ./experiments)",
    )
    parser.add_argument(
        "--cfg",
        help="explicit config path (overrides id/exp-dir)",
    )
    return parser.parse_args()


def _list_experiments(root: Path) -> List[Path]:
    """Return sorted list of experiment directories in *root*."""
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    

def _pick_interactively(candidates: List[Path]) -> Path:
    """Ask the user to choose one of candidates and return its config path."""
    print("\nAvailable experiments:")
    for idx, d in enumerate(candidates, 1):
        print(f"  [{idx}] {d.name}")
    choice = input("Select by number or type a prefix: ").strip()
    # numeric?
    if choice.isdigit():
        i = int(choice) - 1
        if 0 <= i < len(candidates):
            return candidates[i] / "config.yaml"
        raise SystemExit("Invalid selection index")
    # treat as prefix
    matches = [d for d in candidates if d.name.startswith(choice)]
    if len(matches) == 1:
        return matches[0] / "config.yaml"
    raise SystemExit("Could not uniquely resolve that prefix.")


def _resolve_cfg(args: argparse.Namespace) -> Path:
    """
    Decide which config.yaml to load.

    Resolution order
    ----------------
    1. --cfg <path>                        (highest priority)
    2. id is a direct file path
    3. id is an experiment prefix in --exp-dir
    4. No id → prompt user to choose from available experiments
    """
    # 1. explicit path
    if args.cfg:
        return Path(args.cfg)

    # 2. if id present and is a direct file
    if args.id:
        maybe = Path(args.id)
        if maybe.is_file():
            return maybe

    exp_root = Path(args.exp_dir)
    candidates = _list_experiments(exp_root)

    # 3. id given → prefix match inside exp_root
    if args.id:
        matches = [d for d in candidates if d.name.startswith(args.id)]
        if len(matches) == 1:
            return matches[0] / "config.yaml"
        if not matches:
            raise SystemExit(f"No experiment matches prefix '{args.id}'")
        # fall through to interactive if ambiguous
        print(f"Prefix '{args.id}' is ambiguous.")
        return _pick_interactively(matches)

    # 4. no id at all → interactive choice (unless only one dir)
    if len(candidates) == 1:
        return candidates[0] / "config.yaml"
    return _pick_interactively(candidates)


def main() -> None:
    args = _cli()
    try:
        cfg_path = _resolve_cfg(args)
        run_pipeline(cfg_path)
    except Exception as exc:                       # pragma: no cover
        LOG.exception("Pipeline failed")
        print("❌", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
