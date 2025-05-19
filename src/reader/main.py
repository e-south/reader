"""
--------------------------------------------------------------------------------
<reader project>
reader/main.py

Top-level CLI for the reader pipeline with experiment shortcuts.

Run with:
  reader list                       # show available experiments
  reader run <id-or-path>           # run by numeric prefix or full config path
  reader run --exp-dir <dir> <id>   # override experiments root

Pipeline stages
---------------
1. load_config - YAML ➜ `ReaderConfig`
2. init_runtime - create output dir, start logging
3. parse_raw - dispatch via the raw-parser registry
4. parse_plate_map - simple CSV/XLSX helper
5. merge_frames - inner-join on `position`
6. apply_custom_params - blank correction / overflow handling / OD600 normalization
7. write_outputs - tidy CSV
8. plotting - dispatch to reader.plotters.*

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import argparse
import inspect
import logging
import sys
from pathlib import Path
import yaml
from importlib import import_module

from reader.config import ReaderConfig
from reader.parsers.raw import get_raw_parser, ensure_all_parsers_imported
from reader.parsers import parse_plate_map
from reader.processors.custom_params import apply_custom_parameters
from reader.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def load_config(path: Path) -> ReaderConfig:
    data = yaml.safe_load(path.read_text())
    cfg_dir = path.parent
    for key in ("raw_data", "plate_map", "output_dir"):
        if key in data:
            data[key] = str((cfg_dir / data[key]).resolve())
    cfg = ReaderConfig.model_validate(data)
    logger.info("Loaded config from %s", path)
    return cfg

def run_experiment(config_path: Path):
    cfg = load_config(config_path)
    setup_logging(cfg.output_dir)
    logger.info("Starting experiment run")

    # ── parse raw data ──
    ensure_all_parsers_imported()
    Parser = get_raw_parser(cfg.data_parser)

    parser = Parser(
        path=cfg.raw_data,
        channels=cfg.parameters,
        sheet_names=cfg.sheet_names,
        add_sheet=cfg.add_sheet_column
    )
    raw_df = parser.parse()

    if raw_df.duplicated(subset=["position", "time", "channel"], keep=False).any():
        logger.warning("Duplicates detected in raw parser output")
    else:
        logger.info("No duplicates in raw parser output")

    # ── plate map & merge ──
    map_df = parse_plate_map(cfg.plate_map)
    merged = parser.merge(raw_df, map_df)

    # ── genotype naming ──
    if cfg.use_short_genotype_names:
        mapping = {}
        for m in cfg.short_genotype_names:
            mapping.update(m)
        merged["genotype"] = merged["genotype"].replace(mapping)

    # ── QC & custom params ──
    tidy, blanks = apply_custom_parameters(
        merged,
        blank_correction=cfg.blank_correction,
        overflow_action=cfg.overflow_action,
        outlier_filter=False,
        custom_parameters=cfg.custom_parameters,
    )

    tidy.to_csv(Path(cfg.output_dir) / "tidy_data.csv", index=False)
    logger.info("Tidy data written, blanks=%d rows", len(blanks))

    # ── plotting ──
    for spec in cfg.plots:
        logger.info("Generating plot '%s' ▶ module=%s", spec.name, spec.module)
        module_name = Path(spec.module).stem
        mod = import_module(f"reader.plotters.{module_name}")
        fn = getattr(mod, f"plot_{module_name}")

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        call_kwargs = {
            "df": tidy,
            "blanks": blanks,
            "output_dir": out_dir,
            **spec.params,
            "subplots": spec.subplots,
            "groups": spec.groups,
            "iterate_genotypes": spec.iterate_genotypes,
            "fig_kwargs": spec.fig or {},
            "filename": spec.filename,
        }
        sig = inspect.signature(fn)
        filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
        fn(**filtered_kwargs)

    print("✨ All done! Outputs in", cfg.output_dir)

def main():
    p = argparse.ArgumentParser(prog="reader")
    p.add_argument("id", nargs="?", help="experiment prefix or config path")
    p.add_argument("--exp-dir", default="experiments", help="root experiments folder")
    args = p.parse_args()

    exp_root = Path(args.exp_dir)
    if not args.id:
        dirs = [d for d in exp_root.iterdir() if d.is_dir()]
        args.id = dirs[0].name if len(dirs) == 1 else input("Experiment prefix or config path: ").strip()

    try:
        cfg_path = Path(args.id)
        if not cfg_path.is_file():
            matches = [d for d in exp_root.iterdir() if d.is_dir() and d.name.startswith(args.id)]
            if len(matches) == 1:
                cfg_path = matches[0] / "config.yaml"
            else:
                raise ValueError(f"No unique experiment for '{args.id}'")
        run_experiment(cfg_path)
    except Exception as e:
        print("❌", e)
        sys.exit(1)

if __name__ == "__main__":
    main()