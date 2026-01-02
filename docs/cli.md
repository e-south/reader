## reader CLI reference

This is the canonical CLI reference for **reader**. It is intentionally concise and aligned with current behavior.
Examples use `reader`; inside this repo you can run `uv run reader` instead.

Quick links:
- [README](../README.md)
- [Pipeline config + deliverables](./pipeline.md)
- [Notebooks](./notebooks.md)
- [Spec / architecture](./spec.md)

---

### Conventions

**CONFIG|DIR|INDEX**
- Most commands accept a config path, an experiment directory, or a numeric index from `reader ls`.
- If omitted, the CLI searches upward from the current directory for `config.yaml`.

**Journal entries**
- `reader explain`, `reader validate`, `reader run`, `reader deliverables`, and `reader run-step` append a timestamped entry to `JOURNAL.md` in the experiment directory.

**Output style**
- The CLI uses rich tables and panels. Example outputs below are shape‑only; actual content varies.

---

## Commands

### `reader ls`
List experiments under a root (defaults to nearest `./experiments`).

When to use: discover configs and get an index for other commands.

```bash
reader ls --root experiments
```

Example output (shape only):

```text
Experiments
#  Name                            Outputs
1  20250512_sensor_panel_M9_glu     ✓
2  20250620_sensor_panel_crosstalk  —
```

---

### `reader presets [NAME]`
List preset bundles, or describe a specific preset.

```bash
reader presets
reader presets plate_reader/synergy_h1
```

---

### `reader steps CONFIG|DIR|INDEX`
List pipeline step ids and plugins for a config.

```bash
reader steps 1
```

---

### `reader explain CONFIG|DIR|INDEX`
Show the planned steps and contracts (no execution).

```bash
reader explain experiments/my_experiment/config.yaml
```

---

### `reader validate CONFIG|DIR|INDEX`
Validate config and plugin params (no data I/O).

```bash
reader validate 1
```

Checks:
- schema + presets/overrides expansion
- plugin availability and config
- `reads` keys match plugin inputs
- `reads` labels reference prior outputs (or `file:`)
- output labels are unique

---

### `reader config CONFIG|DIR|INDEX`
Print the expanded config (presets + overrides applied).

```bash
reader config 1
reader config 1 --format json
```

Notes:
- Paths are resolved relative to the config; output shows absolute paths.
- Includes `experiment.root` (internal).

---

### `reader check-inputs CONFIG|DIR|INDEX`
Verify that `reads: file:` inputs exist.

```bash
reader check-inputs 1
```

If missing, the CLI prints which step/input is missing and exits non‑zero.

---

### `reader run CONFIG|DIR|INDEX`
Run the pipeline. Deliverables run by default, unless you disable them.

```bash
reader run 1
reader run 1 --no-deliverables
```

Key options:
- `--step <ID|INDEX>` run exactly one step (sugar for `--resume-from/--until`).
- `--resume-from <ID>` start from this step (inclusive).
- `--until <ID>` stop after this step (inclusive).
- `--dry-run` plan only (no execution).
- `--log-level <LEVEL>` set logging verbosity.
- `--deliverables/--no-deliverables` toggle deliverables.

Notes:
- If you use `--resume-from` or `--until` and do not pass a deliverables flag, deliverables are **not** run by default.
- `reader run 14 step 11` and `reader run 14 11` are accepted shorthands.

---

### `reader deliverables CONFIG|DIR|INDEX`
Render deliverables (plots/exports) using existing artifacts.

```bash
reader deliverables 1
reader deliverables 1 --list
```

Options:
- `--list` list deliverable steps and exit.
- `--dry-run` plan only.
- `--log-level <LEVEL>` set logging verbosity.

---

### `reader run-step STEP --config CONFIG|DIR|INDEX`
Run a single step using existing artifacts (no prior steps).

```bash
reader run-step ratio_yfp_od600 --config 1
```

Alias:
- `reader step` is the same command.

---

### `reader artifacts CONFIG|DIR|INDEX`
List emitted artifacts from `outputs/manifest.json`.

```bash
reader artifacts 1
reader artifacts 1 --all
```

Notes:
- Requires `outputs/manifest.json` (created by `reader run`).

`--all` shows revision history counts instead of latest entries.

---

### `reader plugins [--category NAME]`
List discovered plugins; optionally filter by category.

```bash
reader plugins
reader plugins --category plot
```

---

### `reader explore CONFIG|DIR|INDEX`
Scaffold a marimo notebook (no execution).

```bash
reader explore 1 --preset eda/microplate
reader explore 1 --list-presets
```

Options:
- `--name <FILE>` output filename (default: `eda.py`).
- `--preset <NAME>` notebook preset (use `--list-presets`).
- `--force` overwrite existing notebook.

---

### `reader demo`
Show a quick guided walkthrough of common commands.

```bash
reader demo
```

Example output (shape only):

```text
Reader Demo
#  Goal                           Command
1  Find experiments               reader ls
2  List presets                   reader presets
3  Explain plan                   reader explain 1
4  Validate config                reader validate 1
5  Check file inputs              reader check-inputs 1
6  Run pipeline + deliverables    reader run 1
7  Deliverables only              reader deliverables 1
8  List deliverable steps         reader deliverables --list 1
9  Notebook scaffold (marimo)     reader explore 1
10 See artifacts                  reader artifacts 1
```

---

### `reader --help`
Show the full CLI help with all flags and defaults.

```bash
reader --help
```

---

@e-south
