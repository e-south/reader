# reader specification

This document is the developer‑oriented source of truth for how **reader** is structured, how configs map to execution, and how dependencies are managed.

---

### Scope

- **Experiment directory** = unit of work.
- **Pipeline steps** produce dataframe records; **plot/export specs** render file outputs tracked in the record catalog.
- **Notebooks** are optional and read outputs for interactive exploration.

---

### Repo layout

```text
reader/
  experiments/          # workbench directories (inputs, notebooks, outputs)
  docs/                 # documentation (index + grouped references)
    index.md            # docs map
    core/               # core reference (CLI, pipeline, plugins, spec)
    guides/             # how-to + walkthroughs
    lib/                # library-level references
    audits/             # audits and investigations
  src/reader/            # library + CLI
    protocols/          # explicit experiment analysis protocol kernel
    workbench/          # experiment lifecycle, config, decl/graph IR, records, notebooks, CLI
      assets/           # unified asset registry + capability model for plugins/templates
      config/           # wire schema + YAML loading only
      decl/             # compiled declaration IR
      experiment/       # typed experiment-local semantics (protocol binding, annotations, resources, output layout)
      engine/           # planning, validation, contracts, runtime execution
      graph/            # runtime graph nodes and typed refs
      ports/            # typed plugin input/output port ontology
      ontology.py       # shared workbench semantic types
      notebooks/        # notebook scaffold + launch flows
      records/          # record catalog store + dataset discovery helpers
    domains/            # protocol/data semantics by domain
      plate_reader/
        analysis/       # derived plate-reader summary logic (e.g. fold_change)
        ordering.py     # dose/treatment ordering semantics
        io/             # Synergy H1 parsing
        plots/          # plate-reader plotting primitives and figure builders
      cytometry/
        io/             # FCS parsing
      logic/
        sfxi/           # SFXI math, selection, reference handling, writer
        logic_symmetry/ # logic-symmetry plotting/metrics helpers
        crosstalk/      # pairwise crosstalk ranking helpers
    contracts/          # explicit dataframe contract kernel
      builtins/         # built-in contract declarations by semantic domain
    plugins/            # ingest/transform/plot/export/validator
    plotting/           # shared plotting/style/cache infrastructure
    tests/
```

---

### Contracts

Plugins declare input/output contracts (schema identifiers). The engine:
- asserts required inputs are present
- validates declared outputs
- fails fast on mismatches (unless runtime strictness is relaxed, in which case mismatches are logged as warnings)

Built‑in contracts now live entirely under `src/reader/contracts/`.
The contract ontology is explicit and centralized:

- `src/reader/contracts/model.py` defines contract identity and dataframe rules
- `src/reader/contracts/catalog.py` owns `ContractCatalog` and lineage checks
- `src/reader/contracts/builtins/` owns built-in declarations for:
  - `generic`
  - `plate_reader`
  - `logic`
  - `cytometry`
- `src/reader/contracts/__init__.py` exports the explicit built-in catalog
  constructor `builtin_contract_catalog()`

`domains/` no longer declares built-in dataframe contracts. Domain packages now
own algorithms, IO, and semantics only.

The workbench engine is now organized as a package instead of a monolithic module:

- `workbench/engine/planning.py` owns explain/plan rendering
- `workbench/engine/validation.py` owns config/reference checks
- `workbench/engine/contracts.py` owns runtime contract enforcement
- `workbench/engine/inputs.py` owns dataframe-record/file input resolution
- `workbench/engine/runtime.py` owns execution orchestration

That split keeps plan-time semantics, runtime semantics, and filesystem concerns orthogonal.

The workbench asset surface now follows one model:

- `workbench/assets/` is the single semantic registry surface for plugins and
  notebook templates
- `workbench/registry.py` owns executable plugin discovery only; plugin assets
  are exposed through the shared asset model
- `workbench/decl/` owns the internal authored declaration layer for bound
  experiments, recipe-expanded step declarations, and notebook template calls
- `workbench/experiment/` owns experiment-local semantics:
  explicit protocol binding, typed annotation vocabulary, explicit resource catalogs, and output layout
- `protocols/` owns built-in experiment analysis protocols and the typed
  `ProtocolCatalog` used by runtime composition
- `workbench/graph/` owns typed workbench references and normalized runtime
  nodes:
  `AssetRef`, `InputRef`, `OutputRef`, plugin-step nodes, notebook-template
  calls, and typed `source_recipe` provenance for recipe-expanded steps
- `workbench/ports/` owns typed plugin I/O semantics:
  input/output port names, optionality, port kind, and dataframe-contract
  attachment
- `workbench/config/` is wire-schema parsing only; it no longer doubles as the
  internal authored model or the runtime graph model
- `workbench/records/model.py` owns persisted artifact provenance types instead
  of opaque input strings
- `workbench/templates/builtins/*` are static template assets
- `workbench/recipes/*` are internal workflow macros used by protocol
  compilers, not user-facing config surfaces
- `workbench/model/` was deleted; the remaining semantic types now live under
  `workbench/ontology.py`, `workbench/assets/`, `workbench/decl/`, and
  `workbench/graph/`
- operator behavior such as notebook auto-pick and protocol-level plugin
  defaults now comes from protocol execution plans instead of heuristic CLI
  branches or repeated per-step config blobs
- template-local capabilities still own template-specific behavior such as plot
  filtering or injected plot specs, but protocol policy now owns which
  templates are valid by default for a bound experiment

The plate-reader plotting library now follows the domain ontology directly:

- `domains/plate_reader/analysis/fold_change.py` owns fold-change table construction
- `domains/plate_reader/analysis/timepoints.py` owns nearest-time and snapshot selection helpers
- `domains/plate_reader/ordering.py` owns dose/treatment ordering semantics
- `domains/plate_reader/io/sample_map.py` for plate-map parsing
- `domains/plate_reader/plots/common.py` owns plot-shared dataframe/layout/color/output helpers
- `domains/plate_reader/plots/grouping.py` owns figure-group resolution helpers
- `domains/plate_reader/plots/panels/` for axes-level drawing primitives
- figure-specific packages such as `domains/plate_reader/plots/snapshot_barplot/`
  and `domains/plate_reader/plots/snapshot_heatmap/` for figure planning plus
  render orchestration

The logic domain now follows the same rule:

- `domains/semantics.py` owns the canonical plugin domain vocabulary
- `domains/logic/sfxi/` owns vec8 config parsing, selection, math, and output writing
- `domains/logic/logic_symmetry/` owns logic-symmetry preparation, metrics, overlays, and rendering
- `domains/logic/crosstalk/` owns pairwise crosstalk ranking logic

The cytometry domain now follows the same rule:

- `domains/cytometry/io/` owns raw FCS parsing

Raw ingest autodiscovery no longer lives under `workbench/`.
That policy now lives with ingest adapters:

- `plugins/ingest/discovery_policy.py` owns raw-file auto-discovery defaults
  and file search helpers

The shared plotting infrastructure now lives under `plotting/` instead of any domain:

- `plotting/style.py` owns palettes and shared figure construction helpers
- `plotting/mpl.py` owns Matplotlib cache setup and rc defaults

---

### Matplotlib cache

Plotting plugins require a writable Matplotlib cache directory. `reader` sets
`MPLCONFIGDIR` automatically when plotting is needed.

Defaults:
- Commands that resolve a config/experiment (run/explain/validate/plot/export) use
  `<paths.outputs>/.cache/matplotlib`.
- Other commands that load plot plugins without a config (e.g., `reader plugins`)
  use `$XDG_CACHE_HOME/reader/matplotlib` (or `~/.cache/reader/matplotlib`).

Override with `MPLCONFIGDIR` or `READER_MPLCONFIGDIR` if you need a custom path.

---

### Dependency management (uv)

This repo uses **uv**:

```bash
uv sync --locked
```

Developer tooling (lint + tests + notebooks):

```bash
uv sync --locked --group dev --group notebooks
uv run ruff check .
uv run pytest -q
uv run pytest -q -m smoke
uv run pytest -q -m integration
```

The default `uv run pytest -q` lane excludes `integration` tests so local feedback stays fast while still keeping a few real temp-copy smoke runs in the default suite.

Add/remove dependencies:

```bash
uv add <package>
uv add --group dev <package>
uv remove <package>
```

If you edit `pyproject.toml` manually, regenerate the lockfile:

```bash
uv lock
```

---

### Upgrading dependencies

To upgrade a pinned package:

```bash
uv sync --upgrade-package <name>
```

Commit `pyproject.toml` and `uv.lock` together.

---

@e-south
