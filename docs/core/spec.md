---
doc_id: reader-package-spec
surface: architecture-reference
owner: reader-maintainers
last_verified: 2026-07-29
summary: Detailed package map and implementation contracts beneath the top-level Reader architecture.
---

# reader specification

This page is the maintainer-facing map of how `reader` is organized today. Use
it to answer three questions quickly:

1. Where does authored experiment intent live?
2. Which package owns assay semantics versus execution mechanics?
3. Which surfaces are public contract versus internal IR?

## System layers

`reader` stays legible when these layers remain separate:

- Authoring contract:
  [`reader/v8` config](./pipeline.md) in `experiments/<exp>/config.yaml`
- Protocol semantics:
  [`src/reader/protocols/`](../../src/reader/protocols/)
- Experiment-local semantics:
  [`src/reader/workbench/experiment/`](../../src/reader/workbench/experiment/)
- Data Operations Plan overlay:
  [`src/reader/workbench/dop/`](../../src/reader/workbench/dop/)
- Execution IR and runtime:
  [`src/reader/workbench/decl/`](../../src/reader/workbench/decl/),
  [`src/reader/workbench/graph/`](../../src/reader/workbench/graph/),
  [`src/reader/workbench/engine/`](../../src/reader/workbench/engine/)
- Extension mechanics:
  [`src/reader/plugins/`](../../src/reader/plugins/),
  [`src/reader/contracts/`](../../src/reader/contracts/),
  [`src/reader/plotting/`](../../src/reader/plotting/)
- Runtime composition:
  [`src/reader/runtime/`](../../src/reader/runtime/)
- Operator surfaces:
  [`src/reader/api/`](../../src/reader/api/),
  [`src/reader/workbench/cli/`](../../src/reader/workbench/cli/),
  [`docs/guides/preflight_run_verify.md`](../guides/preflight_run_verify.md),
  experiment `outputs/`

The important rule is that authored config should name assays, inputs,
analysis choices, and requested outputs in domain terms. It should not mirror
plugin wiring or internal graph structure.

## Package ownership

- [`src/reader/protocols/model.py`](../../src/reader/protocols/model.py)
  owns the typed protocol contract: config fields, plots, artifacts, semantic
  nodes, notebook policy, and the compiled semantic program contract.
- [`src/reader/protocols/compiler.py`](../../src/reader/protocols/compiler.py)
  owns protocol-specific compilation from bound protocol config into pipeline,
  plots, exports, notebooks, and step assembly.
- [`src/reader/protocols/builtins.py`](../../src/reader/protocols/builtins.py)
  remains the public builtin catalog surface, while
  [`src/reader/protocols/_builtins_plate_reader_variants.py`](../../src/reader/protocols/_builtins_plate_reader_variants.py)
  owns the single-reporter descriptor assembly so family-specific detail does
  not keep accreting in the catalog facade.
- [`src/reader/protocols/semantic_coverage.py`](../../src/reader/protocols/semantic_coverage.py)
  owns execution-bound semantic coverage mapping so semantic status/materialized
  record ids do not stay buried inside the step compiler.
- [`src/reader/workbench/config/`](../../src/reader/workbench/config/)
  parses YAML and validates the wire schema only.
- [`src/reader/workbench/decl/build.py`](../../src/reader/workbench/decl/build.py)
  binds a `reader/v8` document to a protocol and produces the compiled
  workbench declaration.
- [`src/reader/workbench/experiments.py`](../../src/reader/workbench/experiments.py)
  resolves experiment identities beneath the canonical `experiments/` owner
  without encoding year or folder-name conventions.
- [`src/reader/workbench/experiment/model.py`](../../src/reader/workbench/experiment/model.py)
  owns experiment-local semantics: protocol binding, annotations, resources,
  layout, ordered state-space resolution, and the compiled protocol semantic
  program.
- [`src/reader/workbench/dop/`](../../src/reader/workbench/dop/)
  owns the read-only Data Operations Plan overlay: data-class selection,
  metadata minimums, stop conditions, transfer rules, and readiness evidence
  gates. It references protocol ids but does not own protocol execution.
- [`src/reader/workbench/inspection/`](../../src/reader/workbench/inspection/)
  owns read-only payloads and reports for `inspect`, `steps`, `records`, and
  related CLI surfaces.
- [`src/reader/workbench/assets/plugin_manifest.py`](../../src/reader/workbench/assets/plugin_manifest.py)
  is the explicit built-in plugin registry.
- [`src/reader/workbench/templates/catalog.py`](../../src/reader/workbench/templates/catalog.py)
  owns notebook template selection and compatibility checks.
- [`src/reader/workbench/notebooks/launch.py`](../../src/reader/workbench/notebooks/launch.py)
  owns Marimo launch orchestration, while
  [`src/reader/workbench/notebooks/_launch_runtime.py`](../../src/reader/workbench/notebooks/_launch_runtime.py)
  and
  [`src/reader/workbench/notebooks/_launch_registry.py`](../../src/reader/workbench/notebooks/_launch_registry.py)
  keep runtime-path/env setup and managed-session state separate from the
  planner itself.
- [`src/reader/domains/`](../../src/reader/domains/)
  owns domain math, parsing, ordering, and figure-planning logic.
- [`src/reader/domains/plate_reader/analysis/response_window/`](../../src/reader/domains/plate_reader/analysis/response_window/)
  owns response-window contracts and calculations.
- [`src/reader/domains/plate_reader/plots/response_window/`](../../src/reader/domains/plate_reader/plots/response_window/)
  owns response-window-specific review tables and figures.
- [`src/reader/plugins/`](../../src/reader/plugins/)
  owns thin execution adapters only.
- [`src/reader/workbench/records/sources.py`](../../src/reader/workbench/records/sources.py)
  resolves exact source-record revisions for generic record-collection ports;
  domain packages never open another experiment's catalog.
- [`src/reader/contracts/`](../../src/reader/contracts/)
  owns dataframe contract identities, validation rules, and built-in contract
  catalogs.
- [`src/reader/runtime/`](../../src/reader/runtime/)
  owns built-in composition and adapters that resolve workbench state before
  calling domain operations. Domain packages never resolve configs or record
  catalogs themselves.
- [`src/reader/api/`](../../src/reader/api/)
  owns the stable task-oriented Python surface. It delegates to the same
  declaration, engine, and verification paths as the CLI.
- [`src/reader/maintenance/`](../../src/reader/maintenance/)
  owns repository documentation and skill checks exposed through
  `reader maintain`; it is not part of experiment execution.

## Runtime flow

The canonical path is:

`config -> protocol binding -> compiled semantic program + compiled workbench plan -> graph/runtime execution -> records and file bundles`

More concretely:

1. [`src/reader/workbench/config/load.py`](../../src/reader/workbench/config/load.py)
   loads and validates `reader/v8`.
2. [`src/reader/workbench/decl/build.py`](../../src/reader/workbench/decl/build.py)
   binds the protocol and stores the compiled semantic program on the
   experiment semantics object.
3. [`src/reader/workbench/graph/normalize.py`](../../src/reader/workbench/graph/normalize.py)
   normalizes declarations into runtime nodes and refs.
4. [`src/reader/workbench/engine/`](../../src/reader/workbench/engine/)
   validates files and exact source-record revisions before output mutation,
   then executes the selected slice.
5. [`src/reader/workbench/records/store.py`](../../src/reader/workbench/records/store.py)
   persists dataframe records and file-bundle provenance under
   `outputs/manifests/records.json`.

The semantic program is part of that compiled contract. Its compiled snapshot
is authoritative. Inspection surfaces read it directly and do not reconstruct
semantic state through alternate branches. Same-protocol snapshot staleness is a
separate hardening concern.

## Information architecture rules

- Public config lives in [`docs/core/pipeline.md`](./pipeline.md), not in plugin
  docs.
- Protocols own user-facing output vocabulary such as figures, artifacts, and
  notebook policy.
- Plugins stay mechanical. If a maintainer needs assay meaning to understand a
  plugin, that logic probably belongs in a domain or protocol package instead.
- Domains accept explicit data and parameters. Config loading, record lookup,
  and runtime composition stay outside `domains/`.
- `inspection/` is presentation-only. It should not recompile or “repair”
  semantic state.
- Generated artifacts live under `outputs/` and are never the source of truth.
- Cross-experiment work is another experiment. It declares `record` resources,
  compiles through a protocol, and persists through the same `RecordStore` as
  every other run.
- Ordered state spaces describe record identity only. Target masks, metric
  formulas, and calibration stay outside Reader experiment annotations.
- When docs name a code surface, prefer linking to the actual file or package so
  `reader maintain docs` can catch drift.

## Current pressure points

The canonical, changing inventory lives in
[Architecture](../../ARCHITECTURE.md#current-architecture-pressure). This page
intentionally does not duplicate it. Use the package map above to locate an
owner, then move one coherent responsibility behind an explicit contract;
line count alone is not a reason to split a module.

## Dependency management

This repo uses `uv`.

```bash
uv sync --locked --group dev --group notebooks
uv run pytest -q
uv run ruff check .
uv run ruff format . --check
```

For the maintainer gate and docs integrity loop, see
[Repo maintenance](../repo-maintenance.md),
[Quality](../../QUALITY.md), and
[Reliability](../../RELIABILITY.md).
