# reader specification

This page is the maintainer-facing map of how `reader` is organized today. Use
it to answer three questions quickly:

1. Where does authored experiment intent live?
2. Which package owns assay semantics versus execution mechanics?
3. Which surfaces are public contract versus internal IR?

## System layers

`reader` stays legible when these layers remain separate:

- Authoring contract:
  [`reader/v7` config](./pipeline.md) in `experiments/<exp>/config.yaml`
- Protocol semantics:
  [`src/reader/protocols/`](../../src/reader/protocols/)
- Experiment-local semantics:
  [`src/reader/workbench/experiment/`](../../src/reader/workbench/experiment/)
- Execution IR and runtime:
  [`src/reader/workbench/decl/`](../../src/reader/workbench/decl/),
  [`src/reader/workbench/graph/`](../../src/reader/workbench/graph/),
  [`src/reader/workbench/engine/`](../../src/reader/workbench/engine/)
- Extension mechanics:
  [`src/reader/plugins/`](../../src/reader/plugins/),
  [`src/reader/contracts/`](../../src/reader/contracts/),
  [`src/reader/plotting/`](../../src/reader/plotting/)
- Operator surfaces:
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
  owns the heavier single-reporter and retron matched-control descriptor
  assembly so family-specific assay detail does not keep accreting in the
  catalog façade.
- [`src/reader/protocols/semantic_coverage.py`](../../src/reader/protocols/semantic_coverage.py)
  owns execution-bound semantic coverage mapping so semantic status/materialized
  record ids do not stay buried inside the step compiler.
- [`src/reader/workbench/config/`](../../src/reader/workbench/config/)
  parses YAML and validates the wire schema only.
- [`src/reader/workbench/decl/build.py`](../../src/reader/workbench/decl/build.py)
  binds a `reader/v7` document to a protocol and produces the compiled
  workbench declaration.
- [`src/reader/workbench/experiment/model.py`](../../src/reader/workbench/experiment/model.py)
  owns experiment-local semantics: protocol binding, annotations, resources,
  layout, and the compiled protocol semantic program.
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
- [`src/reader/plugins/`](../../src/reader/plugins/)
  owns thin execution adapters only.
- [`src/reader/contracts/`](../../src/reader/contracts/)
  owns dataframe contract identities, validation rules, and built-in contract
  catalogs.

## Runtime flow

The canonical path is:

`config -> protocol binding -> compiled semantic program + compiled workbench plan -> graph/runtime execution -> records and file bundles`

More concretely:

1. [`src/reader/workbench/config/load.py`](../../src/reader/workbench/config/load.py)
   loads and validates `reader/v7`.
2. [`src/reader/workbench/decl/build.py`](../../src/reader/workbench/decl/build.py)
   binds the protocol and stores the compiled semantic program on the
   experiment semantics object.
3. [`src/reader/workbench/graph/normalize.py`](../../src/reader/workbench/graph/normalize.py)
   normalizes declarations into runtime nodes and refs.
4. [`src/reader/workbench/engine/`](../../src/reader/workbench/engine/)
   validates inputs, resolves records/resources, and executes the selected
   slice.
5. [`src/reader/workbench/records/store.py`](../../src/reader/workbench/records/store.py)
   persists dataframe records and file-bundle provenance under
   `outputs/manifests/records.json`.

The semantic program is now part of that compiled contract. Inspection surfaces
should read the compiled program snapshot directly instead of reconstructing it
through fallback branches. That removes one major split-ownership path, even
though deeper staleness checks for same-protocol snapshots would still be a
separate hardening step.

## Information architecture rules

- Public config lives in [`docs/core/pipeline.md`](./pipeline.md), not in plugin
  docs.
- Protocols own user-facing output vocabulary such as figures, artifacts, and
  notebook policy.
- Plugins stay mechanical. If a maintainer needs assay meaning to understand a
  plugin, that logic probably belongs in a domain or protocol package instead.
- `inspection/` is presentation-only. It should not recompile or “repair”
  semantic state.
- Generated artifacts live under `outputs/` and are never the source of truth.
- When docs name a code surface, prefer linking to the actual file or package so
  `tools/check_docs.py` can catch drift.

## Current pressure points

The package is no longer suffering from split semantic ownership between
compiled plans and experiment semantics, but two maintainability hotspots
remain:

- Protocol concentration:
  [`src/reader/protocols/builtins.py`](../../src/reader/protocols/builtins.py),
  [`src/reader/protocols/_builtins_plate_reader_variants.py`](../../src/reader/protocols/_builtins_plate_reader_variants.py),
  [`src/reader/protocols/compiler.py`](../../src/reader/protocols/compiler.py),
  [`src/reader/protocols/model.py`](../../src/reader/protocols/model.py), and
  [`src/reader/protocols/semantic_coverage.py`](../../src/reader/protocols/semantic_coverage.py)
  still carry a large share of assay semantics. The plate-reader variants are
  now in a private family helper instead of the public façade, but new assay
  families should keep pushing descriptor, compiler, and semantic-coverage
  logic down into family-specific helpers instead of back into shared catalog
  files.
- Retron notebook concentration:
  [`src/reader/workbench/notebooks/`](../../src/reader/workbench/notebooks/)
  remains the biggest local cluster of large files. Launch preflight/runtime
  state is now split from the planner, but the retron review stack is still the
  highest-risk area for future monolith drift.

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
