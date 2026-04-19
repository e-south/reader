# Architecture

`reader` is a protocol-driven experimental workbench. The architecture is built around one constraint: experiment authors should declare assay intent once in `config.yaml`, while the runtime stays deterministic, inspectable, and extensible for maintainers.

This file is the top-level map. Use it to understand the layers, ownership boundaries, and extension points before dropping into package-level docs.

## System Model

`reader` has four layers.

1. Authoring model
   Human-facing `reader/v7` YAML in `experiments/<exp>/config.yaml`.
   This is the public contract for users.
2. Semantic model
   Protocols, annotations, resources, and experiment semantics.
   This layer defines what an assay means.
3. Execution model
   Compiled declarations, graph refs, runtime engine, ports, contracts, and records.
   This layer defines how the work gets executed.
4. Workbench surface
   The experiment tree, CLI, notebooks, and generated outputs.
   This layer is how humans and agents inspect and operate the system.

The important rule is that these layers are not interchangeable. User config should not mirror plugin internals, and runtime plugins should not become the hidden source of assay meaning.

## Source Of Truth

The unit of work is an experiment directory:

```text
experiments/
  <experiment>/
    config.yaml
    inputs/
    notebooks/
    outputs/
```

`config.yaml` is the authored source of truth. `outputs/` is generated state. `records.json` under `outputs/manifests/` is the canonical provenance ledger for emitted records and file bundles.

## Component Map

The main ownership boundaries are:

- `src/reader/protocols/`
  Protocol catalog, built-in assay families, semantic protocol bindings, and protocol-owned defaults for plots, exports, and notebooks.
- `src/reader/workbench/config/`
  YAML loading and schema validation only.
- `src/reader/workbench/experiment/`
  Typed experiment-local semantics such as resources, annotations, output layout, and bound protocol state.
- `src/reader/workbench/decl/`
  Compiled declaration layer between authored config and runtime execution.
- `src/reader/workbench/graph/`
  Typed runtime refs for records, files, resources, and workbench nodes.
- `src/reader/workbench/engine/`
  Planning, validation, contract enforcement, and runtime execution.
- `src/reader/workbench/records/`
  Record persistence, provenance, and catalog discovery.
- `src/reader/workbench/assets/`
  Explicit built-in asset catalogs for plugins and notebook templates.
- `src/reader/plugins/`
  Thin execution adapters grouped by category: ingest, transform, validator, plot, export.
- `src/reader/domains/`
  Domain-owned parsing, analysis, ordering, and plotting logic.
- `src/reader/contracts/`
  Dataframe contract kernel and built-in contract catalog.

## Runtime Lifecycle

The deterministic execution path is:

1. Load `reader/v7` YAML.
2. Validate schema and reject removed legacy keys.
3. Bind the experiment to a protocol.
4. Compile authored config into a workbench declaration.
5. Inspect or verify the compiled plan:
   `reader inspect`, `reader steps`, `reader explain`, `reader validate`, `reader run --dry-run`.
6. Execute:
   `reader run`, `reader plot`, `reader export`, `reader notebook`.
7. Persist generated records and artifacts under `outputs/` with manifest-backed provenance.

The CLI mirrors this lifecycle on purpose. Discovery and preflight are first-class, not side effects of execution.

## Registry Model

`reader` uses explicit registries so the information architecture stays scalable as assay families grow.

- Protocol registry
  Owns assay-facing semantics and defaults.
- Plugin asset registry
  Owns executable ingest, transform, validator, plot, and export implementations plus their semantic descriptors.
- Contract registry
  Owns dataframe contract identities and validation rules.
- Notebook template registry
  Owns scaffoldable notebook entry points and protocol compatibility.

These registries are meant to reduce cognitive load, not increase it. Experiment authors should normally interact with protocols and semantic outputs. Maintainers use plugin registries when extending or debugging the workbench kernel.

## Architectural Invariants

These are the invariants the codebase should preserve.

- `reader/v7` is the only supported public config schema.
- `workbench/config/` parses wire format; it does not become the internal authored model or runtime graph.
- Protocols own assay semantics and user-facing output vocabulary.
- Plugins are mechanics. They should be thin adapters around domain logic.
- Domain packages own math, parsing, ordering, and figure-planning logic.
- Generated files live under `outputs/`; they are not hand-edited.
- Records and artifacts must be traceable through `outputs/manifests/records.json`.
- Discovery, validation, and dry-run surfaces must remain available without executing the full pipeline.

## Known Architectural Debt

The deepest remaining debt is now concentration, not split semantic ownership.

The compiled semantic program is explicit end-to-end, from bound protocol
through compiled plan to experiment semantics and inspection payloads. The
remaining architecture pressure is that too much assay detail still collects in
three places:

- `src/reader/protocols/builtins.py`
- `src/reader/protocols/compiler.py`
- `src/reader/workbench/notebooks/` for retron-review flows

Those surfaces are still coherent, but they are large enough that future
assay families can turn them into semantic monoliths if new logic is not pushed
down into domain modules and family-specific helpers.

## Extension Guide

Use these rules when adding new behavior.

- Add a new protocol when the assay family changes.
- Add a new plugin when the execution mechanic changes.
- Add shared logic under `domains/` when the computation or parsing is domain-owned.
- Add new semantic outputs as protocol figures or artifacts, not as raw plugin ids in user config.
- Prefer composition over near-identical plugin variants.

## Related Docs

- [DESIGN.md](./DESIGN.md)
- [QUALITY.md](./QUALITY.md)
- [RELIABILITY.md](./RELIABILITY.md)
- [SECURITY.md](./SECURITY.md)
- [docs/core/spec.md](./docs/core/spec.md)
