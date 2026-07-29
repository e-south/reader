---
doc_id: reader-architecture
surface: architecture
owner: reader-maintainers
last_verified: 2026-07-29
summary: Canonical map of Reader layers, ownership boundaries, lifecycle, registries, and extension points.
---

# Architecture

`reader` is a protocol-driven experimental workbench. The architecture is built around one constraint: experiment authors should declare assay intent once in `config.yaml`, while the runtime stays deterministic, inspectable, and extensible for maintainers.

This file is the top-level map. Use it to understand the layers, ownership boundaries, and extension points before dropping into package-level docs.

## System Model

`reader` has four layers.

1. Authoring model
   Human-facing `reader/v8` YAML in `experiments/<exp>/config.yaml`.
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

`config.yaml` is the authored source of truth. `outputs/` is generated state.
`records.json` under `outputs/manifests/` is the canonical catalog for emitted
records and file bundles. `invocations.jsonl` records execution attempts and
terminal outcomes.

## Component Map

Each source package has one architectural role:

| Package | Owns | Must not own |
| --- | --- | --- |
| `api/` | Stable task-oriented Python operations and typed results | Assay semantics or alternate execution paths |
| `protocols/` | Assay vocabulary, defaults, semantic programs, and family compilers | Raw parser or renderer implementations |
| `domains/` | Experimental parsing, transforms, analysis, ordering, and figure planning | Config loading, record lookup, runtime composition, or CLI behavior |
| `plugins/` | Thin ingest, transform, validator, plot, and export adapters | Hidden assay policy |
| `contracts/` | Dataframe identities, schemas, and validation | Workflow order |
| `plotting/` | Assay-neutral figure styles, sinks, and save mechanics | Assay-specific plotting decisions |
| `runtime/` | Composition and adapters that connect domain operations to workbench state | New domain meaning |
| `workbench/` | Config, compiled declarations, graph refs, execution, records, CLI, notebooks, and experiment-local state | Raw scientific logic that can stand alone |
| `maintenance/` | Repository checks exposed through the Reader CLI | Experiment execution |

Within `workbench/`, the dependency direction is config and protocol binding,
then declarations and graph refs, then engine execution and records. CLI,
inspection, and notebooks are operator surfaces over that path; they are not
alternate semantic owners.

The repository test suite enforces the most important inverse boundary:
`domains/` cannot import `api`, `maintenance`, `plugins`, `protocols`,
`runtime`, or `workbench`.

### Response-window capability map

Response-window processing crosses several layers without giving any one
layer all responsibilities:

| Surface | Responsibility |
| --- | --- |
| `domains/plate_reader/analysis/response_window/` | Request contracts, source validation, reductions, aggregation, and uncertainty |
| `domains/plate_reader/evidence/response_window/` | Bundle publication, preflight, provenance checks, and verification |
| `domains/plate_reader/plots/response_window/` | Assay-specific review tables, figure planning, rendering, and display labels |
| `runtime/response_window.py` | Resolve experiment declarations and records, then compose the domain service |
| `api/response_window/` | Stable Python service and review facades |
| `workbench/cli/response_window.py` and `workbench/notebooks/response_window.py` | Operator commands and notebook generation |

This capability is not a runtime plugin: it coordinates several verified
experiment records and publishes an aggregate experiment bundle. Its figure
decisions are assay-specific, so they do not belong in the assay-neutral
`reader.plotting` package.

### Agent workflow surface

Repository-specific agent workflows live under `.agents/skills/`, the Codex
repository discovery root. They route agents to Reader docs and commands; they
are not Python runtime plugins. An installable agent plugin or MCP server is
appropriate only when a workflow must be distributed as a bundle or backed by
live external tools, authentication, or controlled remote actions.

## Capability Flow

The control path and data path are separate but meet at typed plugin ports:

```text
config -> protocol -> compiled plan -> engine -> plugin adapter
                                                |
source -> ingest -> dataframe record -> transform -> dataframe record
                                                   |
                                      plot | export | notebook
                                                   |
                                             file-bundle record
```

Adding an ingest format, transform, or figure should extend one segment of this
flow. It should not introduce a second route from config to execution or make a
domain package discover workbench state.

## Runtime Lifecycle

The deterministic execution path is:

1. Load `reader/v8` YAML.
2. Validate schema and reject removed config keys.
3. Bind the experiment to a protocol.
4. Compile authored config into a workbench declaration.
5. Inspect or verify the compiled plan:
   `reader inspect`, `reader steps`, `reader explain`, `reader validate`, `reader run --dry-run`.
6. Execute:
   `reader run`, `reader plot`, `reader export`, `reader notebook`.
7. Persist generated records and artifacts under `outputs/` with manifest-backed provenance.
8. Append a structured invocation result that points to exact produced record revisions.

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
- Runtime composition
  Assembles the built-in catalogs once and supplies stateful adapters such as
  record-store access to CLI and API operations.

These registries are meant to reduce cognitive load, not increase it. Experiment authors should normally interact with protocols and semantic outputs. Maintainers use plugin registries when extending or debugging the workbench kernel.

## Architectural Invariants

These are the invariants the codebase should preserve.

- `reader/v8` is the only supported public config schema.
- `workbench/config/` parses wire format; it does not become the internal authored model or runtime graph.
- Protocols own assay semantics and user-facing output vocabulary.
- Plugins are mechanics. They should be thin adapters around domain logic.
- Domain packages own math, parsing, ordering, and figure-planning logic.
- Domain packages accept explicit data and parameters; runtime adapters resolve
  configs, catalogs, and generated records before calling them.
- Generated runtime files live under each experiment's `outputs/`; they are not hand-edited.
- The repository root has no runtime `outputs/` contract. A cross-experiment
  aggregate is itself an experiment and owns its generated bundles under that
  experiment's `outputs/` directory.
- Records and artifacts must be traceable through `outputs/manifests/records.json`.
- Execution attempts and outcomes belong in `outputs/manifests/invocations.jsonl`,
  not an authored journal.
- Discovery, validation, and dry-run surfaces must remain available without executing the full pipeline.

## Current Architecture Pressure

The compiled semantic program has one path from protocol binding through the
runtime plan and inspection payloads. The following concentrated owners remain
worth decomposing when a change reaches them:

- plate-reader protocol descriptors are concentrated in
  `src/reader/protocols/_builtins_plate_reader_variants.py`
- `src/reader/protocols/compilers/plate_reader.py` remains the largest family
  compiler; `src/reader/protocols/compiler.py` is only its stable public facade
- notebook composition belongs under `src/reader/workbench/notebooks/`, while
  reusable calculations belong under `src/reader/domains/`

New work should move one coherent responsibility at a time into a domain or
family package. A split is useful when it creates an explicit contract or
owner; splitting only to reduce line count is not sufficient.

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
