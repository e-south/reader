---
doc_id: reader-architecture
surface: architecture
owner: reader-maintainers
last_verified: 2026-08-01
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
Catalog-schema-v4 `records.json` under `outputs/manifests/` is the canonical
catalog for emitted records and file bundles. It owns the active provenance
epoch. Attempt and terminal events use invocation schema v2 and live in the
matching `outputs/manifests/invocations/<epoch>.jsonl` ledger.

## Component Map

Each source package has one architectural role:

| Package | Owns | Must not own |
| --- | --- | --- |
| `api/` | Stable task-oriented Python operations and typed results | Assay semantics or alternate execution paths |
| `protocols/` | Assay vocabulary, defaults, semantic programs, family compilers, and assay step composition | Raw parser or renderer implementations |
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

### Where reusable calculations belong

A calculation belongs in Reader when its inputs are declared measurement
records, its parameters describe acquisition or reduction mechanics, and its
output remains meaningful without knowing a study id, candidate identity, or
selection preference. The pure operation belongs under `domains/`; a thin
transform plugin binds typed ports; a protocol decides when that transform is
part of an assay plan.

A calculation remains downstream when it binds assay rows to study entities,
chooses a scientific target or control for a particular question, embeds
acceptance thresholds or ranking preferences, or composes evidence into a
study narrative. A named study endpoint does not become a Reader primitive
merely because its equation could be implemented as a dataframe transform.
If several studies later need the same measurement reduction, extract the
preference-free reduction into Reader and keep each study's identity,
calibration, scoring, and interpretation with that study.

Assay-neutral temporal selection and reduction live under
`domains/time_series/`. That package owns endpoint versus interval selection,
absolute versus event-relative coordinates, numerical value space, trace
support, gap, and censor mechanics. Assay packages adapt their authored policy
to that contract. Within-unit observation reduction and across-unit centering
remain explicit and separate. Evidence records replicate kind and replicate
identity as independent facts: the kind may be biological, technical, mixed,
unknown, or not applicable while the within-record grouping remains unresolved.
An absent identity field does not make the experiment, plate, sheet, well, or
position a replicate. Those fields remain acquisition provenance unless the
experiment explicitly declares one as its replicate identity. A descriptive
plot may instead opt into a typed observation-only unit, but it must label that
unit as non-replicate evidence and cannot silently choose an aggregation
relationship. Semantic unit scope is separate from presentation partitioning:
a diagnostic partition must resolve to exactly one declared entity tuple, so
changing panel membership cannot silently pool subject- or genotype-specific
replicate populations. Multi-entity summaries require an explicit comparison
figure with its own aggregation contract.

Cross-repository bridge skills only route between these owners. They may name
public contracts and verification commands, but must not contain equations,
treatment ontologies, candidate mappings, or executable study logic.

### Aggregate capability map

Cross-experiment analyses use the same lifecycle as source experiments. Core
owns record references and provenance; protocols and domain packages own the
meaning of a particular collection:

| Surface | Responsibility |
| --- | --- |
| `workbench/experiments.py` | Resolve experiment identities below the canonical `experiments/` owner without assuming a year or directory name |
| `workbench/records/sources.py` | Resolve exact source-record revisions for typed record-collection ports |
| `domains/plate_reader/analysis/response_window/` | Response-window source validation, reductions, aggregation, and uncertainty |
| `domains/plate_reader/plots/response_window/` | Response-window summary selection, validation, labeling, and rendering |
| `plugins/transform/response_window.py` | Thin adapter from record collections to response-window dataframe records |
| `protocols/` | Compile the declared collection, plots, and exports into the normal workbench plan |

The four-state vector collection uses the same core record-reference seam. Neither
capability owns experiment discovery, direct publication, a custom manifest,
or a second API lifecycle. Their scientific rules remain specialized and do
not enter the generic workbench kernel.

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
                                             plot | export
                                                   |
                                             file-bundle record

record catalog -> verified public reads -> canonical notebook viewport
```

Adding an ingest format, transform, or figure should extend one segment of this
flow. It should not introduce a second route from config to execution or make a
domain package discover workbench state.

For an aggregate experiment, `source` can also be an exact dataframe record
from another declared experiment. The transform still receives explicit typed
data, and the resulting records follow the same engine and manifest path.

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
- Plugin catalog
  Owns executable ingest, transform, validator, plot, and export implementations plus their semantic descriptors.
- Contract registry
  Owns dataframe contract identities and validation rules.
- Runtime composition
  Assembles the built-in catalogs once, injects plugin-owned descriptors into
  the generic workbench registry, and supplies stateful adapters such as
  record-store access to CLI and API operations.

These registries are meant to reduce cognitive load, not increase it. Experiment authors should normally interact with protocols and semantic outputs. Maintainers use plugin registries when extending or debugging the workbench kernel.

Built-in plugins are registered explicitly by category; coordinated external
integrations may contribute descriptors through the `reader_workbench.plugins` entry
point. Registration validates category, identity, typed ports, and contract
surfaces before execution. A plugin becomes an experiment feature only when a
protocol compiler gives it a semantic role—registry presence alone never
creates a second authoring or execution surface.

The notebook is deliberately outside those registries and compiled plans.
Reader packages one fixed `notebook/eda` scaffold that opens the owning
experiment as a verified, RecordStore-backed operator viewport.

Data-producing work belongs in pipeline transforms. Plot and export plugins
consume persisted records and may emit file bundles, but cannot smuggle new
dataframe records into a presentation phase. The canonical notebook is a
read-only record viewport: any cataloged dataframe or file bundle can be
discovered through the verified public API without another computation or
publication lifecycle.

## Architectural Invariants

These are the invariants the codebase should preserve.

- `reader/v8` is the only supported public config schema.
- `workbench/config/` parses wire format; it does not become the internal authored model or runtime graph.
- Protocols own assay semantics and user-facing output vocabulary.
- Plugins are mechanics. They should be thin adapters around domain logic.
- Pipeline transforms own dataframe outputs; plot and export plugins own file
  bundles only.
- Domain packages own math, parsing, ordering, and figure-planning logic.
- Domain packages accept explicit data and parameters; runtime adapters resolve
  configs, catalogs, and generated records before calling them.
- Cross-experiment inputs are `resources` of kind `record`. They resolve by
  experiment id and record id, bind to exact revisions, and pass through typed
  record-collection ports.
- Generated runtime files live under each experiment's `outputs/`; they are not hand-edited.
- The repository root has no runtime `outputs/` contract. A cross-experiment
  aggregate is itself an experiment and owns its generated bundles under that
  experiment's `outputs/` directory.
- Records and artifacts must be traceable through `outputs/manifests/records.json`.
- Execution attempts and outcomes belong in the catalog-selected
  `outputs/manifests/invocations/<epoch>.jsonl`, not an authored journal.
- Discovery, validation, and dry-run surfaces must remain available without executing the full pipeline.

## Current Architecture Pressure

The compiled semantic program has one path from protocol binding through the
runtime plan and inspection payloads. The following concentrated owners remain
worth decomposing when a change reaches them:

- plate-reader protocol descriptors are concentrated in
  `src/reader_workbench/protocols/_builtins_plate_reader_variants.py`
- `src/reader_workbench/protocols/compilers/plate_reader.py` remains the largest family
  compiler; `src/reader_workbench/protocols/compiler.py` is only its stable public facade
- notebook composition belongs under `src/reader_workbench/workbench/notebooks/`, while
  reusable calculations belong under `src/reader_workbench/domains/`

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
- Keep the canonical notebook record-driven so a newly registered contract or
  deliverable is inspectable without a specialized notebook lifecycle.

## Related Docs

- [DESIGN.md](./DESIGN.md)
- [QUALITY.md](./QUALITY.md)
- [RELIABILITY.md](./RELIABILITY.md)
- [SECURITY.md](./SECURITY.md)
- [docs/core/spec.md](./docs/core/spec.md)
