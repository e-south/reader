---
doc_id: reader-design
surface: design-principles
owner: reader-maintainers
last_verified: 2026-07-10
summary: Product and information-design rules for the Reader authoring, protocol, output, and agent surfaces.
---

# Design

`reader` is designed for two audiences at once:

- experiment authors who need an ergonomic, readable, central config surface
- maintainers who need a scalable way to add new assays, ingest paths, transforms, plots, and exports without semantic drift

This document explains the product and information-design choices behind that balance.

## Product Intent

An assay is a reusable vessel. An experiment is a concrete run of that assay with specific data, metadata, and analysis choices.

The design goal is for users to think in assay terms:

- what data gets ingested
- what semantic transforms happen
- what plots or artifacts are requested
- what protocol-specific knobs matter for this assay family

They should not need to think in compiler branches, plugin ids, or graph mutation.

## Public Surface Design

The public surface is `reader/v7`.

```yaml
schema: reader/v7
experiment:
  id: ...
protocol:
  id: ...
  inputs: ...
  analysis: ...
  outputs: ...
resources: ...
annotations: ...
paths: ...
plotting: ...
```

This shape is deliberate.

- `protocol.inputs`
  What the assay needs to ingest or bind.
- `protocol.analysis`
  Semantic choices that affect how the assay is interpreted.
- `protocol.outputs`
  Semantic selections for plots, exports, and notebooks.

That split keeps the authored config understandable and keeps growth pressure away from generic `with` bags or workflow patch surfaces.

## Design Principles

### One concept, one owner, one contract

- User-facing assay semantics belong to protocols.
- Runtime mechanics belong to plugins and the engine.
- Dataframe validity belongs to contracts.
- Provenance belongs to records.

If a concept exists in multiple places, it should exist for different reasons, not because ownership is confused.

### Progressive disclosure

The normal CLI ladder is:

1. `reader ls`
2. `reader init`
3. `reader inspect`
4. `reader steps`
5. `reader explain`
6. `reader validate`
7. `reader run`, `reader plot`, `reader export`, `reader notebook`

Users start with simple questions and only descend into deeper runtime detail when needed. Agents get the same ladder plus `--format json` contracts.

### Semantic outputs, not plugin-shaped outputs

Protocols should expose figures, plot profiles, and artifacts in assay language. The compiler may map those to plot/export plugins, but the user should choose semantic outputs first.

### Composition over nesting

New plot capability should usually come from:

- new figure primitives in domain code
- protocol-owned figure bundles or plot profiles
- thin plot adapters

It should not come from increasingly nested public config or a sprawl of near-identical plugins.

### Fail fast, no silent fallback

The workbench should prefer explicit errors over hidden coercion.

Examples:

- removed config keys are rejected
- unknown bindings fail validation
- path escapes are rejected
- JSON CLI surfaces return explicit empty payloads or hard errors instead of silently changing formats

## Designing New Assays

When onboarding a new assay family, use this decision path.

1. Put domain parsing, math, and plotting helpers in `domains/<domain>/`.
2. Add or extend protocol semantics in `protocols/`.
3. Add thin ingest/transform/plot/export plugins only where execution mechanics differ.
4. Expose user-facing outputs through protocol figures, plot profiles, and artifacts.
5. Keep the experiment config assay-shaped, not plugin-shaped.

If a new assay can only be expressed by growing ad hoc config bags, that is a design failure in the protocol layer.

## Designing New Outputs

There are three output vocabularies and they should stay distinct.

- Figures
  User-facing plot outputs.
- Plot profiles
  Named bundles of figures.
- Artifacts
  User-facing exports.

This keeps the surface flexible without asking users to manage low-level plot composition manually for every experiment.

## Designing For Agents

Naive agents should be able to answer:

- what protocol does this experiment use
- what does the pipeline ingest
- what transform chain runs
- what plots and exports are selected
- what outputs already exist
- whether the config validates
- what would run without mutating state

That is why discovery and preflight commands have machine-readable JSON surfaces. Good harness design is part of product UX, not just internal ops.

## Deliberate Non-Goals

- `reader` is not trying to be a generic workflow graph editor.
- It is not trying to let every experiment mutate the runtime model arbitrarily.
- It is not trying to expose plugin internals as the default user interface.

Those freedoms look flexible at first and usually turn into ontology drift, higher support cost, and unreadable experiment config.

## Current Design Debt

The public surface is protocol-owned and compact. The semantic compiler remains
incomplete: protocol controls, windows, metrics, and ranking are not one
executable typed analysis program.

That means the current design is directionally correct, but not finished. The next honest improvement is to make protocol semantics more executable, not to widen the YAML surface again.

## Related Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [QUALITY.md](./QUALITY.md)
- [RELIABILITY.md](./RELIABILITY.md)
- [SECURITY.md](./SECURITY.md)
