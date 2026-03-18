[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions/workflows/ci.yaml)

![reader banner](assets/reader-banner.svg)

`reader` is a protocol-driven experimental workbench for structured assay data. It gives experiment authors one central YAML surface to declare assay inputs, analysis choices, and requested outputs, then compiles that authoring surface into a deterministic runtime plan that ingests raw data, applies transforms, emits traceable records, and materializes plots, exports, and notebooks.

It is meant to stay ergonomic for users and honest for maintainers. Users should be able to answer plain questions such as “what assay does this experiment use?”, “what gets ingested?”, “what transform chain runs?”, and “what plots or artifacts will be produced?” from the CLI and docs without reading compiler code or plugin internals. Maintainers should be able to add new assay families, ingest adapters, transforms, plots, and artifacts without turning the public YAML surface into a junk drawer.

The default UX is human-first: tables for normal use, JSON for automation and agent harnesses. Start with `reader ls`, `reader protocols`, `reader inspect`, and `reader explain`; the detailed contract shapes for discovery, inspection, validation, records, plots, and exports live in the docs index and CLI reference instead of being duplicated here.

---

## Documentation

- [Docs index](docs/README.md): authoritative route map for user, maintainer, and demo flows.
- [Architecture](ARCHITECTURE.md): top-level system map, ownership boundaries, registries, and invariants.
- [Design](DESIGN.md): product and information-design principles for `reader/v7`, protocols, and progressive disclosure.
- [Quality](QUALITY.md): quality bar, harness endpoints, evidence expectations, and failure taxonomy.
- [Reliability](RELIABILITY.md): preflight/run/verify contract, provenance expectations, and recovery model.
- [Security](SECURITY.md): trust boundaries, safe defaults, and extension-surface guidance.
- [CLI reference](docs/core/cli.md): discovery, inspection, validation, run, plot, export, notebook, and record commands.
- [Configuring reader/v7](docs/core/pipeline.md): the public YAML authoring surface and how protocols expose semantic inputs, analysis knobs, plot outputs, and export artifacts.
- [Plugin development](docs/core/plugins.md): maintainer-facing guide for adding ingest, transform, plot, export, and validator plugins without leaking mechanics into user config.
- [Spec / architecture](docs/core/spec.md): deeper architecture notes and package layout.

## Quickstart

```bash
uv sync --locked --group dev --group notebooks
uv run reader ls --root experiments
uv run reader ls --root experiments --details --format json
uv run reader ls --root experiments --details --protocol plate_reader/dual_reporter_screen
uv run reader plugins --protocol plate_reader/dual_reporter_screen --category transform --format json
uv run reader init ./experiments/20260317_new_assay --protocol plate_reader/dual_reporter_screen
uv run reader inspect experiments/template/config.yaml
uv run reader inspect experiments/template/config.yaml --format json
uv run reader config experiments/template/config.yaml --format json
uv run reader steps experiments/template/config.yaml --format json
uv run reader validate experiments/template/config.yaml --no-files --format json
uv run reader explain experiments/template/config.yaml --format json
uv run reader run experiments/template/config.yaml --dry-run --format json
uv run reader records experiments/template/config.yaml --format json
uv run reader protocols plate_reader/dual_reporter_screen --example-config
uv run reader plot experiments/template/config.yaml --list --format json
uv run reader export experiments/template/config.yaml --list --format json
```

## Workbench layout

```text
experiments/
  <experiment>/
    config.yaml
    inputs/
    notebooks/
    outputs/
```

Generated content belongs under `outputs/`. Fix the config or code and re-run the workflow instead of hand-editing generated artifacts.
