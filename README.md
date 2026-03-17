[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions/workflows/ci.yaml)

![reader banner](assets/reader-banner.svg)

`reader` is a protocol-driven experimental workbench for structured assay data. It gives experiment authors one central YAML surface to declare assay inputs, analysis choices, and requested outputs, then compiles that authoring surface into a deterministic runtime plan that ingests raw data, applies transforms, emits traceable records, and materializes plots, exports, and notebooks.

It is meant to stay ergonomic for users and honest for maintainers. Users should be able to answer plain questions such as “what assay does this experiment use?”, “what gets ingested?”, “what transform chain runs?”, and “what plots or artifacts will be produced?” from the CLI and docs without reading compiler code or plugin internals. Maintainers should be able to add new assay families, ingest adapters, transforms, plots, and artifacts without turning the public YAML surface into a junk drawer.

The discovery commands keep a human table view by default and also support `--format json` for agent harnesses and machine-readable inspection. `reader ls --details` now acts like a workbench inventory surface: it shows the assay protocol, selected runtime plan summary, generated results on disk, and supports `--protocol` / `--status` filters when the experiment tree grows.

---

## Documentation

- [Docs index](docs/README.md): authoritative route map for user, maintainer, and demo flows.
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
uv run reader init ./experiments/20260317_new_assay --protocol plate_reader/dual_reporter_screen
uv run reader inspect experiments/template/config.yaml
uv run reader inspect experiments/template/config.yaml --format json
uv run reader protocols plate_reader/dual_reporter_screen --example-config
uv run reader validate experiments/template/config.yaml --no-files
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
