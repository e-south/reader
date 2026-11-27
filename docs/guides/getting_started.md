---
doc_id: reader-getting-started
surface: tutorial
owner: reader-maintainers
last_verified: 2026-11-27
summary: Install Reader and inspect the config and input requirements for a first experiment.
---

# Getting started

Install the released CLI with [uv](https://docs.astral.sh/uv/). Reader uses
Python 3.12. If your shell cannot find the installed command, run
`uv tool update-shell` and open a new shell.

```bash
uv tool install reader-workbench --python 3.12
reader --version
reader protocols
```

The distribution is `reader-workbench`, the command is `reader`, and the
Python import name is `reader_workbench`. For a source checkout, use the
[development setup](../../README.md#develop-from-a-checkout).

## Create and inspect a starter

From an empty working directory, create a single-reporter plate experiment.
If you already followed the README, start at `reader inspect`.

```bash
exp=./experiments/first_plate
reader init "$exp" --protocol plate_reader/single_reporter_screen
reader inspect "$exp"
reader validate "$exp" --no-files
reader explain "$exp"
```

`init` creates `config.yaml`, `inputs/`, and `notebooks/`. The config selects
the `reader/v8` schema and sets the protocol's channels and output options.
`inspect` shows that setup. `validate --no-files` checks the configuration
and step connections. `explain` shows where each step reads and writes.

## Add measurements

The starter expects one instrument workbook in `inputs/` and a sample map
at `inputs/metadata.xlsx`. The map connects well positions to sample and
treatment identifiers. Supply your label mapping in `annotations.labels`.

- To try Reader with invented measurements, follow [Run a sample plate](./demo.md).
- For your own files, use [Experiment bootstrap](./experiment_bootstrap.md) to establish the metadata, then [Preflight, run, verify](./preflight_run_verify.md) to validate the files and produce records.
- To find an existing experiment, run `reader ls --root experiments --details --readiness` from its parent workspace.

After a run, the catalog at `outputs/manifests/records.json` lists the generated
records. `reader verify "$exp"` checks them against the files and configuration.
See [Common tasks](./common_routes.md) for plots, exports, and JSON, or
[Notebooks](./notebooks.md) for scaffolding and launching a notebook in a
separately managed Marimo environment. Existing Python integrations can use
the [namespace migration guide](./package_namespace_migration.md).
