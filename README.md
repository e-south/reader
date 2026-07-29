![Reader data workbench](https://raw.githubusercontent.com/e-south/reader/main/assets/reader-banner.svg)

[![Checks](https://github.com/e-south/reader/actions/workflows/checks.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions/workflows/checks.yaml)
[![Coverage](https://codecov.io/gh/e-south/reader/graph/badge.svg)](https://codecov.io/gh/e-south/reader)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/downloads/release/python-3120/)
[![MIT license](https://img.shields.io/badge/license-MIT-3D8068.svg)](https://github.com/e-south/reader/blob/main/LICENSE)

Reader ingests instrument files and experiment metadata, applies declared
transformations, and writes validated records, plots, exports, and notebooks.
Every unit of work—including a cross-experiment aggregate—lives under
`experiments/<year>/<experiment>/`, with source material in `inputs/`, a
`reader/v8` contract in `config.yaml`, and generated artifacts in `outputs/`.
The distribution is named `reader-workbench`; its import package and command
remain `reader`.

## Try it

```bash
uv sync --locked
uv run reader demo
```

The demo prints a guided command tour. It does not execute a pipeline or write
files. To see the assay types Reader can scaffold:

```bash
uv run reader protocols
uv run reader init ./experiments/2026/20260728_my_experiment --protocol plate_reader/single_reporter_screen
```

Every generated starter can be inspected and validated before data is added.

## Learn more

- [Getting started](https://github.com/e-south/reader/blob/main/docs/guides/getting_started.md) — install Reader, run the
  demo, and scaffold a first experiment.
- [Common tasks](https://github.com/e-south/reader/blob/main/docs/guides/common_routes.md) — shortest commands for
  discovery, validation, execution, and automation.
- [Python API](https://github.com/e-south/reader/blob/main/docs/core/python_api.md) — typed, task-oriented experiment and
  plugin interfaces for integrations.
- [Documentation index](https://github.com/e-south/reader/blob/main/docs/README.md) — complete user, reference, and
  maintainer documentation.

Reader is available under the [MIT license](https://github.com/e-south/reader/blob/main/LICENSE).
