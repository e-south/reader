![Reader: experimental-data workbench from ingest through trace](https://raw.githubusercontent.com/e-south/reader/main/assets/reader-banner.svg)

[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions/workflows/ci.yaml)
[![Integration](https://github.com/e-south/reader/actions/workflows/integration.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions/workflows/integration.yaml)
[![codecov](https://codecov.io/gh/e-south/reader/graph/badge.svg)](https://codecov.io/gh/e-south/reader)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Package: wheel + sdist](https://img.shields.io/badge/package-wheel%20%2B%20sdist-4c1.svg)](https://github.com/e-south/reader/blob/main/pyproject.toml)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-6f4bf2.svg)](https://docs.astral.sh/uv/)
[![Lint and format: Ruff](https://img.shields.io/badge/lint%20%2B%20format-Ruff-d7ff64.svg)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f855a.svg)](https://github.com/e-south/reader/blob/main/LICENSE)

Reader ingests instrument files and experiment metadata, applies declared
transformations, and writes validated records, plots, exports, and notebooks.
Every unit of work—including a cross-experiment aggregate—lives under
`experiments/<year>/<experiment>/`, with source material in `inputs/`, a
`reader/v8` contract in `config.yaml`, and generated artifacts in `outputs/`.

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
