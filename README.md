![Reader: organize assay inputs, validate before running, and trace every output](assets/reader-banner.svg)

[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions/workflows/ci.yaml)
[![Integration](https://github.com/e-south/reader/actions/workflows/integration.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions/workflows/integration.yaml)
[![codecov](https://codecov.io/gh/e-south/reader/graph/badge.svg)](https://codecov.io/gh/e-south/reader)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Package: wheel + sdist](https://img.shields.io/badge/package-wheel%20%2B%20sdist-4c1.svg)](pyproject.toml)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-6f4bf2.svg)](https://docs.astral.sh/uv/)
[![Lint and format: Ruff](https://img.shields.io/badge/lint%20%2B%20format-Ruff-d7ff64.svg)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f855a.svg)](LICENSE)
[![Schema: reader/v8](https://img.shields.io/badge/schema-reader%2Fv8-1d6f8c.svg)](docs/core/pipeline.md)

Reader turns instrument files and experiment metadata into validated,
traceable analysis outputs. Each experiment keeps source files in `inputs/`,
its small `reader/v8` configuration in `config.yaml`, and generated records,
plots, exports, and notebooks in `outputs/`.
Cross-experiment aggregates follow the same contract: the aggregate is an
experiment, not a second top-level output namespace.

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

- [Getting started](docs/guides/getting_started.md) — install Reader, run the
  demo, and scaffold a first experiment.
- [Common tasks](docs/guides/common_routes.md) — shortest commands for
  discovery, validation, execution, and automation.
- [Python API](docs/core/python_api.md) — typed, task-oriented experiment and
  plugin interfaces for integrations.
- [Documentation index](docs/README.md) — complete user, reference, and
  maintainer documentation.

Reader is available under the [MIT license](LICENSE).
