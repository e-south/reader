# ![Reader data workbench](https://raw.githubusercontent.com/e-south/reader/main/assets/reader-banner.svg)

[![Checks](https://github.com/e-south/reader/actions/workflows/checks.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions/workflows/checks.yaml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/downloads/release/python-3120/)
[![MIT license](https://img.shields.io/badge/license-MIT-3D8068.svg)](https://github.com/e-south/reader/blob/main/LICENSE)

Reader ingests instrument files and experiment metadata, applies declared
transformations, and writes validated records, plots, exports, and notebooks.
Every unit of work—including a cross-experiment aggregate—has an owned directory
beneath `experiments/`, with source material in `inputs/`, a `reader/v8`
contract in `config.yaml`, and generated artifacts in `outputs/`.
The distribution and import package are named `reader-workbench` and
`reader_workbench`; the installed command remains `reader`.

## Install

The `1.0.0` PyPI release is prepared but not published yet. From a checkout,
install Reader as a command-line tool:

```bash
uv tool install .
```

Or install it into an active Python environment from the checkout:

```bash
python -m pip install .
```

After the first release, replace `.` with the distribution name
`reader-workbench`.

The installed command is `reader`:

```bash
reader demo
reader protocols
reader init ./experiments/my_experiment --protocol plate_reader/single_reporter_screen
```

The demo prints a guided command tour. It does not execute a pipeline or write
files. Every generated starter can be inspected and validated before data is
added.

## Contribute from a checkout

```bash
uv sync --locked --group dev
uv run reader demo
```

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
