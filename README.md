[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/e-south/reader/graph/badge.svg)](https://codecov.io/gh/e-south/reader)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

![reader banner](assets/reader-banner.svg)

`reader` is a toolkit for organizing experiment directories and running config-driven analysis pipelines over structured assay data. Each experiment has a clear working layout: raw inputs live in `inputs/`, optional notebooks live in `notebooks/`, generated results live in `outputs/`, and a `reader/v7` `config.yaml` describes what should be run.

---

## Documentation

- [Documentation index](docs/README.md): complete map of user docs, reference docs, and maintainer docs.
- [Getting started](docs/guides/getting_started.md): install `reader`, verify the environment, and inspect the first experiment.
- [Preflight, run, verify](docs/guides/preflight_run_verify.md): deterministic path for inspecting, executing, and checking one experiment.
- [Automation and JSON](docs/guides/automation.md): machine-readable discovery, inspection, and preflight surfaces.
- [CLI reference](docs/core/cli.md): full command reference.
- [Configuring `reader/v7`](docs/core/pipeline.md): the public authoring surface for experiment configs.
- [Repo maintenance](docs/repo-maintenance.md): maintainer verification and CI lanes.
