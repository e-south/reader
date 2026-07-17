---
doc_id: reader-overview
surface: repository-overview
owner: reader-maintainers
last_verified: 2026-07-10
summary: Short entry point for installing reader and finding the canonical operating and maintainer guides.
---

[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/reader/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/e-south/reader/graph/badge.svg)](https://codecov.io/gh/e-south/reader)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

![reader banner](assets/reader-banner.svg)

`reader` organizes experiment directories and runs config-driven analysis
pipelines over structured assay data. Each experiment uses a fixed layout:
raw inputs in `inputs/`, optional notebooks in `notebooks/`, generated results
in `outputs/`, and a `reader/v8` `config.yaml` that declares the run.

---

## Documentation

- [Documentation index](docs/README.md): full map of user, reference, and
  maintainer docs.
- [Getting started](docs/guides/getting_started.md): install `reader`, check
  the environment, and inspect a first experiment.
- [Preflight, run, verify](docs/guides/preflight_run_verify.md): inspect,
  validate, and execute one experiment.
- [Automation and JSON](docs/guides/automation.md): machine-readable
  discovery, inspection, and preflight routes.
- [Data Operations Plan](docs/guides/data_operations_plan.md): classify data
  before intake and capture the minimum metadata needed for reliable reuse.
- [Experiment bootstrap](docs/guides/experiment_bootstrap.md): create an
  experiment from local or Drive-backed inputs.
- [Workbench gardening](docs/guides/workbench_gardening.md): maintainer
  workflow for architecture and docs cleanup.
- [CLI reference](docs/core/cli.md): full command reference.
- [Configuring `reader/v8`](docs/core/pipeline.md): schema and protocol-owned
  config surface.
- [Repo maintenance](docs/repo-maintenance.md): repo-wide checks, CI, and
  maintainer routines.
