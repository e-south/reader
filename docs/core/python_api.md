---
doc_id: reader-python-api
surface: api-reference
owner: reader-maintainers
last_verified: 2026-07-29
summary: Stable task-oriented Python entrypoints for inspecting, verifying, and running Reader experiments without composing workbench internals.
---

# Python API

Use `reader.api` when Python code needs Reader results directly. The API owns
config loading, protocol binding, and workbench compilation, so callers do not
need to assemble runtime, declaration, graph, engine, and record objects.

```python
from reader import open_experiment
from reader.api import inspect, plan, plots, records, run, validate, verify

experiment = open_experiment("experiments/2026/my_experiment")

preflight = validate(experiment)
inspection = inspect(experiment)
execution_plan = plan(experiment)
plot_catalog = plots(experiment)
record_catalog = records(experiment)
verification = verify(experiment)
execution = run(experiment)
```

Each operation returns a typed result with `to_dict()` for serialization. An
`Experiment` loads and compiles its config once and can be reused across calls.

## Experiment operations

- `open_experiment(path)` accepts a `config.yaml` path or its experiment
  directory.
- `inspect(experiment)` returns authoring, assay semantics, and current
  implementation state.
- `validate(experiment, check_files=True)` runs preflight checks.
- `plan(experiment)` returns the resolved protocol plan without execution.
- `plots(experiment, only=(), exclude=())` returns selected plot contracts.
- `records(experiment, include_history=False)` reads the record catalog without
  creating one when it is absent.
- `verify(experiment)` checks current schema-v5 source, config, Reader-build,
  exact-upstream-revision, and generated-artifact evidence without changing
  outputs.
- `run(experiment)` executes pipeline steps through the same engine path as the
  CLI and returns the invocation id, selected steps, exact produced record
  revisions, and invocation-ledger path.

Use `run(experiment, dry_run=True)` to validate and select the effective
pipeline without creating `outputs/`; its result has status `planned` and no
invocation or ledger path. `from_step`, `until_step`, and `only` provide the
same pipeline slicing semantics as the corresponding CLI options. Plot and
export mutation remain behind `reader plot` and `reader export` while their
typed public results are defined.

## Plugin discovery

```python
from reader.api import describe_plugin, plugins

catalog = plugins(category="ingest", domain="cytometry")
descriptor = describe_plugin("ingest/flow_cytometer")
```

Plugin descriptors include the Pydantic config schema, typed input and output
ports, dataframe contracts, and promoted contract surfaces. Protocols remain
the public experiment-authoring layer; plugin discovery is for integrations
and maintainer tooling.

## Response-window operations

```python
from reader.api.response_window import (
    build_response_window_bundle,
    preflight_response_window_request,
    verify_response_window_bundle,
)
from reader.api.response_window.review import load_review_tables, render_review_figure
```

The service facade composes Reader runtime state with the plate-reader domain
contracts. Review helpers expose verified tables and assay-specific figures
without making callers import domain or workbench implementation modules.

## Boundary

Import public operations from `reader.api`, use `reader.api.response_window`
for the response-window capability, or use `open_experiment` from the package
root. Modules under `reader.domains`, `reader.runtime`, and `reader.workbench`
are implementation details.
