---
doc_id: reader-python-api
surface: api-reference
owner: reader-maintainers
last_verified: 2026-07-29
summary: Stable task-oriented Python entrypoints for inspecting, verifying, and running Reader experiments without composing workbench internals.
---

# Python API

Use `reader_workbench.api` when Python code needs Reader results directly. The API owns
config loading, protocol binding, and workbench compilation, so callers do not
need to assemble runtime, declaration, graph, engine, and record objects.

```python
from reader_workbench import open_experiment
from reader_workbench.api import inspect, notebook, plan, plots, read_artifact, read_dataframe, records, run, validate, verify

experiment = open_experiment("experiments/my_experiment")

preflight = validate(experiment)
inspection = inspect(experiment)
execution_plan = plan(experiment)
plot_catalog = plots(experiment)
record_catalog = records(experiment)
tidy = read_dataframe(experiment, "ingest/df")
verification = verify(experiment)
execution = run(experiment)
notebook_result = notebook(experiment)
```

Each operation returns a typed result with `to_dict()` for serialization. An
`Experiment` loads and compiles its config once and can be reused across calls.
Its stable public state is `config_path` and `identity`; parsed config, compiled
declarations, runtime composition, and record stores remain API internals.

## Experiment operations

- `open_experiment(path)` accepts a `config.yaml`, its experiment directory, or
  an existing generated file nested beneath that experiment.
- `inspect(experiment)` returns authoring, assay semantics, and current
  implementation state.
- `validate(experiment, check_files=True)` runs preflight checks.
- `plan(experiment)` returns the resolved protocol plan without execution.
- `plots(experiment, only=(), exclude=())` returns selected plot contracts.
- `records(experiment, include_history=False)` reads the record catalog without
  creating one when it is absent. Catalog metadata reports the catalog schema,
  active provenance epoch, and active invocation-ledger path.
- `read_dataframe(experiment, record_id)` loads the latest dataframe revision,
  verifies its content digest, and returns a defensive dataframe copy together
  with its contract, content digest, revision number, and revision digest. Its
  `to_dict()` projection contains only JSON-ready identity and dataframe-shape
  metadata; dataframe values remain available through `.dataframe` in Python.
- `read_dataframe(..., revision=..., revision_digest=...)` loads the exact
  catalog revision selected by a caller. Add `row_limit=N` for a digest-verified
  bounded preview that does not materialize the full Parquet record.
  `read_artifact(...)` requires the same exact revision identity plus one
  cataloged outputs-relative file path, verifies its recorded size and digest,
  and returns bytes without exposing a local filesystem path.
- `verify(experiment)` checks the catalog-schema-v4 envelope,
  record-schema-v6 source, config, Reader-build, exact-upstream-revision, and
  generated-artifact evidence together with the active invocation-schema-v2
  lifecycle. It does not change outputs.
- `run(experiment)` executes pipeline steps through the same engine path as the
  CLI and returns the invocation id, provenance epoch, selected steps, exact
  produced record revisions, and active invocation-ledger path.
- `notebook(experiment, name=None, overwrite=False)` generates Reader's
  canonical Marimo workbench under the experiment's configured
  output directory. The shared EDA surface uses one deliverable selector, one
  primary viewport, and a lazy single-open accordion for detail.

Use `run(experiment, dry_run=True)` to validate and select the effective
pipeline without creating `outputs/`; its result has status `planned` and no
invocation, provenance epoch, or ledger path. `from_step`, `until_step`, and
`only` provide the same pipeline slicing semantics as the corresponding CLI
options.

Use `run(experiment, reset_records=True)` only to replace an invalid generated
catalog before a complete pipeline rerun. Reset cannot be combined with a dry
run or any partial selector. It creates a fresh catalog epoch and active ledger;
prior epoch ledgers remain inactive forensic residue. Plot and export mutation
remain behind `reader plot` and `reader export` while their typed public results
are defined.

## Notebook components and verified artifacts

Generated notebooks import reusable controls from `reader_workbench.api.notebooks` and
load data only through `records()`, `read_dataframe()`, `read_artifact()`, and
`verify()`. Each selector option binds the explicit `revision` and
`revision_digest` returned by `records()`; a refreshed or invalid selection
never falls through to a different artifact. File previews consume verified
bytes rather than direct local paths, and verification failures remain visible
as readiness notes. `load_notebook_context()` exposes only the experiment,
owned output paths, and compiled pipeline step ids needed by that viewport.
Protocol-owned plots carry assay-specific rendering into the shared viewport;
the notebook API does not project protocol inputs or effective plugin config.

Downstream integrations should preserve the same exact record identity:

```python
entry = next(item for item in record_catalog.entries if item["record_id"] == "plot:summary")
preview = read_artifact(
    experiment,
    entry["record_id"],
    revision=entry["revision"],
    revision_digest=entry["revision_digest"],
    path=entry["files"][0],
)
```

## Plugin discovery

```python
from reader_workbench.api import describe_plugin, plugins

catalog = plugins(category="ingest", domain="cytometry")
descriptor = describe_plugin("ingest/flow_cytometer")
```

Plugin descriptors include the Pydantic config schema, typed input and output
ports, dataframe contracts, and promoted contract surfaces. Protocols remain
the public experiment-authoring layer; plugin discovery is for integrations
and maintainer tooling.

## Boundary

Import public operations from `reader_workbench.api` or use `open_experiment` from the
package root. Domain-specific protocols, including record-backed aggregates,
use these same operations. Modules under `reader_workbench.domains`, `reader_workbench.runtime`,
and `reader_workbench.workbench` are implementation details.
