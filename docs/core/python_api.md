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
from reader.api import inspect, notebook, plan, plots, read_dataframe, records, run, validate, verify

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
- `verify(experiment)` checks the catalog-schema-v4 envelope,
  record-schema-v5 source, config, Reader-build, exact-upstream-revision, and
  generated-artifact evidence together with the active invocation-schema-v2
  lifecycle. It does not change outputs.
- `run(experiment)` executes pipeline steps through the same engine path as the
  CLI and returns the invocation id, provenance epoch, selected steps, exact
  produced record revisions, and active invocation-ledger path.
- `notebook(experiment, name=None, template=None, overwrite=False)` generates a
  protocol-compatible Marimo workbench under the experiment's configured
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

## Notebook components and artifact publication

Generated notebooks import reusable controls from `reader.api.notebooks` and
load data only through `records()` and `read_dataframe()`. Record selection,
digest verification, and dataframe-contract validation therefore use the same
public path as non-notebook integrations. `load_notebook_context()` also
projects compiled pipeline-step metadata and configured ordered state spaces;
`resolve_effective_step_config()` returns one selected step's protocol-bound
plugin configuration. Those projections are domain-neutral, so assay templates
select and adapt the steps they understand.

When an interactive notebook produces files worth retaining, publish them as
one experiment-owned bundle:

```python
from reader.api import ArtifactSpec, publish_artifact_bundle

result = publish_artifact_bundle(
    experiment,
    record_id="notebook:review",
    producer_id="review",
    template="notebook/eda",
    upstream_records={"table": "analysis/summary"},
    producer_config={"view": "overview"},
    description="Reviewed summary and figure.",
    artifacts=(
        ArtifactSpec(
            relative_path="summary.pdf",
            description="Reviewed summary figure.",
            writer=lambda path: figure.savefig(path),
        ),
    ),
)
```

Publication is confined to the experiment's configured exports directory. It
captures exact upstream revisions, writes an immutable file-bundle revision,
registers the bundle in `records.json`, and records the operation in the normal
invocation ledger. Invalid paths, missing inputs, or writer failures stop before
catalog publication.

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

## Boundary

Import public operations from `reader.api` or use `open_experiment` from the
package root. Domain-specific protocols, including record-backed aggregates,
use these same operations. Modules under `reader.domains`, `reader.runtime`,
and `reader.workbench` are implementation details.
