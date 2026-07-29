---
doc_id: reader-data-operations-plan
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-07-28
summary: Entry point for classifying datasets and recording metadata, transfer, readiness, and verification decisions.
---

# Data Operations Plan

Use this guide as the `reader`-local overlay for deciding what must be
captured before an experiment is run. It adapts the lab-facing parts of a
[Data Operations Plan](https://merelogic.net/data_operations_plans/how) to the
workbench without turning `reader/v8` into an organization-wide policy schema.

The short path is:

1. Classify the dataset.
2. Capture the minimum metadata for that class.
3. Stage raw files under the standard experiment layout.
4. Use `reader` preflight, execution, and records to prove what happened.

## Choose the Smallest Needed Reference

- [Operating model](./data_operations_plan/operating_model.md): understand what
  belongs in repo-local DOP policy, what remains outside `reader`, and which
  surface owns each fact.
- [Data classes](./data_operations_plan/data_classes.md): choose the protocol
  family or draft/template path before copying configs.
- [Metadata minimums](./data_operations_plan/metadata_minimums.md): decide what
  must be captured and when to stop for clarification.
- [Transfer and verification](./data_operations_plan/transfer_and_verification.md):
  stage inputs, run checks, and verify generated evidence.

For the concrete intake workflow, continue with
[Experiment bootstrap](./experiment_bootstrap.md). For the execution loop, use
[Preflight, run, verify](./preflight_run_verify.md).

Machine-readable inspection is available through the read-only
[`reader` DOP registry](../../src/reader/workbench/dop/):

```bash
uv run reader dop classes
uv run reader dop classes --format json
uv run reader dop ready-specs --format json
```

Repo-local agent routing lives in
[reader-data-operations-plan](../../skills/reader-data-operations-plan/SKILL.md).
Use that skill when the task is DOP classification, DOP registry/docs
maintenance, or checking that experiment-intake guidance still matches this
overlay.

## Operating Contract

- The DOP registry defines the available data classes. An experiment may record
  its selected class and reason in `evidence`; that block is evidence of the
  intake decision, not a second protocol or execution schema.
- `config.yaml`, `inputs/`, and hand-authored notes are the source of truth.
- Generated artifacts under `outputs/` are evidence, not source material.
- If well identity, treatment meaning, channel semantics, or control
  interpretation is ambiguous, stop and ask instead of encoding a guess.
- Add new config fields or CLI surfaces only after the docs-level contract has
  proven stable across real experiments.
- Keep the four DOP concerns separate: requirements explain why capture
  matters, design explains the repo operating path, configuration defines
  classes and canonical names, and instructions tell users or agents what to do
  next.

## Maintenance

Update this overlay when:

- a new protocol family is added;
- a recurring metadata ambiguity appears during experiment intake;
- a new external transfer path becomes common;
- validation misses a class of preventable failure; or
- a long-tail assay graduates from draft/template handling into a formal
  protocol.

Keep changes small: update one reference page first, then update
[Experiment bootstrap](./experiment_bootstrap.md) or repo-local skills only when
the operating workflow changes.
