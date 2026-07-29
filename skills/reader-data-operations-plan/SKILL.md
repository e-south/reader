---
name: reader-data-operations-plan
description: Classifies reader datasets and maintains DOP registry/docs alignment. Use when selecting data classes, auditing gates, or updating rules. Do not use for full experiment creation, result interpretation, or outputs edits.
metadata:
  version: 0.2.0
  category: scientific-workbench
  tags: [reader, data-operations-plan, metadata, intake, registry]
---

# Reader Data Operations Plan

## Purpose

Keep `reader` Data Operations Plan classification and maintenance explicit by
routing agents to the DOP registry, owned docs, and source-backed checks.

## Scope

In scope:

- classifying a dataset before experiment bootstrap
- checking DOP data classes, metadata minimums, stop conditions, transfer rules,
  and ready-spec gates
- maintaining alignment between DOP docs, the read-only DOP registry, and
  repo-local skills
- source-backed updates to DOP guidance from the Merelogic Data Operations Plan
  resource

Out of scope:

- organization-wide ELN/LIMS, archive, retention, or role-assignment policy
- widening `reader/v8` without a separate code-change contract
- generic scientific result interpretation
- hand-editing generated `experiments/**/outputs/`

## Skill Composition

- Pair with `reader-experiment-bootstrap` when the user is creating or staging
  an experiment.
- Pair with `reader-workbench-gardening` when the user is reorganizing DOP
  surfaces or reducing monolith pressure.
- Pair with `code-change-discipline` for registry, CLI, or test changes.
- Pair with `harness-engineering` when docs, skills, or CLI evidence routes need
  stronger deterministic checks.

## Required Inputs

- target dataset, protocol, DOP registry entry, or docs/skill surface
- whether the task is classification, intake support, or DOP maintenance
- raw input provenance or representative protocol id when classifying data
- explicit constraints around schema changes, generated outputs, and external
  systems

Clarification policy:

- ask only when missing context would change data-class selection, metadata
  interpretation, or whether a change belongs in code versus docs
- otherwise proceed with explicit assumptions and record them

## Success Criteria

- exactly one DOP mode is selected before work starts
- automation-facing DOP facts come from `uv run reader dop ...` or
  `src/reader/workbench/dop/`
- docs and skill routes point to owned surfaces instead of duplicating registry
  facts
- stop conditions block ambiguous metadata instead of encouraging inference
- external DOP claims map to dated rows in
  [External sources](./references/external-sources.md)

## Workflow

1. Choose one mode: `classification`, `intake-support`, or `maintenance`.
2. Start with
   [docs/guides/data_operations_plan.md](../../docs/guides/data_operations_plan.md)
   and load only the smallest referenced page needed for the decision.
3. Use [Operating model](../../docs/guides/data_operations_plan/operating_model.md)
   when the task changes DOP ownership, repo routing, or maintenance policy.
4. Use `uv run reader dop classes --format json` for data-class and
   protocol-candidate facts instead of parsing prose tables.
5. Use `uv run reader dop ready-specs --format json` for ready-spec gates and
   evidence expectations.
6. For experiment creation, route into
   [Experiment bootstrap](../../docs/guides/experiment_bootstrap.md) after the
   data class and metadata stop conditions are known.
7. For DOP maintenance, use [Workflow reference](./references/workflow.md) and
   keep each fact in its owned surface.
8. Use [Endpoint contracts](./references/endpoint-contracts.md) when changing
   docs, skills, registry, or CLI evidence routes.
9. Use [Test matrix](./references/test-matrix.md) for trigger, functional, and
   deterministic checks.
10. Use [External sources](./references/external-sources.md) when external DOP
   claims shape the update.

## Guardrails

- Do not infer well identity, treatment meaning, channel semantics, control
  interpretation, or source provenance to make an experiment appear ready.
- Do not treat DOP data classes as new `reader/v8` schema fields.
- Do not duplicate the DOP registry into prose when automation needs the fact.
- Do not copy generated outputs as source material for a new experiment.
- Do not turn this skill into broad workbench gardening; route architecture
  maintenance to `reader-workbench-gardening`.

## Required Deliverables

- chosen DOP mode: classification, intake support, or maintenance
- data class or registry surface reviewed
- metadata minimums and stop conditions checked
- source evidence used for any external DOP claim
- changed files or explicit audit-only result
- verification evidence and skipped checks
- assumptions, open questions, and residual risks

## Output Contract

Return:

1. Decision summary
   - mode, target surface, selected data class or maintenance surface, and
     assumptions
2. DOP contract check
   - requirements, design, configuration, and instructions concerns touched
3. Evidence bundle
   - registry commands, docs or skill routes, source rows, and stop conditions
4. Change summary
   - files changed or audit-only finding, with ownership boundary notes
5. Verification bundle
   - commands run, pass/fail status, skipped checks, and residual risks

## Trigger Tests

Should trigger:

- "Classify this dataset against the reader DOP."
- "Check whether the DOP registry and docs are aligned."
- "Update the reader DOP skill and source evidence."
- "Add a DOP data-class route for a new protocol."
- "Audit DOP metadata minimums before experiment bootstrap."

Should not trigger:

- "Interpret these generated plots."
- "Create a new experiment from this workbook."
- "Refactor the protocol compiler."
- "Clean up my Downloads folder."
- "Hand-edit files under outputs/."

## Troubleshooting

- Data class fits multiple routes:
  - choose the strictest class whose protocol assumptions match the assay, then
    record why broader classes were rejected
- Metadata is missing:
  - keep intake blocked and ask for the missing semantic fact
- DOP docs and registry disagree:
  - update the owned source first, then repair summaries and route checks
- The request becomes architecture maintenance:
  - route to `reader-workbench-gardening` and preserve this skill as DOP policy
    routing

## Additional Resources

- [Workflow reference](./references/workflow.md)
- [Endpoint contracts](./references/endpoint-contracts.md)
- [Test matrix](./references/test-matrix.md)
- [External sources](./references/external-sources.md)
