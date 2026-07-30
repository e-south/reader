---
doc_id: reader-dop-operating-model
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-07-29
summary: Decision sequence, ownership boundaries, and stop conditions for a Reader Data Operations Plan.
---

# Operating Model

Use this page when changing the `reader` Data Operations Plan overlay or when a
new intake task does not fit cleanly into the existing data-class pages.

Return to the [Data Operations Plan](../data_operations_plan.md) when you only
need the overview.

## Component Boundaries

Merelogic frames a Data Operations Plan as four related documents. In `reader`,
those concerns map to repo-local surfaces instead of becoming one large policy
document.

| DOP concern | `reader` surface | Contract |
| --- | --- | --- |
| Requirements | Intake prompt, handoff notes, and experiment context | Capture why the data matters, who will consume it, and what ambiguity would make it unusable later. Do not store organization-wide requirements in `reader/v8`. |
| Design | Standard experiment layout, workbench architecture, and preflight/run/verify loop | Keep the operating path stable: `config.yaml`, `inputs/`, `notebooks/`, generated `outputs/`, and manifest-backed records. |
| Configuration | DOP registry, protocol catalog, config files, sample maps, and canonical labels | Define data classes, protocol candidates, metadata minimums, naming expectations, and stop conditions in one owned place. |
| Instructions | DOP guide pages, experiment bootstrap guide, repo-local skills, and CLI commands | Give humans and agents the shortest safe next step without duplicating every detail into one monolith. |

## Ownership Rules

- `src/reader_workbench/workbench/dop/` owns the machine-readable data-class and
  ready-spec registry.
- `docs/guides/data_operations_plan/` owns human-facing explanations,
  checklists, and examples.
- `.agents/skills/reader-data-operations-plan/` owns agent routing for DOP
  classification and maintenance.
- `src/reader_workbench/protocols/` owns executable assay semantics.
- `reader/v8` config owns authored experiment intent, not lab-wide policy.
- `outputs/manifests/records.json` owns generated evidence after execution.

When a fact must be consumed by automation, put it in the registry first and
let docs summarize it. When a fact is explanatory or procedural, keep it in the
smallest guide page that owns that decision.

## Change Contract

Use this order for a DOP change:

1. Identify whether the change affects requirements, design, configuration, or
   instructions.
2. Update the smallest owned surface.
3. Keep any registry change read-only and fail-fast until a later slice proves
   it should affect execution.
4. Link from the overview or skill only when the route is recurring.
5. Run the docs, skill, and targeted registry checks before broad tests.

Do not use a DOP change to widen `reader/v8`, restore removed config keys, or
encode a guessed metadata interpretation. If a new assay needs different
execution semantics, add or change a protocol after the intake contract is
clear.

## Maintenance Triggers

Update the DOP overlay when:

- a protocol is added, removed, or renamed;
- an intake task repeatedly stops on the same metadata ambiguity;
- a data class needs a different prescribed order or stop condition;
- transfer paths change for raw inputs or metadata files;
- generated evidence no longer answers the review question; or
- a long-tail assay graduates from draft/template handling to an executable
  protocol.

## Verification

For DOP docs or skill changes:

```bash
uv run reader maintain skills
uv run reader maintain docs
git diff --check
```

For registry or CLI changes, add:

```bash
uv run pytest -q src/reader_workbench/tests/workbench/test_dop_registry.py
uv run reader dop classes --format json
uv run reader dop ready-specs --format json
```
