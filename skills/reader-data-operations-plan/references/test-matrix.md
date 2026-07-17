# Test Matrix

Use this matrix when validating `reader-data-operations-plan` after edits.

## Trigger Checks

Should trigger:

- "Classify this plate-reader workbook against the reader DOP."
- "Audit the DOP registry for protocol coverage."
- "Refresh the DOP skill against the Merelogic resource."
- "Check whether DOP metadata minimums are enough before bootstrap."
- "Add DOP guidance for a long-tail assay without changing reader/v8."

Should not trigger:

- "Interpret the assay results."
- "Implement a new transform plugin."
- "Create the experiment workspace now."
- "Refactor the notebook launcher."
- "Delete duplicate files in Downloads."

## Functional Checks

- top-level `SKILL.md` routes to the DOP overview before deeper references
- top-level `SKILL.md` exposes [Endpoint contracts](./endpoint-contracts.md)
- the skill names exactly one mode: `classification`, `intake-support`, or
  `maintenance`
- DOP registry commands are preferred for data-class and ready-spec facts
- generated outputs remain out of scope
- organization-wide policy is recorded as external unless `reader` has concrete
  behavior to validate

## Deterministic Checks

Run:

```bash
uv run python tools/audit_repo_skills.py
uv run python tools/check_docs.py
uv run pytest -q src/reader/tests/repo/test_docs_routes.py
uv run reader dop classes --format json
uv run reader dop ready-specs --format json
git diff --check
```

When registry or CLI behavior changes, add:

```bash
uv run pytest -q src/reader/tests/workbench/test_dop_registry.py
```

## Content-Correctness Checks

- `references/external-sources.md` has URL, retrieved date, and mapped update
  rows for external DOP claims
- Merelogic-derived guidance is paraphrased and mapped to `reader` behavior
  instead of copied wholesale
- repo-local claims point to docs or code surfaces rather than stale memory
  summaries
