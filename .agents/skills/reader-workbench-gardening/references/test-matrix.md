# Test matrix

Use this matrix when validating `reader-workbench-gardening` after edits.

## Trigger checks

Should trigger:

- "Audit `reader` for monolith pressure and assay lock-in."
- "Sync stale maintainer docs back to current runtime behavior."
- "Harden the preflight and JSON surfaces so agents can verify changes faster."
- "Choose one small boundary cut to keep the workbench assay-extensible."

Should not trigger:

- "Bootstrap a new experiment from this workbook."
- "Interpret these assay results."
- "Implement this new plugin."
- "Handle release, branch, or CI publish steps."

## Functional checks

- the top-level skill routes to the primary guide before deeper references
- the skill names one mode per cycle: `audit-only`, `docs-sync`,
  `boundary-hardening`, or `surface-contracts`
- the output contract requires evidence, a selected slice, verification, and
  residual risks
- adjacent routes are explicit for experiment bootstrap, plugin work, and repo
  maintenance

## Deterministic checks

Run:

```bash
uv run reader maintain skills
uv run reader maintain docs
git diff --check
```

For repeated-run consistency, use the same prompt three times and confirm the
response keeps:

- the same mode selection
- the same adjacent-route decision
- the same required deliverable structure

## Content-correctness checks

- external source rows have URL, retrieval date, and mapped update
- external claims are official-source-backed when the skill depends on them
- repo-local invariants are cited from canonical repo docs instead of restated
  from memory
