---
name: reader-experiment-bootstrap
description: Bootstraps `reader` experiment workspaces by selecting a matching protocol/template, materializing raw assay inputs, building sample metadata, and running the preflight/run/verify loop. Use when creating a new experiment, cloning prior assay semantics into a new run, or auditing local experiments. Do not use for generic result interpretation, ad hoc config edits with no bootstrap objective, or hand-editing generated outputs.
metadata:
  version: 0.2.0
  category: scientific-workbench
  tags: [reader, experiments, metadata, google-drive, audit]
---

# Reader Experiment Bootstrap

## Purpose

Create or audit `reader` experiment workspaces without re-deriving the same
protocol, metadata, and verification decisions every time.

## Scope

In scope:

- selecting the nearest matching `reader` experiment or protocol template
- materializing raw inputs into a new experiment workspace
- building or rewriting metadata maps for the new run
- running `reader` preflight, execution, plot, and export steps
- auditing the local experiment list under `experiments/`

Out of scope:

- hand-editing generated `outputs/`
- generic scientific interpretation disconnected from experiment setup
- destructive data cleanup

## Skill composition

- Pair with `gws-cli` when the raw input lives on Google Drive.
- Pair with `xlsx` when workbook inspection or metadata workbook rewriting is
  required.
- Pair with `pragmatic-programming-principles` when changing experiment
  config or audit structure.

## Inputs

- target experiment date/slug or enough context to create one
- assay family or a nearby prior experiment
- raw input location
- known metadata semantics: layout, treatments, controls, aliases

Clarification policy:

- ask for missing metadata only when it changes well identity, treatment
  meaning, or control semantics
- otherwise proceed with explicit assumptions and record them

## Workflow

1. Treat [docs/guides/experiment_bootstrap.md](../../docs/guides/experiment_bootstrap.md)
   as the primary workflow. This skill stays thin and routes to that guide.
2. Use [Workflow reference](./references/workflow.md) only for the concrete
   command list while following the guide.
3. Prefer JSON command output whenever another tool or agent will consume the
   result.
4. For repo-wide local experiment checks, use
   `uv run python tools/audit_local_experiments.py [--years <yyyy> [<yyyy> ...]]`
   and add `--include-non-active` only when draft/template configs are
   intentionally in scope.
5. Prefer `reader notebook --mode none` when the task needs a generated review
   scaffold without launching Marimo during intake or audit.

## Guardrails

- Do not invent plate semantics to get past metadata ambiguity.
- Do not treat repo fixture tests as proof that the local experiment list is
  healthy.
- Do not copy generated outputs between experiments.
- Do not hide channel/schema drift; encode it explicitly in config.

## Required Deliverables

- chosen template/protocol and why
- raw input provenance and staged path
- metadata contract summary and unresolved assumptions
- preflight evidence
- execution + verification evidence
- local experiment audit summary when the task is repo-wide

## Output Contract

Return:

1. Decision summary
   - target experiment, chosen protocol/template, and copied prior context
2. Metadata contract
   - key columns, control semantics, unresolved assumptions, and explicit user confirmations still needed
3. Preflight bundle
   - commands run, JSON/table evidence, and whether the experiment is blocked, runnable, or draft
4. Execution + verification bundle
   - run/plot/export/notebook steps executed plus records/artifact checks
5. Local audit bundle
   - local experiment audit results when the task is repo-wide

## Trigger Tests

Should trigger:

- "Bootstrap a new reader experiment from this workbook."
- "Find the nearest matching experiment and stage a new run."
- "Audit local experiments under experiments/."
- "Materialize raw inputs and wire metadata for a new reader run."

Should not trigger:

- "Interpret these results."
- "Hand-edit the generated outputs."
- "Tweak one plot label in an existing notebook."

## Troubleshooting

- Missing workbook or Drive provenance:
  - keep the raw-input step explicit and incomplete instead of guessing filenames or source state
- Metadata ambiguity:
  - stop when well identity, treatment meaning, or control semantics are unclear
- Preflight passes but outputs drift:
  - re-run the smallest failing surface and verify `outputs/manifests/records.json` instead of trusting filesystem shape alone

## Additional resources

- [Workflow reference](./references/workflow.md)
- [External sources](./references/external-sources.md)
