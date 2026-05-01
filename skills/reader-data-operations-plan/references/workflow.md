# Workflow Reference

Use this reference for DOP classification or DOP maintenance after the
top-level skill has routed the task.

## Mode Selection

- `classification`
  - select a DOP data class, protocol candidate set, and stop conditions for a
    dataset before experiment bootstrap
- `intake-support`
  - check metadata minimums, transfer rules, and ready-spec gates for an
    experiment that is being created or repaired
- `maintenance`
  - update DOP docs, registry facts, or skill routes while keeping ownership
    boundaries explicit

Choose exactly one mode. If a task starts as classification and then becomes
workspace creation, finish the DOP decision and hand off to
`reader-experiment-bootstrap` instead of continuing inside this skill.

## Read Order

1. [Data Operations Plan](../../../docs/guides/data_operations_plan.md)
2. [Operating model](../../../docs/guides/data_operations_plan/operating_model.md)
   when the task changes ownership or maintenance policy
3. [Data classes](../../../docs/guides/data_operations_plan/data_classes.md)
   for class and protocol selection
4. [Metadata minimums](../../../docs/guides/data_operations_plan/metadata_minimums.md)
   for required capture and stop conditions
5. [Transfer and verification](../../../docs/guides/data_operations_plan/transfer_and_verification.md)
   for staging and evidence checks
6. [Experiment bootstrap](../../../docs/guides/experiment_bootstrap.md) only
   after the DOP decision is made

## Command Loop

Use CLI output for stable facts. Start with the smallest command that answers
the current mode:

```bash
uv run reader dop classes --format json
uv run reader dop ready-specs --format json
```

For intake-support work, add only the relevant preflight command:

```bash
uv run reader protocols <protocol-id> --example-config
uv run reader validate <config|dir|index> --no-files --format json
uv run reader validate <config|dir|index> --format json
uv run reader run <config|dir|index> --dry-run --format json
```

Use only the commands that match the current task. Do not execute a run just to
prove DOP classification.

## Ownership Map

| Fact type | Owned by | Maintenance rule |
| --- | --- | --- |
| DOP data-class ids, protocol candidates, stop conditions, transfer rules, ready-spec gates | `src/reader/workbench/dop/` | Update registry and targeted tests first when automation consumes the fact. |
| DOP overview and operator explanation | `docs/guides/data_operations_plan.md` and subpages | Keep pages short and link to the owned source instead of duplicating tables. |
| Experiment creation procedure | `docs/guides/experiment_bootstrap.md` and `reader-experiment-bootstrap` | Start after DOP classification and metadata stop conditions are known. |
| Agent routing | `skills/reader-data-operations-plan/` | Route to docs and CLI; do not become a long policy document. |
| Executable assay semantics | `src/reader/protocols/` | Add or change protocols only when intake policy is not enough. |
| Generated evidence | `outputs/manifests/records.json` | Verify outputs through records; do not use generated files as source inputs. |

## Merelogic Principle Map

- Data classes:
  - use a prescribed order and simple criteria so classification is faster than
    inventing local rules
- Requirements:
  - capture usage expectations, data requirements, consumers, sources, existing
    infrastructure, and user constraints before designing the route
- Design:
  - keep the storage, metadata, compute, version-control, transfer, and
    convention decisions distinct from changing assay details
- Configuration:
  - maintain data categories, canonical names, role references, and technical
    references as living facts
- Instructions:
  - make the first-use path complete and the later-reference path short
- Maintenance:
  - expect updates from real use; do not treat the first DOP pass as final

In `reader`, only the repo-local subset belongs in code or docs. Organization
ownership, training rituals, retention, enterprise catalog policy, and ELN/LIMS
governance remain external unless a later feature gives them concrete behavior.

## Maintenance Checklist

For a DOP maintenance change:

1. Name the changed concern: requirements, design, configuration, or
   instructions.
2. Check whether the fact is automation-facing. If yes, update the DOP registry
   and targeted tests.
3. Update the smallest doc page that explains the behavior.
4. Update skill routes only when the recurring agent workflow changes.
5. Add or refresh source rows when external DOP claims changed.
6. Check [Endpoint contracts](./endpoint-contracts.md) for required evidence.
7. Run the deterministic checks listed in the DOP operating model.

## Stop Conditions

Stop and ask or report blocked when:

- the closest protocol would change assay semantics;
- metadata ambiguity affects identity, treatment, channels, controls, or source
  provenance;
- a requested change would make generated outputs source material;
- the change requires lab-wide policy that `reader` cannot enforce; or
- the change would widen `reader/v7` without a separate schema decision.
