# Repo-local skills

Repo-local skills live here when a recurring agent workflow is specific to
`reader` and worth keeping as a reusable pattern.

Guidelines:

- Keep `AGENTS.md` short and route to a skill or doc when the task is recurring.
- Treat the matching guide in `docs/` as the primary workflow. The skill should stay thin and point to it.
- Keep detailed workflow guidance in `docs/` and let skills point to those docs.
- Prefer one narrowly-owned skill over a broad catch-all.
- Pair repo-local skills with existing global skills when the task also needs
  external tooling such as Google Workspace access or spreadsheet editing.

Current repo-local skill:

- [`reader-data-operations-plan`](./reader-data-operations-plan/SKILL.md):
  DOP data-class classification, DOP registry/docs alignment, and DOP
  maintenance routing.
  Primary workflow: [docs/guides/data_operations_plan.md](../docs/guides/data_operations_plan.md)
- [`reader-experiment-bootstrap`](./reader-experiment-bootstrap/SKILL.md): new experiment intake, metadata mapping, Drive-backed input staging, and local experiment audits.
  Primary workflow: [docs/guides/experiment_bootstrap.md](../docs/guides/experiment_bootstrap.md)
- [`reader-workbench-gardening`](./reader-workbench-gardening/SKILL.md): maintain `reader`'s information architecture, semantic boundaries, and verification surfaces without locking the repo into one assay family.
  Primary workflow: [docs/guides/workbench_gardening.md](../docs/guides/workbench_gardening.md)
