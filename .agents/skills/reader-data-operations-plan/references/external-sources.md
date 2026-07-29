# External Sources

This skill is grounded first in repository sources:

- [Data Operations Plan](../../../../docs/guides/data_operations_plan.md)
- [Operating model](../../../../docs/guides/data_operations_plan/operating_model.md)
- [DOP registry](../../../../src/reader/workbench/dop/)
- [Experiment bootstrap](../../../../docs/guides/experiment_bootstrap.md)
- [Repo change gate](../../../../docs/repo-change-gate.md)

External source rows:

| URL | Retrieved | Mapped update |
| --- | --- | --- |
| https://merelogic.net/data_operations_plans/how | 2026-05-01 | Source for the DOP framing: group long-tail assays into simple data classes, separate requirements/design/configuration/instructions, keep instructions easy to follow, and maintain the plan from real use. |
| https://merelogic.net/static/js/main.18e51494.js.map | 2026-05-01 | Used to verify the JS-rendered page text for the DOP component and maintenance sections because the public route returns a JavaScript app shell. |

Use external sources to shape repo-local guidance, not to import an
organization-wide DOP wholesale. Claims in the skill should stay paraphrased and
mapped to `reader` behavior, with the repo docs and registry remaining the
operating source of truth.
