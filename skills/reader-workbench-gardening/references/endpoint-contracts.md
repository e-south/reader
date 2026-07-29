# Endpoint contracts

Use these endpoint contracts when hardening or validating
`reader-workbench-gardening`.

Start with these three endpoints:

## `knowledge-integrity`

This skill should keep repo-local maintainer guidance current, cross-linked,
and anchored to canonical docs.

Required evidence:

- [docs/guides/workbench_gardening.md](../../../docs/guides/workbench_gardening.md)
  remains the primary workflow
- [skills/reader-workbench-gardening/SKILL.md](../SKILL.md) routes to the guide
  instead of duplicating it
- [references/external-sources.md](./external-sources.md) records official
  source rows when external claims shape the contract
- `uv run reader maintain docs` passes after docs or routing edits

## `autonomy-capability`

This skill should be executable by an agent with a small, deterministic read
surface and explicit deliverables.

Required evidence:

- top-level `SKILL.md` stays router-first and points deeper detail into
  `references/`
- a gardening cycle can choose one explicit mode and one bounded slice
- [references/test-matrix.md](./test-matrix.md) provides should/should-not
  trigger prompts and consistency checks
- `uv run reader maintain skills` passes

## `architecture-invariants`

This skill must reinforce `reader`'s workbench architecture instead of
smearing boundaries across docs and instructions.

Required evidence:

- the skill routes new-experiment intake to `reader-experiment-bootstrap`
- the skill routes publish, branch, or CI topology work to
  [docs/repo-maintenance.md](../../../docs/repo-maintenance.md)
- the skill keeps generated outputs out of scope
- the guide points to `ARCHITECTURE.md`, `DESIGN.md`, `QUALITY.md`, and
  `RELIABILITY.md` as canonical invariants
