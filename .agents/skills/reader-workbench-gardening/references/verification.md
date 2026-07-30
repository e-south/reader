# Verification

Choose the smallest bundle that proves the gardening cycle did not weaken the
workbench.

## Docs-only or routing-only changes

```bash
uv run reader maintain skills
uv run reader maintain docs
git diff --check
```

Also confirm that changed routes point to current canonical docs and that any
embedded commands still match the current CLI surface.

## Skill and maintainer-doc changes

```bash
uv run reader maintain skills
uv run reader maintain docs
git diff --check
```

If the changed guidance includes concrete commands, run the smallest command in
scope to confirm it still matches reality.

## Code, CLI, or contract changes

```bash
uv run ruff check .
uv run ruff format . --check
uv run pytest -q <targeted-tests>
test -d src/reader_workbench && uv run python -m compileall -q src/reader_workbench
git diff --check
```

Add a representative CLI preflight bundle when the change affects runtime or
maintainer-facing command surfaces:

```bash
uv run reader ls --root experiments --details --readiness --format json
uv run reader inspect <config|dir|index> --format json
uv run reader explain <config|dir|index> --format json
uv run reader validate <config|dir|index> --no-files --format json
uv run reader run <config|dir|index> --dry-run --format json
```

## End-to-end or experiment-surface changes

Start with the code and CLI bundle above, then run the portable integration
subset and one named local experiment when its ignored inputs are available:

```bash
uv run pytest -q -m integration
uv run reader validate <config|dir|index>
uv run reader run <config|dir|index> --dry-run
uv run reader verify <config|dir|index>
```

If plots, exports, or notebooks changed, add the matching `reader plot --list`,
`reader export --list`, `reader records`, or notebook mode command for that
experiment.

## Continue into repo maintenance when

- branch state or remote publish readiness is part of the task
- CI workflow behavior or policy changed
- the gardening slice now spans multiple unrelated repo surfaces
- the change needs the broader maintainer workflow in
  `docs/repo-maintenance.md`

## Evidence expectations

- name the experiment or protocol used for representative CLI checks
- name the mode used for the gardening cycle
- name which endpoint contracts were in scope
- state which checks were skipped and why
- do not claim success from filesystem shape alone when records or manifests
  are available
