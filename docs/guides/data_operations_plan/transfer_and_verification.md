# Transfer and Verification

Use this page after the data class and metadata contract are clear.

Return to the [Data Operations Plan](../data_operations_plan.md) when you only
need the overview.

## Transfer Rules

- Put raw inputs in `inputs/` with the original filename when practical.
- Keep hand-authored notebooks in `notebooks/`; generated scaffolds belong in
  `outputs/notebooks/`.
- Keep generated records, plots, exports, and manifests under `outputs/`.
- Use explicit `resources` entries for files or directories consumed by the
  compiled plan.
- When materializing from Google Drive or another external system, record the
  source and staged path in the handoff.
- Do not copy generated outputs from an old experiment into a new one. Copy
  config/metadata intent only when the new run is semantically close, then
  regenerate outputs.
- If source ownership, source location, or transfer status is unknown, leave the
  intake blocked instead of inventing a path that makes preflight look cleaner.

## Verification Commands

Use the cheapest check that answers the next question, then broaden only when
the surface is ready:

```bash
uv run reader validate <config|dir|index> --no-files
uv run reader validate <config|dir|index>
uv run reader run <config|dir|index> --dry-run --format json
uv run reader plot <config|dir|index> --list
uv run reader export <config|dir|index> --list
uv run reader run <config|dir|index>
uv run reader records <config|dir|index>
```

## Evidence Bar

Verification is complete only when it proves:

- the config schema and protocol binding are valid;
- declared files/resources exist or the experiment is intentionally non-active;
- the compiled pipeline, plots, exports, and notebooks match the intended data
  class;
- `outputs/manifests/records.json` records the generated dataframe and
  file-bundle evidence; and
- unresolved metadata assumptions are visible in the final handoff.
