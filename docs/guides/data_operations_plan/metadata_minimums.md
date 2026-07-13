---
doc_id: reader-dop-metadata-minimums
surface: operator-reference
owner: reader-maintainers
last_verified: 2026-07-10
summary: Minimum metadata required before Reader can validate or execute each supported data class.
---

# Metadata Minimums

Use this page when building or reviewing `config.yaml`, sample maps, metadata
workbooks, or intake handoffs.

Return to the [Data Operations Plan](../data_operations_plan.md) when you only
need the overview.

## Required Before Execution

- Usage context: why the dataset is being captured, likely downstream
  consumers, and any immediate decision the outputs must support.
- Dataset identity: experiment id, date/slug, assay family, and lifecycle
  (`active`, `draft`, or `template`).
- Raw provenance: original filename, source location, and whether the file was
  copied from local storage, Drive, or an instrument export.
- Assay semantics: instrument/readout family, channel labels, denominators or
  ratios, and protocol-specific analysis choices.
- Sample map: every measured well, position, or sample accounted for when the
  protocol expects complete coverage.
- Controls: blank, reference, negative, positive, paired-control, and treatment
  meanings when the assay uses them.
- Canonical labels: design ids, strain ids, treatments, aliases, orders,
  collections, and logic-map corners.
- Requested outputs: plot profile, export artifacts, and notebook template only
  when they differ from protocol defaults.

## Stop Conditions

Stop intake and ask for clarification when any of these affect interpretation:

- well coordinates or sample positions conflict;
- treatment meaning is incomplete or overloaded;
- blank/control rows are present but their role is unclear;
- channel labels drift from the selected protocol;
- reference design ids or logic-map corners cannot be reconstructed; or
- the closest existing protocol would silently change the assay meaning.

## Storage Rules

Metadata belongs in `config.yaml`, `inputs/` metadata files, or hand-authored
notes under `notebooks/`. Generated artifacts under `outputs/` are evidence,
not the source of truth.

Keep lab-wide owners, approval policy, retention rules, and enterprise catalog
policy outside `reader/v7` until there is a concrete repo-local behavior to
validate. Record those details in the handoff or organization system of record
instead of widening experiment config.

After metadata is stable, move to
[Transfer and verification](./transfer_and_verification.md).
