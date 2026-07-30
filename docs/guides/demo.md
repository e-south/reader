---
doc_id: reader-end-to-end-demo
surface: tutorial
owner: reader-maintainers
last_verified: 2026-07-10
summary: Concrete Reader walkthrough from experiment discovery through records, plots, exports, and verification.
---

# End-to-end demo

This walkthrough shows the canonical route from discovery to validated records,
then optional plots, exports, and notebooks. It assumes an existing configured
experiment; replace `<experiment>` with its experiment-directory name. Prefer
an explicit path because it is deterministic and easy to replay.

---

1) Find experiments

```bash
uv run reader ls
```

2) Inspect the plan (no execution)

```bash
uv run reader explain ./experiments/<experiment>/config.yaml
```

3) Validate the config + inputs

```bash
uv run reader validate ./experiments/<experiment>/config.yaml
```

4) Run the pipeline (records only)

```bash
uv run reader run ./experiments/<experiment>/config.yaml
```

5) Inspect records

```bash
uv run reader records ./experiments/<experiment>/config.yaml
```

6) Generate plots

```bash
uv run reader plot ./experiments/<experiment>/config.yaml --list
uv run reader plot ./experiments/<experiment>/config.yaml
```

7) Generate exports

```bash
uv run reader export ./experiments/<experiment>/config.yaml --list
uv run reader export ./experiments/<experiment>/config.yaml
```

8) Scaffold a notebook

```bash
uv run reader notebook ./experiments/<experiment>/config.yaml
```

Reader always generates the canonical record-driven notebook; assay-specific
views arrive through registered plot and export records.

9) Verify current artifacts and provenance

```bash
uv run reader verify ./experiments/<experiment>/config.yaml
```

Verification is the final read-only gate. It checks the current record catalog,
artifact digests, contracts, and captured upstream revisions.

See the [Notebooks guide](./notebooks.md) for opening and dependency setup.
