---
doc_id: reader-package-namespace-migration
surface: migration-guide
owner: reader-maintainers
last_verified: 2026-07-30
summary: Public migration from the retired reader Python namespace to reader_workbench without compatibility shims.
---

# Package namespace migration

Reader's distribution remains `reader-workbench`, but its public Python import
package is `reader_workbench`. Update integrations before installing this
release:

| Consumer surface | Before | Now |
| --- | --- | --- |
| Python imports | `reader.*` | `reader_workbench.*` |
| External plugin entry-point group | `reader.plugins` | `reader_workbench.plugins` |
| Installed command | `reader` | `reader` |

For example, replace `from reader.api import open_experiment` with
`from reader_workbench.api import open_experiment`. External plugin
distributions must publish their entry points under
`reader_workbench.plugins`.

There is no `reader` import compatibility shim. This prevents Reader from
claiming the generic `reader` package namespace on PyPI.

Only Python imports and the external plugin discovery group change. Persisted
wire identities beginning with `reader.*` remain unchanged, including values
such as `reader.cli/v1` and
`reader.domains.time_series.temporal_reduction.v1`. Do not rewrite those
identities as Python package names.
