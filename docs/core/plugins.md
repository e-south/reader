---
doc_id: reader-plugin-development
surface: maintainer-guide
owner: reader-maintainers
last_verified: 2026-07-28
summary: Maintainer guide for adding mechanical Reader plugins behind protocol-owned semantics and typed contracts.
---

# Extending reader with plugins

Plugins are the execution layer of `reader`. They are for maintainers, not the
public authoring surface. Experiment authors should stay in
[`reader/v8`](./pipeline.md), protocol selection, and protocol-owned output
choices. Public configs do not list raw plugin ids.

## Plugin categories

Built-in plugin implementations live under
[`src/reader_workbench/plugins/`](../../src/reader_workbench/plugins/):

- [`ingest/`](../../src/reader_workbench/plugins/ingest/)
  read raw files into tidy tables
- [`transform/`](../../src/reader_workbench/plugins/transform/)
  derive or enrich dataframe records
- [`validator/`](../../src/reader_workbench/plugins/validator/)
  enforce or promote contracts
- [`plot/`](../../src/reader_workbench/plugins/plot/)
  render file-bundle plot outputs
- [`export/`](../../src/reader_workbench/plugins/export/)
  write file-bundle export artifacts

Built-in registration is explicit in
[`src/reader_workbench/plugins/catalog.py`](../../src/reader_workbench/plugins/catalog.py), with
category manifests beside the implementations under `reader_workbench.plugins`. The
runtime injects those descriptors into the generic workbench registry; it does
not discover built-ins by scanning package trees.

The registry has a `reader_workbench.plugins` entry-point hook for coordinated
maintainer integrations. It is not a standalone public plugin SDK: the current
descriptor and port imports live under `reader_workbench.workbench`, and an entry point
cannot add `reader/v8` protocol semantics by itself. Do not advertise an
external package as a Reader feature until Reader protocol code owns its
authoring and compilation path.

## Ownership rules

A good plugin is thin orchestration.

- Keep domain parsing and math in
  [`src/reader_workbench/domains/`](../../src/reader_workbench/domains/).
- Keep shared plotting mechanics in
  [`src/reader_workbench/plotting/`](../../src/reader_workbench/plotting/).
- Keep shared ingest autodiscovery in
  [`src/reader_workbench/plugins/ingest/discovery_policy.py`](../../src/reader_workbench/plugins/ingest/discovery_policy.py)
  or [`src/reader_workbench/plugins/ingest/_discovery.py`](../../src/reader_workbench/plugins/ingest/_discovery.py),
  not duplicated across adapters.
- Keep built-in plugin metadata in the plugin-owned catalog and shared ontology
  types:
  [`src/reader_workbench/plugins/catalog.py`](../../src/reader_workbench/plugins/catalog.py),
  [`src/reader_workbench/workbench/assets/types.py`](../../src/reader_workbench/workbench/assets/types.py)
  and [`src/reader_workbench/workbench/ontology.py`](../../src/reader_workbench/workbench/ontology.py).
- Keep protocol-facing defaults and output selection in
  [`src/reader_workbench/protocols/`](../../src/reader_workbench/protocols/), not in ad hoc CLI or
  docs-only conventions.

If maintainers need to widen the public config just to reach a plugin, the
design is probably heading in the wrong direction.

## How a plugin reaches users

The maintainer path is:

1. Implement the plugin class under `src/reader_workbench/plugins/<category>/`.
2. Register it in the built-in manifest. A coordinated external integration
   may instead expose a `reader_workbench.plugins` entry point that resolves to an
   `AssetDescriptor`.
3. Wire it into a
   [`protocol compiler`](../../src/reader_workbench/protocols/compiler.py) so the protocol
   owns when it runs and what semantic output it represents.
4. Expose it through protocol inputs, analysis knobs, plot profiles, or export
   artifacts rather than raw plugin ids in user config.

That last step matters. `reader` is intentionally protocol-driven. A new plugin
is not a public feature until a protocol gives it a semantic role.

For the external registry hook, the entry-point name must equal the descriptor
plugin id in `category/key` form. Category-scoped registry loads skip unrelated
entry points before importing their packages. This avoids import cost and side
effects during commands that request only one plugin category.

## Minimal implementation pattern

```python
from typing import Any

from reader_workbench.workbench.ports import dataframe_output, file_path_input
from reader_workbench.workbench.registry import Plugin, PluginConfig


class MyCfg(PluginConfig):
    pass


class MyIngest(Plugin):
    ConfigModel = MyCfg

    @classmethod
    def input_ports(cls):
        return {"raw": file_path_input("raw")}

    @classmethod
    def output_ports(cls):
        return {"df": dataframe_output("df", "tidy.v1")}

    def run(self, ctx, inputs: dict[str, Any], cfg: MyCfg):
        del ctx, cfg
        return {"df": parse_my_format(inputs["raw"])}
```

Register it in the manifest:

```python
build_plugin_asset(
    plugin_id="ingest/my_format",
    semantics=PluginSemantics(
        domain="plate_reader",
        family="workbook_ingest",
        summary="Parse a custom workbook format into tidy traces.",
    ),
    plugin_cls=MyIngest,
)
```

Then wire it through a protocol compiler:

```python
return CompiledProtocolPlan(
    semantic_program=protocol.semantic_program(),
    pipeline=(
        PluginStepDecl(
            id="ingest",
            plugin="ingest/my_format",
            reads={"raw": FileInputDecl(path="./inputs/run001.ext")},
        ),
    ),
)
```

The important contract is not the example code. It is the layering:
domain logic -> plugin adapter -> asset registration -> protocol-owned exposure.

## Port and contract rules

Plugin I/O is declared through
[`reader_workbench.workbench.ports`](../../src/reader_workbench/workbench/ports/), not string
conventions.

- Optional inputs use `optional=True`, not `?` suffixes.
- Dataframe outputs declare a registered contract id. A dataframe input may
  omit its contract only when it intentionally accepts any registered
  dataframe artifact.
- Plot/export outputs use `file_bundle` ports.
- Port contract ids such as `"none"` and `"files"` are invalid.

Runtime validation then checks reads, writes, and contract compatibility before
execution.

An ingest plugin that discovers input files may implement
`resolve_missing_file_inputs`. Use an optional `file_path` port for one file or
an optional `file_set` port for a nonempty tuple. The engine forbids replacing
an explicit graph binding, confines every resolved regular file to the
experiment root after symlink resolution, and records each exact file as input
evidence. The plugin's `run` method consumes the resolved path or tuple; it does
not discover the files a second time.

## Plot and export guidance

Plot and export plugins are invoked through protocol-owned surfaces:

- `reader plot`
- `reader export`
- `protocol.outputs.plots`
- `protocol.outputs.exports`

They should be deterministic, assertive, and provenance-friendly:

- read only declared inputs
- fail fast on missing required columns or invalid config
- return a nonempty explicit file output for every declared file port; Reader
  does not infer outputs from directory changes
- write plot bundles under `outputs/plots` and resolve export paths relative to
  `outputs/exports`
- let records/manifests explain what was produced

Every newly persisted plot path has a typed description. The normal happy path
is to give the compiled plot step the same id as its
`ProtocolFigureSpec`; Reader then persists that figure summary for every file
in the figure bundle. If an unmapped plugin writes files with different
meanings through `FigurePlotPlugin`, describe each public `PlotFigure` before
the shared sink writes it:

```python
from reader_workbench.plotting.sinks import PlotFigure

return [
    PlotFigure(
        fig=kinetics_figure,
        filename="kinetics",
        description="Reporter kinetics over assay time.",
    ),
    PlotFigure(
        fig=summary_figure,
        filename="summary",
        description="Endpoint summary by treatment.",
    ),
]
```

The shared plot adapter converts those descriptions into public
[`PathDescription`](../../src/reader_workbench/workbench/records/model.py) values after
saving. A lower-level custom `Plugin` may return those values directly:

```python
from reader_workbench.workbench.records import PathDescription

return {
    "artifacts": [
        PathDescription(path=kinetics_path, description="Reporter kinetics over assay time."),
        PathDescription(path=summary_path, description="Endpoint summary by treatment."),
    ]
}
```

An unmapped single-file plot may use its registered plugin summary. An unmapped
multi-file plot without complete explicit descriptions fails. Reader rejects
missing, duplicate, partial, and unmatched path-description metadata and never
infers scientific meaning from filenames. New file-bundle records also reject
missing paths and directories instead of cataloging them as files. Export
bundles keep one operational plugin description because their files are not
treated as scientific figures.

For file output helpers, start with
[`src/reader_workbench/plotting/sinks.py`](../../src/reader_workbench/plotting/sinks.py).

## Useful inspection routes

```bash
uv run reader plugins
uv run reader plugins --category plot
uv run reader plugins --protocol plate_reader/dual_reporter_screen --category transform
uv run reader protocols <protocol-id>
uv run reader explain <config|dir|index>
```

Use `reader plugins` to inspect the registry. Use `reader protocols` and
`reader explain` to verify that a protocol actually exposes the new plugin in a
maintainable way.

## Related docs

- [Configuring `reader/v8`](./pipeline.md)
- [reader specification](./spec.md)
- [Architecture](../../ARCHITECTURE.md)
- [Crosstalk pairs](../lib/crosstalk_pairs.md)
