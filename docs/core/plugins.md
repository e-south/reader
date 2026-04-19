# Extending reader with plugins

Plugins are the execution layer of `reader`. They are for maintainers, not the
public authoring surface. Experiment authors should stay in
[`reader/v7`](./pipeline.md), protocol selection, and protocol-owned output
choices. Public configs do not list raw plugin ids.

## Plugin categories

Built-in plugin implementations live under
[`src/reader/plugins/`](../../src/reader/plugins/):

- [`ingest/`](../../src/reader/plugins/ingest/)
  read raw files into tidy tables
- [`transform/`](../../src/reader/plugins/transform/)
  derive or enrich dataframe records
- [`validator/`](../../src/reader/plugins/validator/)
  enforce or promote contracts
- [`plot/`](../../src/reader/plugins/plot/)
  render file-bundle plot outputs
- [`export/`](../../src/reader/plugins/export/)
  write file-bundle export artifacts

Built-in registration is explicit in
[`src/reader/workbench/assets/plugin_manifest.py`](../../src/reader/workbench/assets/plugin_manifest.py).
The runtime does not discover built-ins by scanning package trees.

External plugins are still supported. `reader` loads third-party plugin
descriptors from the [`reader.plugins` entry-point group](../../src/reader/workbench/registry.py)
after it registers the built-in manifest.

## Ownership rules

A good plugin is thin orchestration.

- Keep domain parsing and math in
  [`src/reader/domains/`](../../src/reader/domains/).
- Keep shared plotting mechanics in
  [`src/reader/plotting/`](../../src/reader/plotting/).
- Keep shared ingest autodiscovery in
  [`src/reader/plugins/ingest/discovery_policy.py`](../../src/reader/plugins/ingest/discovery_policy.py)
  or [`src/reader/plugins/ingest/_discovery.py`](../../src/reader/plugins/ingest/_discovery.py),
  not duplicated across adapters.
- Keep plugin metadata in the asset manifest and ontology types:
  [`src/reader/workbench/assets/types.py`](../../src/reader/workbench/assets/types.py)
  and [`src/reader/workbench/ontology.py`](../../src/reader/workbench/ontology.py).
- Keep protocol-facing defaults and output selection in
  [`src/reader/protocols/`](../../src/reader/protocols/), not in ad hoc CLI or
  docs-only conventions.

If maintainers need to widen the public config just to reach a plugin, the
design is probably heading in the wrong direction.

## How a plugin reaches users

The maintainer path is:

1. Implement the plugin class under `src/reader/plugins/<category>/`.
2. Register it in the built-in manifest, or expose an external
   [`reader.plugins` entry point](../../src/reader/workbench/registry.py) that
   resolves to an `AssetDescriptor`.
3. Wire it into a
   [`protocol compiler`](../../src/reader/protocols/compiler.py) or recipe so
   the protocol owns when it runs and what semantic output it represents.
4. Expose it through protocol inputs, analysis knobs, plot profiles, or export
   artifacts rather than raw plugin ids in user config.

That last step matters. `reader` is intentionally protocol-driven. A new plugin
is not a public feature until a protocol gives it a semantic role.

## Minimal implementation pattern

```python
from typing import Any

from reader.workbench.ports import dataframe_output, file_path_input
from reader.workbench.registry import Plugin, PluginConfig


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
[`reader.workbench.ports`](../../src/reader/workbench/ports/), not string
conventions.

- Optional inputs use `optional=True`, not `?` suffixes.
- Dataframe ports declare a contract id or `None`.
- Plot/export outputs use `file_bundle` ports.
- Removed legacy conventions such as `"none"` and `"files"` are not valid.

Runtime validation then checks reads, writes, and contract compatibility before
execution.

## Plot and export guidance

Plot and export plugins are invoked through protocol-owned surfaces:

- `reader plot`
- `reader export`
- `protocol.outputs.plots`
- `protocol.outputs.exports`

They should be deterministic, assertive, and provenance-friendly:

- read only declared inputs
- fail fast on missing required columns or invalid config
- write file bundles under `outputs/plots` or `outputs/exports`
- let records/manifests explain what was produced

For file output helpers, start with
[`src/reader/plotting/sinks.py`](../../src/reader/plotting/sinks.py).

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

- [Configuring `reader/v7`](./pipeline.md)
- [reader specification](./spec.md)
- [Architecture](../../ARCHITECTURE.md)
- [Crosstalk pairs](../lib/crosstalk_pairs.md)
