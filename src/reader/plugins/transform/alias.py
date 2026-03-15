"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/alias.py

Alias mappings for categorical columns. Either replace in-place or create
<column>_alias columns. Prints a succinct per-column summary of applied aliases.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from reader.core.registry import Plugin, PluginConfig
from reader.core.workbench import PluginSemantics
from reader.plugins.transform._labeling import apply_label_mappings


class AliasCfg(PluginConfig):
    """
    mappings:
      <column_name>:
        <raw_value>: <alias_value>
        ...
    in_place:      if true, mutate <column_name> directly; else create <column_name>_alias
    case_insensitive: map using casefold() on incoming values (keys in 'aliases' are matched case-insensitively)
    """

    mappings: Mapping[str, Mapping[str, str]] | None = None
    in_place: bool = False
    case_insensitive: bool = True
    suffix: str = "_alias"


class AliasTransform(Plugin):
    key = "alias"
    category = "transform"
    semantics = PluginSemantics(
        category="transform",
        domain="generic",
        family="label_enrichment",
        summary="Add alias columns for configured categorical metadata.",
        tags=("aliases", "annotation"),
    )
    ConfigModel = AliasCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}  # works equally well on annotated plate-reader tables

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contract_surfaces(cls) -> Mapping[str, object]:
        return cls.passthrough_output_contract_surfaces(
            passthrough={"df": "df"},
            promoted_examples={"df": ("plate_reader.annotated.v1",)},
        )

    def resolve_output_contracts(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_contracts(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

    def run(self, ctx, inputs, cfg: AliasCfg):
        if cfg.mappings is None:
            raise ValueError("alias: provide with.mappings")
        if not isinstance(cfg.mappings, Mapping):
            raise ValueError("alias: mappings must be a mapping of column -> {raw: alias}")
        mappings = {str(col): mapping for col, mapping in cfg.mappings.items()}
        output_names = {str(col): f"{col}{cfg.suffix}" for col in mappings}
        return {
            "df": apply_label_mappings(
                ctx=ctx,
                df=inputs["df"],
                mappings=mappings,
                output_names=output_names,
                in_place=cfg.in_place,
                case_insensitive=cfg.case_insensitive,
                label="alias",
            )
        }
