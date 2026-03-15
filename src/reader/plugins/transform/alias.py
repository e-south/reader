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

import pandas as pd

from reader.core.registry import Plugin, PluginConfig
from reader.core.workbench import PluginSemantics


class AliasCfg(PluginConfig):
    """
    mappings:
      <column_name>:
        <raw_value>: <alias_value>
        ...
    refs:         reference keys under assay.labels
    in_place:      if true, mutate <column_name> directly; else create <column_name>_alias
    case_insensitive: map using casefold() on incoming values (keys in 'aliases' are matched case-insensitively)
    """

    mappings: Mapping[str, Mapping[str, str]] | None = None
    refs: list[str] = []
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

    # ---------------- internal helpers ----------------

    @staticmethod
    def _norm_key(s: str, *, ci: bool) -> str:
        return str(s).casefold() if ci else str(s)

    @staticmethod
    def _build_norm_map(mapping: Mapping[str, str], *, ci: bool) -> Mapping[str, str]:
        if ci:
            return {str(k).casefold(): str(v) for k, v in mapping.items()}
        return {str(k): str(v) for k, v in mapping.items()}

    @staticmethod
    def _log_summary(
        ctx,
        *,
        col: str,
        out_col: str,
        rules_total: int,
        used_rules: int,
        changed_rows: int,
        uniq_vals: int,
        examples: list[str],
        unused_rules_preview: list[str],
    ) -> None:
        unused_count = len(unused_rules_preview)
        ctx.logger.info(
            "alias • [accent]%s[/accent] → %s • rules=%d used=%d changed=%d uniques=%d%s",
            col,
            out_col,
            rules_total,
            used_rules,
            changed_rows,
            uniq_vals,
            f" • unused={unused_count}" if unused_count else "",
        )
        if examples or unused_rules_preview:
            ex_block = "\n".join(f"      • {s}" for s in examples) if examples else "      —"
            unused_block = (
                "\n".join(f"      • {s}" for s in unused_rules_preview) if unused_rules_preview else "      —"
            )
            ctx.logger.debug(
                ("alias details • %s → %s\n   examples:\n%s\n   unused_rule_keys:\n%s"),
                col,
                out_col,
                ex_block,
                unused_block,
            )

    # ---------------- main ----------------

    def run(self, ctx, inputs, cfg: AliasCfg):
        df: pd.DataFrame = inputs["df"].copy()

        if cfg.mappings is not None and cfg.refs:
            raise ValueError("alias: mappings and refs are mutually exclusive")
        if cfg.mappings is None and not cfg.refs:
            raise ValueError("alias: provide with.mappings or with.refs")

        mappings: dict[str, Mapping[str, str]] = {}
        output_names: dict[str, str] = {}
        if cfg.mappings is not None:
            if not isinstance(cfg.mappings, Mapping):
                raise ValueError("alias: mappings must be a mapping of column -> {raw: alias}")
            mappings = {str(col): mapping for col, mapping in cfg.mappings.items()}
            output_names = {str(col): f"{col}{cfg.suffix}" for col in mappings}
        else:
            assay = ctx.assay or {}
            label_specs = (assay.get("labels") or {}) if isinstance(assay, Mapping) else {}
            for ref in cfg.refs:
                label_spec = label_specs.get(ref)
                if label_spec is None:
                    raise ValueError(f"alias: assay.labels missing key '{ref}'")
                if hasattr(label_spec, "model_dump"):
                    label_spec = label_spec.model_dump()
                if not isinstance(label_spec, Mapping):
                    raise ValueError(f"alias: assay.labels.{ref} must resolve to a mapping")
                source = str(label_spec.get("source") or "").strip()
                if not source:
                    raise ValueError(f"alias: assay.labels.{ref}.source must be a non-empty string")
                values = label_spec.get("values", {}) or {}
                if not isinstance(values, Mapping):
                    raise ValueError(f"alias: assay.labels.{ref}.values must be a mapping")
                mappings[source] = {str(k): str(v) for k, v in values.items()}
                output = label_spec.get("output")
                output_names[source] = (
                    str(output) if isinstance(output, str) and output.strip() else f"{source}{cfg.suffix}"
                )

        for col, mapping in mappings.items():
            if col not in df.columns:
                raise ValueError(f"alias: column '{col}' not found in dataframe")

            # capture "before" as strings for robust, explicit comparisons
            before = df[col].astype(str)
            uniq_vals = int(before.nunique(dropna=False))

            # normalized keys for matching
            norm_map = self._build_norm_map(mapping, ci=cfg.case_insensitive)
            rules_total = len(norm_map)

            before_norm = before.str.casefold() if cfg.case_insensitive else before

            # vectorized mapping: map normalized values → alias; keep original when not mapped
            mapped = before_norm.map(norm_map)  # Series[str or NaN]
            after_series = mapped.fillna(before)  # keep original where no rule

            # write output column
            out_col = col if cfg.in_place else output_names.get(col, f"{col}{cfg.suffix}")
            if cfg.in_place:
                df[col] = after_series
            else:
                df[out_col] = after_series

            # rows where alias actually changed the value (avoid counting rules that map to same text)
            changed_mask = mapped.notna() & (mapped.astype(str) != before)
            changed_rows = int(changed_mask.sum())

            # rules actually used = normalized keys present at least once in the data
            used_rule_keys = set(before_norm[before_norm.isin(norm_map.keys())].unique())
            used_rules = len(used_rule_keys)

            # examples of raw → alias for first few changed pairs
            if changed_rows:
                sample = pd.DataFrame({"raw": before, "alias": after_series})[changed_mask].drop_duplicates("raw")
                examples = [f"{r!r} → {a!r}" for r, a in sample.head(6).itertuples(index=False)]
            else:
                examples = []

            # preview of rules that matched nothing (normalized keys)
            unused_rules = sorted(set(norm_map.keys()) - used_rule_keys)
            unused_rules_preview = [repr(k) for k in unused_rules[:6]]

            # concise, pretty summary
            self._log_summary(
                ctx,
                col=col,
                out_col=out_col,
                rules_total=rules_total,
                used_rules=used_rules,
                changed_rows=changed_rows,
                uniq_vals=uniq_vals,
                examples=examples,
                unused_rules_preview=unused_rules_preview,
            )

        return {"df": df}
