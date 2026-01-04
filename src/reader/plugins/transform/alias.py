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


class AliasCfg(PluginConfig):
    """
    aliases:
      <column_name>:
        <raw_value>: <alias_value>
        ...
    aliases_ref:   reference key under data.aliases (used if aliases is omitted)
    in_place:      if true, mutate <column_name> directly; else create <column_name>_alias
    case_insensitive: map using casefold() on incoming values (keys in 'aliases' are matched case-insensitively)
    """

    aliases: Mapping[str, Mapping[str, str]] | None = None
    aliases_ref: str | None = None
    in_place: bool = False
    case_insensitive: bool = True
    suffix: str = "_alias"


class AliasTransform(Plugin):
    key = "alias"
    category = "transform"
    ConfigModel = AliasCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}  # works equally well on tidy+map

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

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
            unused_block = "\n".join(f"      • {s}" for s in unused_rules_preview) if unused_rules_preview else "      —"
            ctx.logger.debug(
                (
                    "alias details • %s → %s\n"
                    "   examples:\n%s\n"
                    "   unused_rule_keys:\n%s"
                ),
                col,
                out_col,
                ex_block,
                unused_block,
            )

    # ---------------- main ----------------

    def run(self, ctx, inputs, cfg: AliasCfg):
        df: pd.DataFrame = inputs["df"].copy()

        aliases = cfg.aliases
        if aliases is None:
            if not cfg.aliases_ref:
                raise ValueError("alias: provide with.aliases or with.aliases_ref")
            if not ctx.aliases or cfg.aliases_ref not in ctx.aliases:
                raise ValueError(f"alias: data.aliases missing key '{cfg.aliases_ref}'")
            aliases = ctx.aliases[cfg.aliases_ref]
        if not isinstance(aliases, Mapping):
            raise ValueError("alias: aliases must be a mapping of column -> {raw: alias}")
        if aliases:
            if all(not isinstance(v, Mapping) for v in aliases.values()):
                if not cfg.aliases_ref:
                    raise ValueError(
                        "alias: aliases_ref is required when aliases is a raw->alias mapping (single column)"
                    )
                aliases = {cfg.aliases_ref: aliases}
        elif cfg.aliases_ref:
            # allow empty alias maps to create <col>_alias columns without changes
            aliases = {cfg.aliases_ref: {}}

        for col, mapping in aliases.items():
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
            out_col = col if cfg.in_place else f"{col}{cfg.suffix}"
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
