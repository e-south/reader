from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def build_norm_map(mapping: Mapping[str, str], *, case_insensitive: bool) -> Mapping[str, str]:
    if case_insensitive:
        return {str(key).casefold(): str(value) for key, value in mapping.items()}
    return {str(key): str(value) for key, value in mapping.items()}


def log_label_summary(
    ctx,
    *,
    column: str,
    output_column: str,
    rules_total: int,
    used_rules: int,
    changed_rows: int,
    unique_values: int,
    examples: list[str],
    unused_rules_preview: list[str],
    label: str,
) -> None:
    unused_count = len(unused_rules_preview)
    ctx.logger.info(
        "%s • [accent]%s[/accent] → %s • rules=%d used=%d changed=%d uniques=%d%s",
        label,
        column,
        output_column,
        rules_total,
        used_rules,
        changed_rows,
        unique_values,
        f" • unused={unused_count}" if unused_count else "",
    )
    if examples or unused_rules_preview:
        examples_block = "\n".join(f"      • {entry}" for entry in examples) if examples else "      —"
        unused_block = (
            "\n".join(f"      • {entry}" for entry in unused_rules_preview) if unused_rules_preview else "      —"
        )
        ctx.logger.debug(
            ("%s details • %s → %s\n   examples:\n%s\n   unused_rule_keys:\n%s"),
            label,
            column,
            output_column,
            examples_block,
            unused_block,
        )


def apply_label_mappings(
    *,
    ctx,
    df: pd.DataFrame,
    mappings: Mapping[str, Mapping[str, str]],
    output_names: Mapping[str, str],
    in_place: bool,
    case_insensitive: bool,
    label: str,
) -> pd.DataFrame:
    out = df.copy()
    for column, mapping in mappings.items():
        if column not in out.columns:
            raise ValueError(f"{label}: column '{column}' not found in dataframe")

        before = out[column].astype(str)
        unique_values = int(before.nunique(dropna=False))
        normalized_before = before.str.casefold() if case_insensitive else before
        normalized_map = build_norm_map(mapping, case_insensitive=case_insensitive)
        rules_total = len(normalized_map)

        mapped = normalized_before.map(normalized_map)
        after_series = mapped.fillna(before)

        output_column = column if in_place else output_names[column]
        if in_place:
            out[column] = after_series
        else:
            out[output_column] = after_series

        changed_mask = mapped.notna() & (mapped.astype(str) != before)
        changed_rows = int(changed_mask.sum())
        used_rule_keys = set(normalized_before[normalized_before.isin(normalized_map.keys())].unique())
        used_rules = len(used_rule_keys)

        if changed_rows:
            sample = pd.DataFrame({"raw": before, "alias": after_series})[changed_mask].drop_duplicates("raw")
            examples = [f"{raw!r} → {alias!r}" for raw, alias in sample.head(6).itertuples(index=False)]
        else:
            examples = []
        unused_rules = sorted(set(normalized_map.keys()) - used_rule_keys)
        unused_rules_preview = [repr(key) for key in unused_rules[:6]]

        log_label_summary(
            ctx,
            column=column,
            output_column=output_column,
            rules_total=rules_total,
            used_rules=used_rules,
            changed_rows=changed_rows,
            unique_values=unique_values,
            examples=examples,
            unused_rules_preview=unused_rules_preview,
            label=label,
        )
    return out
