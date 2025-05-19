"""
--------------------------------------------------------------------------------
<reader project>
src/reader/processors/custom_params.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import logging
from typing import Tuple, List, Optional

import pandas as pd

from reader.config import CustomParameter

logger = logging.getLogger(__name__)

def apply_custom_parameters(
    df: pd.DataFrame,
    blank_correction: str = 'avg_blank',
    overflow_action: str = 'max',
    outlier_filter: bool = False,
    custom_parameters: Optional[List[CustomParameter]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    QC + derive custom parameters.

    Steps:
      1) Split blanks vs. data
      2) Handle 'OVRFLW' values per overflow_action
      3) Optionally subtract blanks (unless blank_correction == 'disregard')
      4) Optionally remove outliers
      5) Drop duplicate readings
      6) Append ratio parameters

    Returns
    -------
    data_corrected, blanks
    """
    # 1) Split blanks
    blanks = df[df['type'] == 'blank'].copy()
    data = df[df['type'] != 'blank'].copy()
    logger.info("Split into %d blanks and %d samples", len(blanks), len(data))

    # 2) Overflow handling
    overflow_mask = data['value'] == 'OVRFLW'
    if overflow_mask.any():
        events = data.loc[overflow_mask, ['position', 'time', 'channel']]
        logger.warning("Detected %d overflow readings:\n%s",
                       overflow_mask.sum(), events.to_string(index=False))
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    if overflow_action == 'zero':
        data['value'] = data['value'].fillna(0)
    elif overflow_action in ('max', 'min'):
        fill_vals = data.groupby('channel')['value'].transform(overflow_action)
        data['value'] = data['value'].fillna(fill_vals)
    elif overflow_action == 'drop':
        before = len(data)
        data = data.loc[~overflow_mask].copy()
        logger.info("Dropped %d overflow rows", before - len(data))
    else:
        raise ValueError(f"Unknown overflow_action '{overflow_action}'")
    logger.info("Overflow handled using '%s' strategy", overflow_action)

    # 3) Blank subtraction
    if blank_correction != 'disregard' and not blanks.empty:
        blanks['value'] = pd.to_numeric(blanks['value'], errors='coerce')
        # Choose subtraction metric
        if blank_correction == 'avg_blank':
            blanks_metric = blanks.groupby('channel')['value'].mean()
        else:  # median_blank or any other default
            blanks_metric = blanks.groupby('channel')['value'].median()
        logger.info("Computed blank metric (%s) per channel:\n%s",
                    blank_correction, blanks_metric.to_string())

        def subtract_and_clamp(row):
            med = blanks_metric.get(row['channel'], 0.0)
            corrected = row['value'] - med
            # For 'small_pos', enforce minimum positive floor
            if blank_correction == 'small_pos':
                return corrected if corrected >= 1e-3 else 1e-3
            # Otherwise clamp at zero
            return corrected if corrected >= 0 else 0

        data['value'] = data.apply(subtract_and_clamp, axis=1)
        logger.info("Applied '%s' blank subtraction", blank_correction)
    else:
        logger.info("Skipped blank subtraction (blank_correction='%s')",
                    blank_correction)

    # 4) Outlier removal
    if outlier_filter:
        def remove_outliers(grp):
            mu, sigma = grp['value'].mean(), grp['value'].std()
            mask = (grp['value'] >= mu - 3*sigma) & (grp['value'] <= mu + 3*sigma)
            removed = len(grp) - mask.sum()
            if removed:
                logger.info("Removed %d outliers in %s", removed, grp.name)
            return grp[mask]
        data = data.groupby('channel', group_keys=False).apply(remove_outliers)

    # 5) Deduplicate
    before = len(data)
    data = data.drop_duplicates(subset=['position', 'time', 'channel'])
    if len(data) < before:
        logger.info("Dropped %d duplicate readings", before - len(data))

    # 6) Custom parameter (ratio) derivation
    if custom_parameters:
        logger.info("Computing %d custom parameters", len(custom_parameters))
        derived = []
        # meta columns beyond position/time/channel/value
        meta_cols = [c for c in data.columns
                     if c not in ('position', 'time', 'channel', 'value')]

        for cp in custom_parameters:
            if cp.type.lower() != 'ratio' or len(cp.parameters) != 2:
                logger.warning("Skipping non‑ratio CP: %s", cp)
                continue
            name, (a, b) = cp.name, cp.parameters
            df_a = data[data['channel'] == a]
            df_b = data[data['channel'] == b]
            common = set(df_a['position']) & set(df_b['position'])
            for pos in common:
                da = df_a[df_a['position'] == pos].rename(columns={'value': a})
                db = df_b[df_b['position'] == pos].rename(columns={'value': b})
                merged = pd.merge(
                    da[['position', 'time'] + meta_cols + [a]],
                    db[['position', 'time', b]],
                    on=['position', 'time']
                )
                if merged.empty:
                    continue
                out = merged[['position', 'time'] + meta_cols].copy()
                out['channel'] = name
                out['value'] = merged[a] / (merged[b] + 1e-6)
                derived.append(out)
                logger.debug("Derived %s at %s: %d rows", name, pos, len(out))

        if derived:
            data = pd.concat([data] + derived, ignore_index=True)
            total = sum(len(d) for d in derived)
            logger.info("Appended %d ratio rows", total)

    return data, blanks
