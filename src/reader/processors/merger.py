"""
--------------------------------------------------------------------------------
<reader project>
src/reader/processors/merger.py

Combine raw data + plate-map → merged tidy DataFrame

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def merge_raw_and_map(raw: pd.DataFrame, plate_map: pd.DataFrame) -> pd.DataFrame:
    """
    Generic inner-join on 'position', with the following enhancements:

      1) Drop any plate_map columns that are entirely empty (and log them).
      2) Identify and drop any positions that carry *no* metadata beyond 'position'
         (i.e. all other metadata columns are NaN), since those wells were unused.
         Log which positions are dropped, and remove their raw measurements.
      3) Ensure that every remaining raw position exists in the cleaned plate_map.
         Raise if any are missing.
      4) Perform the merge and log the result.

    Returns
    -------
    pd.DataFrame
      The merged DataFrame.
    """
    # 1) sanity checks
    if 'position' not in raw.columns:
        raise ValueError("Raw DataFrame must have 'position' column")
    if 'position' not in plate_map.columns:
        raise ValueError("Plate-map DataFrame must have 'position' column")

    # 2) drop empty columns
    before_cols = set(plate_map.columns)
    plate_map_clean = plate_map.dropna(axis=1, how='all')
    dropped_cols = before_cols - set(plate_map_clean.columns)
    if dropped_cols:
        logger.debug(f"Dropped empty plate_map columns: {sorted(dropped_cols)}")

    # 3) drop positions with no metadata
    meta_cols = [c for c in plate_map_clean.columns if c != 'position']
    if meta_cols:
        no_meta_mask = plate_map_clean[meta_cols].isna().all(axis=1)
        drop_positions = plate_map_clean.loc[no_meta_mask, 'position'].tolist()
        if drop_positions:
            logger.info(f"Dropping positions with no metadata: {drop_positions}")
            # remove from plate_map
            plate_map_clean = plate_map_clean.loc[~no_meta_mask].copy()
            # also remove their raw measurements
            before_raw = len(raw)
            raw = raw.loc[~raw['position'].isin(drop_positions)].copy()
            after_raw = len(raw)
            logger.debug(f"Dropped {before_raw - after_raw} raw rows for unused positions")
    else:
        # if for some reason there are no meta columns at all, drop everything
        logger.warning(
            "Plate map contains only 'position' and no metadata columns; "
            "all positions will be dropped, no merge performed."
        )
        return pd.DataFrame(columns=raw.columns)

    # 4) ensure no remaining raw positions are missing from the map
    raw_positions = set(raw['position'].unique())
    map_positions = set(plate_map_clean['position'].unique())
    missing = raw_positions - map_positions
    if missing:
        raise ValueError(f"Plate-map missing entries for positions: {sorted(missing)}")

    # 5) perform the merge
    merged = pd.merge(
        raw,
        plate_map_clean,
        on='position',
        how='left',
        validate='many_to_one',
    )
    logger.info(
        f"Merged raw ({len(raw)} rows) with plate_map "
        f"({len(plate_map_clean)} rows) → {len(merged)} total rows"
    )
    return merged
