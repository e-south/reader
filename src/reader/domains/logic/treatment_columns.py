from __future__ import annotations

import pandas as pd


def normalize_treatment_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold()


def choose_treatment_column(
    df: pd.DataFrame,
    treatment_map: dict[str, str],
    *,
    case_sensitive: bool,
    preferred: str | None = None,
) -> str:
    if preferred is not None:
        preferred = str(preferred).strip()
        if not preferred:
            raise ValueError("preferred treatment column must be a non-empty string when provided")
        if preferred not in df.columns:
            available = ", ".join(sorted(df.columns)) or "—"
            raise ValueError(
                f"Configured treatment column {preferred!r} is missing from the dataframe. Available columns: {available}"
            )
        return preferred

    candidates = [column for column in ("treatment", "treatment_alias") if column in df.columns]
    if not candidates:
        raise ValueError("Neither 'treatment' nor 'treatment_alias' is present in the dataframe.")

    def _score(column: str) -> int:
        values = df[column].astype(str)
        if case_sensitive:
            wanted = {str(item) for item in treatment_map.values()}
        else:
            wanted = {str(item).strip().casefold() for item in treatment_map.values()}
            values = normalize_treatment_series(values)
        return int(values.isin(list(wanted)).sum())

    scores = {column: _score(column) for column in candidates}
    return max(scores, key=lambda column: (scores[column], column == "treatment"))
