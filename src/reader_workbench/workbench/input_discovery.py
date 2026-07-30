"""Neutral defaults shared by protocol authoring and ingest discovery mechanics."""

DEFAULT_INPUT_ROOTS = ("./inputs", "./raw", "./raw_data")
DEFAULT_WORKBOOK_INCLUDE = ("*.xlsx", "*.XLSX")
DEFAULT_INPUT_EXCLUDE = (
    "~$*",
    "._*",
    "#*#",
    "*.tmp",
    "*.temp",
    "*.bak",
    "metadata.*",
    "metadata_filtered.*",
    "sample_map.*",
    "sample_metadata.*",
    "plate_map.*",
)

__all__ = ["DEFAULT_INPUT_EXCLUDE", "DEFAULT_INPUT_ROOTS", "DEFAULT_WORKBOOK_INCLUDE"]
