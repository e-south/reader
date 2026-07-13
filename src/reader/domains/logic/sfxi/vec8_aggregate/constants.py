from __future__ import annotations

SFXI_VEC8_RECORD_ID = "sfxi_vec8/vec8"
VEC8_CHANNELS = ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")
REQUIRED_VEC8_COLUMNS = (
    "design_id",
    "time_selected_h",
    "reference_design_id",
    "intensity_log2_offset_delta",
    "r_logic",
    *VEC8_CHANNELS,
    "flat_logic",
)

DIRECT_TABLE_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls"}
METADATA_COLUMNS = (
    "source_index",
    "source_id",
    "source_path",
    "table_path",
    "source_kind",
    "source_row_index",
    "row_label",
    "design_id",
    "sequence",
    "id",
    "time_selected_h",
    "reference_design_id",
    "intensity_log2_offset_delta",
    "r_logic",
    "flat_logic",
)
