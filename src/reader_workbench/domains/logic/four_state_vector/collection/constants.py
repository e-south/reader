from __future__ import annotations

VECTOR_CHANNELS = ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")
REQUIRED_VECTOR_COLUMNS = (
    "design_id",
    "reference_design_id",
    "intensity_log2_offset_delta",
    "r_logic",
    *VECTOR_CHANNELS,
    "flat_logic",
)

METADATA_COLUMNS = (
    "source_index",
    "source_resource_id",
    "source_experiment_id",
    "source_record_id",
    "source_record_revision_digest",
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
