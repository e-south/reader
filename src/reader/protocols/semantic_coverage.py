from __future__ import annotations

from typing import Any

from reader.protocols.model import ProtocolSemanticExecution, ProtocolSemanticProgram


def _semantic_program(
    protocol: Any,
    *,
    overrides: dict[str, ProtocolSemanticExecution],
    active_profile: str | None = None,
) -> ProtocolSemanticProgram:
    return protocol.semantic_program(active_profile=active_profile, execution_overrides=overrides)


def _plate_reader_semantic_program(
    protocol: Any,
    *,
    include_crosstalk_pairs: bool,
    include_fold_change: bool,
) -> ProtocolSemanticProgram:
    active_profile = _dual_reporter_semantic_profile(
        include_fold_change=include_fold_change,
        include_crosstalk_pairs=include_crosstalk_pairs,
    )
    overrides: dict[str, ProtocolSemanticExecution] = {
        "OD": ProtocolSemanticExecution(
            status="compiled",
            step_ids=("ingest",),
            plugin_ids=("ingest/synergy_h1",),
            record_ids=("ingest/df",),
            note="Raw OD600 values are materialized on the ingest dataframe.",
        ),
    }
    overrides.update(
        {
            "CFP": ProtocolSemanticExecution(
                status="compiled",
                step_ids=("ingest",),
                plugin_ids=("ingest/synergy_h1",),
                record_ids=("ingest/df",),
                note="Raw CFP values are materialized on the ingest dataframe.",
            ),
            "YFP": ProtocolSemanticExecution(
                status="compiled",
                step_ids=("ingest",),
                plugin_ids=("ingest/synergy_h1",),
                record_ids=("ingest/df",),
                note="Raw YFP values are materialized on the ingest dataframe.",
            ),
            "CFP_OD": ProtocolSemanticExecution(
                status="compiled",
                step_ids=("ratio_cfp_od600",),
                plugin_ids=("transform/ratio",),
                record_ids=("ratio_cfp_od600/df",),
                note="The CFP/OD600 support channel is materialized as a ratio step output.",
            ),
            "YFP_OD": ProtocolSemanticExecution(
                status="compiled",
                step_ids=("ratio_yfp_od600",),
                plugin_ids=("transform/ratio",),
                record_ids=("ratio_yfp_od600/df",),
                note="The YFP/OD600 support channel is materialized as a ratio step output.",
            ),
            "Ratio": ProtocolSemanticExecution(
                status="compiled",
                step_ids=("ratio_yfp_cfp",),
                plugin_ids=("transform/ratio",),
                record_ids=("ratio_yfp_cfp/df",),
                note="The primary YFP/CFP ratio is materialized as a ratio step output.",
            ),
        }
    )
    if include_fold_change:
        fold_change_step_id = "fold_change__yfp_over_cfp"
        fold_change_record_id = "fold_change__yfp_over_cfp/table"
        fold_change_note = "Nearest-time fold-change summaries are materialized from the primary ratio channel."
        overrides.update(
            {
                "FC": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=(fold_change_step_id,),
                    plugin_ids=("transform/fold_change",),
                    record_ids=(fold_change_record_id,),
                    note=fold_change_note,
                ),
                "log2FC": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=(fold_change_step_id,),
                    plugin_ids=("transform/fold_change",),
                    record_ids=(fold_change_record_id,),
                    note=fold_change_note,
                ),
            }
        )
    if include_crosstalk_pairs:
        overrides["ranking"] = ProtocolSemanticExecution(
            status="compiled",
            step_ids=("crosstalk_pairs",),
            plugin_ids=("transform/crosstalk_pairs",),
            record_ids=("crosstalk_pairs/table",),
            config_paths=("protocol.analysis.crosstalk_pairs",),
            note="When crosstalk pair analysis is enabled, pair selection is compiled from fold-change output.",
        )
    return _semantic_program(protocol, overrides=overrides, active_profile=active_profile)


def _plate_reader_single_reporter_semantic_program(
    protocol: Any,
    *,
    reporter_channel: str,
    normalizer_channel: str,
    include_fold_change: bool,
) -> ProtocolSemanticProgram:
    ratio_label = _single_reporter_ratio_label(
        reporter_channel=reporter_channel,
        normalizer_channel=normalizer_channel,
    )
    ratio_note = f"The primary {ratio_label} ratio is materialized as a ratio step output."
    fold_change_note = f"Nearest-time fold-change summaries are materialized from the primary {ratio_label} channel."
    overrides: dict[str, ProtocolSemanticExecution] = {
        "Normalizer": ProtocolSemanticExecution(
            status="compiled",
            step_ids=("ingest",),
            plugin_ids=("ingest/synergy_h1",),
            record_ids=("ingest/df",),
            note=f"Raw {normalizer_channel} values are materialized on the ingest dataframe.",
        ),
        "Reporter": ProtocolSemanticExecution(
            status="compiled",
            step_ids=("ingest",),
            plugin_ids=("ingest/synergy_h1",),
            record_ids=("ingest/df",),
            note=f"Raw {reporter_channel} values are materialized on the ingest dataframe.",
        ),
        "Reporter_Normalizer": ProtocolSemanticExecution(
            status="compiled",
            step_ids=("ratio_reporter_normalizer",),
            plugin_ids=("transform/ratio",),
            record_ids=("ratio_reporter_normalizer/df",),
            note=ratio_note,
        ),
    }
    if include_fold_change:
        overrides.update(
            {
                "FC": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("fold_change__single_reporter",),
                    plugin_ids=("transform/fold_change",),
                    record_ids=("fold_change__single_reporter/table",),
                    note=fold_change_note,
                ),
                "log2FC": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("fold_change__single_reporter",),
                    plugin_ids=("transform/fold_change",),
                    record_ids=("fold_change__single_reporter/table",),
                    note=fold_change_note,
                ),
            }
        )
    return _semantic_program(
        protocol, overrides=overrides, active_profile=_single_reporter_semantic_profile(include_fold_change)
    )


def _plate_reader_retron_sponge_semantic_program(
    protocol: Any,
    *,
    measurement: str,
    reporter_channel: str,
    growth_channel: str,
) -> ProtocolSemanticProgram:
    trace_binding = ProtocolSemanticExecution(
        status="compiled",
        step_ids=("semantic_metrics",),
        plugin_ids=("transform/retron_sponge_metrics",),
        record_ids=("semantic_metrics/trace",),
        config_paths=("protocol.analysis.semantic_metrics",),
        note="Matched-control sponge kinetics are materialized as a typed trace table.",
    )
    summary_binding = ProtocolSemanticExecution(
        status="compiled",
        step_ids=("semantic_metrics",),
        plugin_ids=("transform/retron_sponge_metrics",),
        record_ids=("semantic_metrics/summary",),
        config_paths=("protocol.analysis.semantic_metrics",),
        note="Matched-control sponge summaries are materialized as a typed summary table.",
    )
    overrides: dict[str, ProtocolSemanticExecution] = {
        "matched_same_sensor_control": trace_binding,
        "pre_stress_last_n": trace_binding,
        "primary_post_stress": trace_binding,
        "endpoint_last_n": trace_binding,
        "OD": ProtocolSemanticExecution(
            status="compiled",
            step_ids=("ingest",),
            plugin_ids=("ingest/synergy_h1",),
            record_ids=("ingest/df",),
            note=f"Raw {growth_channel} values are materialized on the ingest dataframe.",
        ),
        "R": trace_binding,
        "R_pre": summary_binding,
        "P_pre": summary_binding,
        "B": trace_binding,
        "C": trace_binding,
        "C_AUC": summary_binding,
        "C_END": summary_binding,
        "mu": trace_binding,
        "D": trace_binding,
        "D_AUC": summary_binding,
        "D_END": summary_binding,
        "D_abs": trace_binding,
        "D_abs_AUC": summary_binding,
        "D_abs_END": summary_binding,
        "D_growth": trace_binding,
        "D_growth_AUC": summary_binding,
        "D_growth_END": summary_binding,
        "M": trace_binding,
        "M_AUC": summary_binding,
        "M_END": summary_binding,
        "O": trace_binding,
        "O_AUC": summary_binding,
        "O_abs": trace_binding,
        "O_abs_AUC": summary_binding,
        "G_sensor": summary_binding,
        "S_AUC": summary_binding,
        "S_abs_AUC": summary_binding,
        "L_pre": summary_binding,
        "L_post_AUC": summary_binding,
        "T_ratio_AUC": summary_binding,
        "T_growth_AUC": summary_binding,
        "T_finalOD": summary_binding,
        "ranking": summary_binding,
    }
    if measurement == "yfp_cfp":
        overrides.update(
            {
                "CFP": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ingest",),
                    plugin_ids=("ingest/synergy_h1",),
                    record_ids=("ingest/df",),
                    note="Raw CFP values are materialized on the ingest dataframe.",
                ),
                "YFP": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ingest",),
                    plugin_ids=("ingest/synergy_h1",),
                    record_ids=("ingest/df",),
                    note="Raw YFP values are materialized on the ingest dataframe.",
                ),
                "CFP_OD": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ratio_cfp_od600",),
                    plugin_ids=("transform/ratio",),
                    record_ids=("ratio_cfp_od600/df",),
                    note="The CFP/OD600 support channel is materialized as a ratio step output.",
                ),
                "YFP_OD": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ratio_yfp_od600",),
                    plugin_ids=("transform/ratio",),
                    record_ids=("ratio_yfp_od600/df",),
                    note="The YFP/OD600 support channel is materialized as a ratio step output.",
                ),
            }
        )
    else:
        overrides.update(
            {
                "Reporter": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ingest",),
                    plugin_ids=("ingest/synergy_h1",),
                    record_ids=("ingest/df",),
                    note=f"Raw {reporter_channel} values are materialized on the ingest dataframe.",
                ),
                "Reporter_OD": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ratio_reporter_normalizer",),
                    plugin_ids=("transform/ratio",),
                    record_ids=("ratio_reporter_normalizer/df",),
                    note=(
                        "The "
                        f"{reporter_channel}/{growth_channel} support channel is materialized as a ratio step output."
                    ),
                ),
            }
        )
    return _semantic_program(protocol, overrides=overrides, active_profile=measurement)


def _logic_semantic_program(protocol: Any, *, include_vec8: bool) -> ProtocolSemanticProgram:
    overrides: dict[str, ProtocolSemanticExecution] = {}
    if include_vec8:
        vec8_binding = ProtocolSemanticExecution(
            status="compiled",
            step_ids=("sfxi_vec8",),
            plugin_ids=("transform/sfxi",),
            record_ids=("sfxi_vec8/vec8",),
            config_paths=(
                "protocol.inputs.response",
                "protocol.inputs.reference",
                "protocol.inputs.design_by",
                "protocol.inputs.logic_map_ref",
                "protocol.inputs.time_mode",
                "protocol.inputs.target_time_h",
                "protocol.inputs.time_tolerance_h",
            ),
            note="The SFXI vec8 transform materializes the protocol control rule, summary window, and vector metric.",
        )
        overrides.update(
            {
                "logic_corner_map": vec8_binding,
                "summary_timepoint": vec8_binding,
                "vec8": vec8_binding,
            }
        )
    return _semantic_program(protocol, overrides=overrides)


def _cytometry_semantic_program(protocol: Any) -> ProtocolSemanticProgram:
    return _semantic_program(
        protocol,
        overrides={
            "ranking": ProtocolSemanticExecution(
                status="descriptive_only",
                note="Cytometry ranking remains domain-defined until a typed analysis program is introduced.",
            )
        },
    )


def _dual_reporter_semantic_profile(*, include_fold_change: bool, include_crosstalk_pairs: bool) -> str:
    if include_crosstalk_pairs:
        return "yfp_cfp_crosstalk"
    if include_fold_change:
        return "yfp_cfp_fold_change"
    return "yfp_cfp_raw"


def _single_reporter_semantic_profile(include_fold_change: bool) -> str:
    return "single_reporter_fold_change" if include_fold_change else "single_reporter_raw"


def _single_reporter_ratio_label(*, reporter_channel: str, normalizer_channel: str) -> str:
    return f"{reporter_channel}/{normalizer_channel}"
