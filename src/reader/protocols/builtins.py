from __future__ import annotations

from functools import cache

from reader.plugins.ingest.discovery_policy import DEFAULT_EXCLUDE

from .compiler import (
    compile_cytometry_flow_panel,
    compile_generic_protocol,
    compile_logic_sfxi_screen,
    compile_plate_reader_dual_reporter_screen,
)
from .model import (
    ProtocolCatalog,
    ProtocolControlRule,
    ProtocolDeliverableSpec,
    ProtocolDescriptor,
    ProtocolEffectSignSpec,
    ProtocolExecutionPlan,
    ProtocolFactorSpec,
    ProtocolFigureSpec,
    ProtocolMetricSpec,
    ProtocolNotebookPolicy,
    ProtocolPluginDefaultsSpec,
    ProtocolRankingSpec,
    ProtocolWindowSpec,
    binding_value,
)

BUILTIN_PROTOCOLS: tuple[ProtocolDescriptor, ...] = (
    ProtocolDescriptor(
        protocol="workbench/generic",
        domain="generic",
        family="general_workbench",
        summary="Generic explicit protocol binding for experiments that do not yet fit a domain-specific protocol.",
        tags=("generic", "explicit_binding"),
        factors=(
            ProtocolFactorSpec(name="sample", role="sample", summary="Primary experimental unit."),
            ProtocolFactorSpec(name="replicate", role="replicate", summary="Replicate grouping axis.", required=False),
            ProtocolFactorSpec(name="time", role="time", summary="Observation axis when present.", required=False),
        ),
        figures=(
            ProtocolFigureSpec(
                id="generic_qc",
                kind="qc",
                summary="Use domain-specific QC views before interpreting downstream summaries.",
                primary=True,
            ),
        ),
        ranking=ProtocolRankingSpec(
            primary_metric="domain_defined",
            direction="higher_is_better",
            summary="Generic protocol leaves ranking to the domain-specific analysis layer.",
        ),
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/basic",
                allowed_templates=(
                    "notebook/basic",
                    "notebook/eda",
                    "notebook/microplate",
                    "notebook/cytometry",
                    "notebook/sfxi_eda",
                ),
                summary="Generic workbench protocol defaults to the minimal record explorer and allows explicit overrides.",
            ),
            compiler=compile_generic_protocol,
        ),
    ),
    ProtocolDescriptor(
        protocol="plate_reader/dual_reporter_screen",
        domain="plate_reader",
        family="screen_analysis",
        summary=(
            "Dual-reporter screen protocol for inducer/stress assays with matched same-sensor controls, "
            "baseline-shifted ratio traces, and burden/leakiness-aware ranking."
        ),
        tags=("plate_reader", "dual_reporter", "screen", "ratio"),
        factors=(
            ProtocolFactorSpec(name="sensor", role="sensor", summary="Reporter promoter/sensor arm."),
            ProtocolFactorSpec(name="sponge", role="construct", summary="Real or control sponge design."),
            ProtocolFactorSpec(name="stress_condition", role="stress", summary="Stress or H2O condition."),
            ProtocolFactorSpec(name="IPTG", role="induction", summary="Induction state for sponge expression."),
            ProtocolFactorSpec(name="replicate_id", role="replicate", summary="Replicate well identifier."),
            ProtocolFactorSpec(name="time", role="time", summary="Time after assay start."),
            ProtocolFactorSpec(name="plate_id", role="plate", summary="Plate-local normalization boundary."),
        ),
        control_rules=(
            ProtocolControlRule(
                id="matched_same_sensor_control",
                summary=(
                    "Normalize every real sponge well to the same-sensor control on the same plate, "
                    "matched by stress state, induction state, and timepoint."
                ),
                match_on=("sensor", "plate_id", "stress_condition", "IPTG", "time"),
                control_selector="matched_tetO_group",
            ),
        ),
        windows=(
            ProtocolWindowSpec(
                id="pre_stress_last_n",
                summary="Use the last N reads before stress addition as the baseline window.",
                anchor="stress_time_zero",
                selector="last_n_before",
                params={"n": 3},
            ),
            ProtocolWindowSpec(
                id="primary_post_stress",
                summary="Use the post-stress kinetic window from first read after stress until the matched control plateaus.",
                anchor="stress_time_zero",
                selector="first_after_until_plateau",
                params={"plateau_reference": "matched_same_sensor_control"},
            ),
            ProtocolWindowSpec(
                id="endpoint_last_n",
                summary="Use the last N reads inside the primary post-stress window as the endpoint window.",
                anchor="primary_post_stress",
                selector="last_n_within",
                params={"n": 3},
            ),
        ),
        metrics=(
            ProtocolMetricSpec(id="OD", stage="raw", summary="Raw OD600 trace.", formula="OD600"),
            ProtocolMetricSpec(id="CFP", stage="raw", summary="Raw CFP trace.", formula="CFP"),
            ProtocolMetricSpec(id="YFP", stage="raw", summary="Raw YFP trace.", formula="YFP"),
            ProtocolMetricSpec(
                id="YFP_OD",
                stage="support",
                summary="Supporting YFP per biomass proxy.",
                formula="YFP / OD600",
                depends_on=("YFP", "OD"),
            ),
            ProtocolMetricSpec(
                id="CFP_OD",
                stage="support",
                summary="Supporting CFP per biomass proxy.",
                formula="CFP / OD600",
                depends_on=("CFP", "OD"),
            ),
            ProtocolMetricSpec(
                id="R",
                stage="derived",
                summary="Primary within-well dual-reporter ratio.",
                formula="log2(YFP / CFP)",
                depends_on=("YFP", "CFP"),
            ),
            ProtocolMetricSpec(
                id="B",
                stage="derived",
                summary="Baseline-shifted reporter ratio relative to the well's own pre-stress state.",
                formula="R(t) - mean(R over pre_stress_last_n)",
                depends_on=("R", "pre_stress_last_n"),
            ),
            ProtocolMetricSpec(
                id="C",
                stage="comparison",
                summary="Matched-control-normalized sponge deviation.",
                formula="B(t) - mean(B matched_same_sensor_control at t)",
                depends_on=("B", "matched_same_sensor_control"),
            ),
            ProtocolMetricSpec(
                id="D",
                stage="comparison",
                summary="Induced sponge effect after matched-control normalization.",
                formula="mean(C +IPTG) - mean(C -IPTG)",
                depends_on=("C",),
            ),
            ProtocolMetricSpec(
                id="M",
                stage="comparison",
                summary="Stress modulation of the induced sponge effect.",
                formula="D(relevant_stress) - D(H2O)",
                depends_on=("D",),
            ),
            ProtocolMetricSpec(
                id="O",
                stage="ranking",
                summary="Sign-corrected induced sponge effect for cross-sensor ranking.",
                formula="expected_decoy_sign * D",
                depends_on=("D",),
            ),
            ProtocolMetricSpec(
                id="S_AUC",
                stage="ranking",
                summary="Cross-sensor scaled effect size relative to native sensor response.",
                formula="O_AUC / abs(G_sensor)",
                depends_on=("O", "G_sensor"),
            ),
            ProtocolMetricSpec(
                id="L_pre",
                stage="leakiness",
                summary="Pre-stress leakiness relative to matched control.",
                formula="R_pre(real,-IPTG) - mean(R_pre matched_control,-IPTG)",
                depends_on=("R",),
            ),
            ProtocolMetricSpec(
                id="L_post_AUC",
                stage="leakiness",
                summary="Uninduced post-stress leakiness over the primary window.",
                formula="AUC(mean(C -IPTG))",
                depends_on=("C", "primary_post_stress"),
            ),
            ProtocolMetricSpec(
                id="T_ratio_AUC",
                stage="burden",
                summary="tetO ratio burden under induction.",
                formula="AUC(mean(B tetO,+IPTG) - mean(B tetO,-IPTG))",
                depends_on=("B", "primary_post_stress"),
            ),
            ProtocolMetricSpec(
                id="T_growth_AUC",
                stage="burden",
                summary="tetO growth burden under induction.",
                formula="AUC(mean(mu tetO,+IPTG) - mean(mu tetO,-IPTG))",
                depends_on=("primary_post_stress",),
            ),
            ProtocolMetricSpec(
                id="T_finalOD",
                stage="burden",
                summary="Endpoint OD burden for the tetO control.",
                formula="mean(OD tetO,+IPTG,end) - mean(OD tetO,-IPTG,end)",
                depends_on=("OD", "endpoint_last_n"),
            ),
        ),
        effect_signs=(
            ProtocolEffectSignSpec(
                target="spyP",
                expected_sign="negative",
                summary="Effective decoys reduce the spyP ratio after sign correction.",
            ),
            ProtocolEffectSignSpec(
                target="sulAp",
                expected_sign="positive",
                summary="Effective LexA decoys increase the sulAp ratio.",
            ),
            ProtocolEffectSignSpec(
                target="soxSp",
                expected_sign="negative",
                summary="Effective SoxR/SoxS decoys reduce the soxSp ratio after sign correction.",
            ),
        ),
        figures=(
            ProtocolFigureSpec(
                id="tetO_burden", kind="qc", summary="tetO burden panel isolates induction burden.", primary=True
            ),
            ProtocolFigureSpec(
                id="tetO_normalized_kinetics",
                kind="kinetics",
                summary="Matched-control-normalized C(t) traces show sponge-specific deviations.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="induced_effect",
                kind="kinetics",
                summary="D(t) traces isolate induced sequence-specific sponge effects.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="library_heatmap",
                kind="summary",
                summary="Library-wide O_AUC/S_AUC heatmaps compare effects across sensors.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="pareto_ranking",
                kind="ranking",
                summary="Pareto ranking balances on-target effect, burden, and leakiness.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="architecture_analysis",
                kind="architecture",
                summary="Architecture plots compare mono-, bi-, tri-, and quad-functional sponge behavior.",
            ),
        ),
        deliverables=(
            ProtocolDeliverableSpec(
                id="time_series",
                surface="plots",
                summary="Primary time-series kinetics panel.",
                default=True,
            ),
            ProtocolDeliverableSpec(
                id="snapshot_by_channel",
                surface="plots",
                summary="Snapshot barplots grouped by treatment/channel.",
                default=True,
            ),
            ProtocolDeliverableSpec(
                id="snapshot_by_design",
                surface="plots",
                summary="Snapshot barplots grouped by design.",
                default=True,
            ),
            ProtocolDeliverableSpec(
                id="snapshot_state",
                surface="plots",
                summary="2x2 state summary panel for alias-mapped treatment states.",
            ),
            ProtocolDeliverableSpec(
                id="ts_and_snap_intensity",
                surface="plots",
                summary="Combined time-series and endpoint view of the intensity channel.",
                default=True,
            ),
            ProtocolDeliverableSpec(
                id="ts_and_snap_ratio",
                surface="plots",
                summary="Combined time-series and endpoint view of the dual-reporter ratio.",
            ),
            ProtocolDeliverableSpec(
                id="distributions",
                surface="plots",
                summary="Distribution view of the primary ratio channel.",
            ),
            ProtocolDeliverableSpec(
                id="snapshot_heatmap_yfp_cfp",
                surface="plots",
                summary="Snapshot heatmap for YFP/CFP.",
            ),
            ProtocolDeliverableSpec(
                id="snapshot_heatmap_cfp_od600",
                surface="plots",
                summary="Snapshot heatmap for CFP/OD600.",
            ),
            ProtocolDeliverableSpec(
                id="logic_symmetry_yfp_cfp",
                surface="plots",
                summary="Logic symmetry view over the YFP/CFP channel.",
            ),
            ProtocolDeliverableSpec(
                id="crosstalk_pairs_csv",
                surface="exports",
                summary="CSV export of crosstalk-safe pair candidates.",
            ),
        ),
        ranking=ProtocolRankingSpec(
            primary_metric="O_AUC",
            direction="higher_is_better",
            penalties=("T_growth_AUC", "T_finalOD", "L_pre", "L_post_AUC"),
            supporting_metrics=("S_AUC", "D_END", "M_AUC"),
            summary="Rank hits by sign-corrected effect size, then penalize burden and leakiness.",
        ),
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/eda",
                allowed_templates=("notebook/eda", "notebook/microplate", "notebook/basic"),
                summary="Dual-reporter plate-reader screens default to the EDA notebook with plot support.",
            ),
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/synergy_h1",
                    summary="Dual-reporter screens default to CFP/YFP/OD600 ingest in auto mode.",
                    with_={
                        "mode": binding_value("ingest.mode", "auto"),
                        "channels": binding_value("ingest.channels", ["OD600", "CFP", "YFP"]),
                        "sheet_names": binding_value("ingest.sheet_names", None),
                        "add_sheet": binding_value("ingest.add_sheet", False),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/fold_change",
                    summary="Shared fold-change defaults keep plate-reader comparisons and naming consistent.",
                    with_={
                        "target": binding_value("fold_change.target", "YFP/CFP"),
                        "report_times": binding_value("fold_change.report_times"),
                        "time_tolerance": binding_value("fold_change.time_tolerance", 0.51),
                        "agg": binding_value("fold_change.agg", "median"),
                        "treatment_column": binding_value("fold_change.treatment_column", "treatment"),
                        "group_by": binding_value("fold_change.group_by", ["design_id"]),
                        "use_global_baseline": binding_value("fold_change.use_global_baseline", False),
                        "global_baseline_value": binding_value("fold_change.global_baseline_value", None),
                        "overrides": binding_value("fold_change.overrides", []),
                        "fc_column": binding_value("fold_change.fc_column", "FC"),
                        "log2fc_column": binding_value("fold_change.log2fc_column", "log2FC"),
                    },
                ),
            ),
            compiler=compile_plate_reader_dual_reporter_screen,
        ),
    ),
    ProtocolDescriptor(
        protocol="logic/sfxi_screen",
        domain="logic",
        family="logic_summary",
        summary="SFXI logic-screen protocol for mapping inducible corners to vec8 logic/intensity summaries.",
        tags=("logic", "sfxi", "screen"),
        factors=(
            ProtocolFactorSpec(name="design_id", role="design", summary="Design grouping for logic comparison."),
            ProtocolFactorSpec(name="time", role="time", summary="Selected summary timepoint."),
            ProtocolFactorSpec(name="logic_map", role="mapping", summary="00/10/01/11 treatment-to-corner map."),
            ProtocolFactorSpec(
                name="reference_design", role="control", summary="Reference design for intensity normalization."
            ),
        ),
        control_rules=(
            ProtocolControlRule(
                id="logic_corner_map",
                summary="Resolve treatment states to 00/10/01/11 corners through the configured logic map.",
                control_selector="logic_map_ref",
            ),
        ),
        windows=(
            ProtocolWindowSpec(
                id="summary_timepoint",
                summary="Select a single summary timepoint by nearest/exact/neighbor rules.",
                anchor="analysis_time",
                selector="time_mode",
                params={"default_mode": "nearest"},
            ),
        ),
        metrics=(
            ProtocolMetricSpec(
                id="vec8",
                stage="summary",
                summary="Eight-value SFXI summary vector over logic and intensity channels.",
                formula="v00,v10,v01,v11,y00,y10,y01,y11",
            ),
        ),
        figures=(
            ProtocolFigureSpec(
                id="logic_summary",
                kind="summary",
                summary="SFXI vec8 summary and corner-balanced logic view.",
                primary=True,
            ),
        ),
        deliverables=(
            ProtocolDeliverableSpec(
                id="time_series",
                surface="plots",
                summary="Primary time-series kinetics panel.",
                default=True,
            ),
            ProtocolDeliverableSpec(
                id="snapshot_by_channel",
                surface="plots",
                summary="Snapshot barplots grouped by treatment/channel.",
                default=True,
            ),
            ProtocolDeliverableSpec(
                id="snapshot_by_design",
                surface="plots",
                summary="Snapshot barplots grouped by design.",
                default=True,
            ),
            ProtocolDeliverableSpec(
                id="snapshot_state",
                surface="plots",
                summary="2x2 state summary panel for alias-mapped treatment states.",
            ),
            ProtocolDeliverableSpec(
                id="ts_and_snap_intensity",
                surface="plots",
                summary="Combined time-series and endpoint view of the intensity channel.",
                default=True,
            ),
            ProtocolDeliverableSpec(
                id="ts_and_snap_ratio",
                surface="plots",
                summary="Combined time-series and endpoint view of the dual-reporter ratio.",
            ),
            ProtocolDeliverableSpec(
                id="distributions",
                surface="plots",
                summary="Distribution view of the primary ratio channel.",
            ),
            ProtocolDeliverableSpec(
                id="snapshot_heatmap_yfp_cfp",
                surface="plots",
                summary="Snapshot heatmap for YFP/CFP.",
            ),
            ProtocolDeliverableSpec(
                id="snapshot_heatmap_cfp_od600",
                surface="plots",
                summary="Snapshot heatmap for CFP/OD600.",
            ),
            ProtocolDeliverableSpec(
                id="logic_symmetry_yfp_cfp",
                surface="plots",
                summary="Logic symmetry view over the YFP/CFP channel.",
            ),
            ProtocolDeliverableSpec(
                id="vec8_xlsx",
                surface="exports",
                summary="Workbook export of the SFXI vec8 summary.",
            ),
        ),
        ranking=ProtocolRankingSpec(
            primary_metric="vec8",
            direction="higher_is_better",
            summary="Interpret SFXI through the vec8 summary surface and its downstream logic-specific analyses.",
        ),
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/sfxi_eda",
                allowed_templates=("notebook/sfxi_eda", "notebook/eda", "notebook/basic"),
                summary="SFXI logic screens default to the vec8-aware notebook scaffold.",
            ),
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/synergy_h1",
                    summary="Logic screens default to CFP/YFP/OD600 Synergy H1 ingest with protocol-bound channel policy.",
                    with_={
                        "mode": binding_value("ingest.mode", "auto"),
                        "channels": binding_value("ingest.channels", ["OD600", "CFP", "YFP"]),
                        "sheet_names": binding_value("ingest.sheet_names", None),
                        "add_sheet": binding_value("ingest.add_sheet", False),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="validator/to_tidy_plus_map",
                    summary="Promote plate-reader tidy data into the annotated SFXI-compatible table shape.",
                    with_={"synthesize_batch": binding_value("promote.synthesize_batch", True)},
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/fold_change",
                    summary="Shared logic-screen fold-change defaults keep target and baseline policy in one place.",
                    with_={
                        "target": binding_value("fold_change.target", "YFP/CFP"),
                        "report_times": binding_value("fold_change.report_times"),
                        "time_tolerance": binding_value("fold_change.time_tolerance", 0.51),
                        "agg": binding_value("fold_change.agg", "median"),
                        "treatment_column": binding_value("fold_change.treatment_column", "treatment"),
                        "group_by": binding_value("fold_change.group_by", ["design_id"]),
                        "use_global_baseline": binding_value("fold_change.use_global_baseline", True),
                        "global_baseline_value": binding_value("fold_change.global_baseline_value", None),
                        "overrides": binding_value("fold_change.overrides", []),
                        "fc_column": binding_value("fold_change.fc_column", "FC"),
                        "log2fc_column": binding_value("fold_change.log2fc_column", "log2FC"),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/sfxi",
                    summary="Default SFXI vec8 build settings derive from the protocol instead of every step config.",
                    with_={
                        "response": {
                            "logic_channel": binding_value("response.logic_channel", "YFP/CFP"),
                            "intensity_channel": binding_value("response.intensity_channel", "YFP/OD600"),
                        },
                        "design_by": binding_value("design_by", ["design_id"]),
                        "time_column": binding_value("time_column", "time"),
                        "time_mode": binding_value("time_mode", "nearest"),
                        "target_time_h": binding_value("target_time_h", None),
                        "time_tolerance_h": binding_value("time_tolerance_h", 0.5),
                        "logic_map_ref": binding_value("logic_map_ref", "induction_logic"),
                        "reference": {
                            "design_id": binding_value("reference.design_id", "REF"),
                            "stat": binding_value("reference.stat", "mean"),
                        },
                        "require_all_corners_per_design": binding_value("require_all_corners_per_design", True),
                        "exclude_reference_from_output": binding_value("exclude_reference_from_output", True),
                        "carry_metadata": binding_value("carry_metadata", ["sequence", "id"]),
                    },
                ),
            ),
            compiler=compile_logic_sfxi_screen,
        ),
    ),
    ProtocolDescriptor(
        protocol="cytometry/flow_panel",
        domain="cytometry",
        family="panel_analysis",
        summary="Flow-cytometry panel protocol for gated event tables and channel-level summaries.",
        tags=("cytometry", "fcs", "panel"),
        factors=(
            ProtocolFactorSpec(name="sample", role="sample", summary="Sample/run identifier."),
            ProtocolFactorSpec(name="condition", role="condition", summary="Experimental condition."),
            ProtocolFactorSpec(name="gate", role="gate", summary="Gate or subset definition.", required=False),
        ),
        figures=(
            ProtocolFigureSpec(
                id="cytometry_qc",
                kind="qc",
                summary="Channel/gating QC precedes downstream comparisons.",
                primary=True,
            ),
        ),
        ranking=ProtocolRankingSpec(
            primary_metric="domain_defined",
            direction="higher_is_better",
            summary="Use domain-defined cytometry comparisons and gating summaries for ranking.",
        ),
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/cytometry",
                allowed_templates=("notebook/cytometry", "notebook/basic"),
                summary="Flow-cytometry panels default to the cytometry EDA notebook scaffold.",
            ),
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/flow_cytometer",
                    summary="Flow-panel ingest defaults come from protocol parameters instead of per-experiment plugin blobs.",
                    with_={
                        "auto_roots": binding_value("ingest.auto_roots", None),
                        "auto_include": binding_value("ingest.auto_include", ["*.fcs", "*.FCS"]),
                        "auto_exclude": binding_value("ingest.auto_exclude", list(DEFAULT_EXCLUDE)),
                        "auto_pick": binding_value("ingest.auto_pick", "merge"),
                        "auto_recursive": binding_value("ingest.auto_recursive", False),
                        "channel_name_field": binding_value("ingest.channel_name_field", "pns"),
                        "channel_map": binding_value("ingest.channel_map", None),
                        "drop_channels": binding_value("ingest.drop_channels", None),
                        "sample_id_from": binding_value("ingest.sample_id_from", "stem"),
                        "time_value": binding_value("ingest.time_value", 0.0),
                        "print_summary": binding_value("ingest.print_summary", True),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/sample_metadata",
                    summary="Metadata merge requirements are protocol parameters, not raw plugin overrides.",
                    with_={
                        "key": binding_value("metadata.key", "sample_id"),
                        "require_columns": binding_value("metadata.require_columns", []),
                        "require_non_null": binding_value("metadata.require_non_null", False),
                    },
                ),
            ),
            compiler=compile_cytometry_flow_panel,
        ),
    ),
)


@cache
def builtin_protocol_catalog() -> ProtocolCatalog:
    return ProtocolCatalog(list(BUILTIN_PROTOCOLS))
