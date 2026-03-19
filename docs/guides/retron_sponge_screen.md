# Retron Sponge Screen Guide

`plate_reader/retron_sponge_screen` is the matched-control sponge assay in `reader`.
Use it when the experiment depends on:

- same-sensor matched controls such as `tetO`
- a fixed 2x2 induction/stress design
- compiled matched-control semantics such as `B`, `C`, `D`, `M`, `O`, burden, leakiness, and ranking

This workflow uses the direct-ratio path only. It does not apply residual correction or background subtraction.

## Assay contract

Lock these assumptions before analysis:

- each genotype is one sensor plasmid plus one sponge plasmid
- each genotype is measured under four states:
  - `-IPTG/-stress`
  - `+IPTG/-stress`
  - `-IPTG/+stress`
  - `+IPTG/+stress`
- matched controls stay same-sensor, same-plate, same-stress, same-IPTG, same-timepoint
- all core calculations happen at the well level before aggregation

The transform `transform/retron_sponge_metrics` materializes two typed records:

- `semantic_metrics/trace`
  - contract: `plate_reader.sponge_trace.v1`
  - carries `R`, `B`, `C`, `D`, `M`, `O`, `mu`, window flags, and the derived matching metadata
- `semantic_metrics/summary`
  - contract: `plate_reader.sponge_summary.v1`
  - carries `R_pre`, `C_AUC`, `C_END`, `D_AUC`, `D_END`, `M_AUC`, `M_END`, `O_AUC`, `S_AUC`, `L_pre`, `L_post_AUC`, `T_ratio_AUC`, `T_growth_AUC`, and `T_finalOD`

## Metric flow

The compiled semantic sequence is:

1. inspect raw channels such as `OD600`, `CFP`, `YFP`
2. keep growth-normalized support channels such as `YFP/OD600` and `CFP/OD600`
3. compute the primary within-well readout `R = log2(YFP / CFP)` or the configured single-reporter analogue
4. baseline each well to `R_pre`
5. compute `B = R - R_pre`
6. normalize to the matched same-sensor control with `C`
7. isolate the induced effect with `D`
8. compare relevant stress to the no-stress condition with `M`
9. sign-correct for cross-sensor ranking with `O`
10. summarize with AUC and endpoint metrics, then rank with burden and leakiness penalties

For single-reporter retron assays, the same matched-control program applies, but the primary ratio becomes `log2(reporter / growth_channel)` and the support channel becomes `reporter / growth_channel`.

## CLI flow

Start with discovery and contract inspection:

```bash
uv run reader protocols plate_reader/retron_sponge_screen
uv run reader protocols plate_reader/retron_sponge_screen --format json
uv run reader inspect experiments/2026/20260317_tetra_functional_sponges/config.yaml
uv run reader explain experiments/2026/20260317_tetra_functional_sponges/config.yaml --format json
```

Validate before execution:

```bash
uv run reader validate experiments/2026/20260317_tetra_functional_sponges/config.yaml
uv run reader validate experiments/2026/20260317_tetra_functional_sponges/config.yaml --format json
```

Run the pipeline, then materialize plots and exports:

```bash
uv run reader run experiments/2026/20260317_tetra_functional_sponges/config.yaml
uv run reader plot experiments/2026/20260317_tetra_functional_sponges/config.yaml --list
uv run reader plot experiments/2026/20260317_tetra_functional_sponges/config.yaml
uv run reader export experiments/2026/20260317_tetra_functional_sponges/config.yaml --list
uv run reader export experiments/2026/20260317_tetra_functional_sponges/config.yaml
uv run reader records experiments/2026/20260317_tetra_functional_sponges/config.yaml --format json
uv run reader notebook experiments/2026/20260317_tetra_functional_sponges/config.yaml --mode none
```

## Compiled plot surface

The default `screen_overview` profile materializes the core review portfolio:

- `raw_kinetics`
- `support_kinetics`
- `control_burden_panel`
- `matched_control_kinetics`
- `induced_effect_kinetics`
- `interaction_summary`
- `library_heatmaps`
- `stress_modulation_scores`
- `pareto_ranking`

Additional profiles:

- `kinetics_qc`
  - raw/support/control burden review
- `analysis_review`
  - baseline-shifted, matched-control, induced, summary, and ranking review

To materialize the full 10-figure retron portfolio without dropping QC plots, keep
`profile: screen_overview` and add `baseline_shifted_kinetics` under
`protocol.outputs.plots.include`. The March 2026 mono/bi/tri/tetra sponge-screen
configs use that pattern so the newer screens exercise the complete registered
retron plot surface.

## Notebook surface

`reader notebook` defaults retron sponge screens to `notebook/retron_sponge`.
That scaffold is experiment-scoped and progressive:

- it inventories the selected plot portfolio by review phase
- it adds a direct-ratio transform ladder so each figure is tied back to the math that produced it
- it shows whether each selected plot/export has already been rendered
- it prefers the semantic summary/trace records when present
- it keeps the semantic table review decoupled from bespoke assay plotting code

The guide figures from the full analysis template that are not first-class compiled plot ids should be built from the exported semantic tables rather than by adding assay-specific one-off plot plugins prematurely.
For cross-run library review, scaffold a small manifest-backed `workbench/generic`
experiment that selects `notebook/retron_sponge_aggregate`. That notebook combines
semantic exports from the March 2026 screen families into cross-run figures such as
specificity matrices, architecture plots, expected-versus-observed multifunction
comparisons, and sponge fingerprints.

## Export surface

The retron assay exports the semantic tables directly:

- `semantic_trace_table` -> `outputs/exports/retron/semantic_trace.csv`
- `semantic_summary_table` -> `outputs/exports/retron/semantic_summary.csv`

Those exports are the didactic and extensible bridge between the compiled assay program and downstream bespoke figures such as architecture comparisons, constituent-vs-multifunction expected/observed plots, sponge fingerprints, or plate-position diagnostics.

## Config guidance

The minimum assay-specific block lives under `protocol.analysis.semantic_metrics`.
Common keys:

- `control_name`
- `no_stress_label`
- `states`
- `relevant_stress_map`
- `sensor_target_map`
- `expected_sign_map`
- `plateau`

Keep state labels explicit and stable. Do not hide assay meaning in plot-specific overrides.

## Pressure-test checklist

Before trusting a new retron sponge experiment:

- `reader inspect` shows `plate_reader/retron_sponge_screen`
- `reader explain --format json` shows `semantics.program.summary.descriptive_only == 0`
- `reader plot --list` shows the retron-specific outputs above, not generic dual-reporter-only outputs
- `reader export --list` shows both semantic table exports
- `reader records --format json` shows `semantic_metrics/trace` and `semantic_metrics/summary`

If one of those fails, fix the protocol/config boundary instead of patching generated outputs.
