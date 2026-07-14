---
doc_id: reader-sfxi-plots
surface: plot-reference
owner: reader-maintainers
last_verified: 2026-07-14
summary: Reader SFXI plot inputs, configuration, outputs, descriptions, and dependency boundaries.
---

# SFXI plot surfaces

Reader plots consume typed records. Plot plugins handle configuration and file
publication; calculation and figure assembly remain in
`reader.domains.logic.sfxi`. This keeps plot rendering separate from vec8
generation and from OPAL campaign analysis.

Return to the [SFXI entry point](../sfxi_vec8_in_reader.md) for the calculation,
operation, and handoff routes.

## Discover and preflight plots

Specialized SFXI figures are opt-in. The default `logic_overview` profile does
not select them.

```bash
uv run reader plot <config-or-experiment> --list --format json
uv run reader plot <config-or-experiment> --dry-run
```

To select a per-experiment heatmap and setpoint scatter without the default
plot profile:

```yaml
protocol:
  id: logic/sfxi_screen
  outputs:
    plots:
      profile: none
      include:
        - sfxi_vec8_heatmap
        - sfxi_setpoint_scatter
```

Run a selected figure after its dry-run passes:

```bash
uv run reader plot <config-or-experiment> --only sfxi_vec8_heatmap
```

## Per-experiment vec8 heatmap

`sfxi_vec8_heatmap` reads `sfxi_vec8/vec8` and writes
`outputs/plots/sfxi_vec8_heatmap.pdf` by default.

Figure description: one row represents one `design_id`. The first four columns
show `v00`, `v10`, `v01`, and `v11` with a unit-interval color scale. The next
four show `y00_star`, `y10_star`, `y01_star`, and `y11_star` with a separate
log2-intensity color scale centered at zero. A vertical divider and separate
color bars keep the two value spaces distinct.

Use this figure for one experiment. Use the aggregate command below when rows
come from multiple Reader experiments or explicit table snapshots.

## Setpoint scatter

`sfxi_setpoint_scatter` reads `sfxi_vec8/vec8`, calls the public
`dnadesign.opal.api.sfxi` scoring API, and writes
`outputs/plots/sfxi_setpoint_scatter.pdf` by default. Reader does not import
OPAL internals or define a second scalar objective.

Each configured setpoint gets one panel:

- x-axis: `logic_fidelity`
- y-axis: `effect_scaled`
- point color: scalar `sfxi`, fixed from 0 to 1

Figure description: each panel places measured designs in a unit-square
logic-versus-effect plane. Point color encodes the combined SFXI score. When a
protocol record is used, each measured design contributes one point per
setpoint. Optional labels show `design_id`.

Configure setpoints and the shared intensity offset under the protocol analysis
surface:

```yaml
protocol:
  analysis:
    sfxi_objective:
      setpoints:
        and: [0.0, 0.0, 0.0, 1.0]
        or: [0.0, 1.0, 1.0, 1.0]
      scaling:
        percentile: 95
        min_n: 5
        eps: 1.0e-8
      exponents:
        logic_exponent_beta: 1.0
        intensity_exponent_gamma: 1.0
      intensity_log2_offset_delta: 0.0
```

Preflight fails when the public dnadesign API is unavailable, has an
unsupported version, or when the vec8 offset provenance disagrees with the
configured scorer offset. Reader source checkouts pin a compatible dnadesign
revision through `uv sync --locked --group dnadesign`. Packaged Reader
installations must install a compatible dnadesign build separately.

## Triptych sequence bundle

`sfxi_triptych_sequence` reads both `sfxi_vec8/vec8` and the annotated assay
record `promote_to_tidy_plus_map/df`. Reader owns the assay panels, layout,
output paths, and record provenance. Candidate identity and sequence metadata
come from one exact, study-issued Reader candidate-binding resource. Reader
does not open a study candidate table or resolve aliases itself. Sequence
rendering uses only the public `dnadesign.baserender` API under contract
`dnadesign.baserender.sequence_panel.v1`.

Figure description: each design page has three assay panels across the top:
growth kinetics, reporter-ratio kinetics, and a selected-time treatment
snapshot. A sequence-annotation panel spans the bottom. The page title uses
the design display label. Treatment colors and markers remain consistent
across the assay panels.

Treatment identity is not configured again for this plot. The compiler carries
`protocol.inputs.logic_map_ref` into the plot, and the plot resolves the same
`annotations.logic_maps` entry used by the SFXI vec8 transform. The four plot
series use the stable states `00`, `10`, `01`, and `11`; the authored condition
labels and treatment column come from that resolved map. The bundle manifest
records the map reference, column, corner labels, and case-sensitivity policy.
Missing states or a second plot-local treatment mapping are errors.

Declare the binding manifest as a file resource, then opt into the figure:

```yaml
resources:
  promoter_candidate_bindings:
    kind: file
    path: ./inputs/promoter_candidate_bindings/manifest.json
protocol:
  analysis:
    sfxi_triptych_sequence:
      candidate_bindings_resource: promoter_candidate_bindings
      sequence_panel:
        profile: promoter_compact_slide.v1
      snapshot_target_time_h: 12.0
  outputs:
    plots:
      profile: none
      include: [sfxi_triptych_sequence]
```

The plot rejects duplicate vec8 rows per design, missing or duplicate exact
alias bindings, sequence mismatches, binding digest or schema drift, missing
assay rows, treatment labels that do not satisfy the resolved SFXI logic map,
unsupported sequence adapters, and an unavailable public dnadesign contract.
It does not fall back to fuzzy, prefix, sequence-only, or direct USR joins, and
it does not substitute a local sequence renderer.

Rendering happens in a staging directory. Before publication, Reader verifies
the complete poster, PDF, index, manifest, frame directory, and optional movie.
Publication backs up the complete prior bundle and restores it after a caught
backup or install failure; a failed first publication removes every partially
installed artifact. Disabling movie output removes a prior MP4 within the same
rollback-safe transaction. Cleanup failures are reported. A successful bundle
contains:

- poster PNG and multipage PDF under
  `outputs/plots/sfxi_triptych_sequence/`
- ordered per-design PNG frames in a bundle-specific subdirectory
- index CSV under `outputs/exports/sfxi_triptych_sequence/`
- JSON manifest under `outputs/manifests/` with schema
  `reader.sfxi_triptych_sequence_bundle.v2`
- file-bundle record `plot:sfxi_triptych_sequence`
- optional MP4 when `movie_enabled: true` and `ffmpeg` is available

The index and manifest provide the text description for each page: design,
candidate and sequence-authority identities, binding and candidate-table
digests, selected and observed snapshot time, optional acquisition transition,
frame path, and sequence-panel diagnostics. An acquisition transition is
workbook provenance, not a biological event declaration.

## Cross-experiment aggregate heatmap

`reader aggregate-sfxi-vec8` is separate from protocol plot execution. It
accepts experiment configs, experiment directories, output directories, or
explicit `.csv`, `.parquet`, and `.xlsx` vec8 tables.

Experiment and output-directory inputs must provide the typed
`sfxi_vec8/vec8` record. Reader verifies the record artifact instead of
silently choosing an exported workbook. Pass a workbook directly only when the
workbook itself is the intended review snapshot.

```bash
uv run reader aggregate-sfxi-vec8 \
  experiments/2026/20260706_sfxi_sensor-panel-m9-glu-secg/config.yaml \
  experiments/2026/20260707_sfxi_sensor-panel-m9-glu-secg/config.yaml \
  --out-dir outputs/reviews/sfxi_vec8_aggregate
```

The command writes one bundle:

- `sfxi_vec8_heatmap.png`
- `sfxi_vec8_heatmap_tidy.csv`
- `sfxi_vec8_heatmap_manifest.json`

Figure description: rows are measured designs across sources. Controls sort
first, followed by natural design order. The two four-column blocks and color
scales have the same meanings as the per-experiment heatmap. Row labels include
the source, design, and selected snapshot time when space permits.

The tidy CSV preserves full source and row provenance. Record-backed manifest
entries include contract, content digest, config digest, creation time, and
producer metadata. The manifest also reports every observed
`intensity_log2_offset_delta`; mixed values are explicit because those rows do
not share one linear-intensity inverse.

Existing bundles are not replaced unless `--overwrite` is passed. The writer
stages all three files and rolls back a partial commit.

## Source map

- [vec8_heatmap.py](../../../src/reader/domains/logic/sfxi/vec8_heatmap.py)
- [setpoint_scatter.py](../../../src/reader/domains/logic/sfxi/setpoint_scatter.py)
- [triptych_sequence.py](../../../src/reader/domains/logic/sfxi/triptych_sequence.py)
- [vec8_aggregate](../../../src/reader/domains/logic/sfxi/vec8_aggregate/)
