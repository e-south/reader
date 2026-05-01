# SFXI Triptych Sequence Plot Plugin Spec

Status: Initial implementation slice landed

Audience: `reader` and `dnadesign` maintainers

## Summary

The current SFXI triptych sequence renderer is useful as an
experiment-scoped preview, but it is not yet a durable `reader` integration.
It owns a standalone JSON config, custom manifest, dry-run behavior, output
cleanup, BaseRender style shims, and raster normalization path outside the
canonical `reader` experiment and records surfaces.

This spec promotes the workflow into a formal `reader` plot plugin while
preserving the package boundary:

- `reader` owns experiment config, protocol semantics, plate-reader plots,
  figure composition, output persistence, records, dry-run, and validation.
- `dnadesign` owns sequence records, USR/DenseGen projections, GenBank
  overlays, BaseRender styles, render profiles, and sequence-render contract
  versions.

The target outcome is a plot that "just works" through normal `reader`
surfaces, without private `dnadesign.*.src.*` imports, silent fallbacks, or
manual output cleanup.

## Implementation Status

The initial vertical slice is implemented:

- `plot/sfxi_triptych_sequence` is registered as a formal reader plot plugin.
- `logic/sfxi_screen` can expose `sfxi_triptych_sequence` as a semantic plot
  output when configured.
- The plot emits one canonical file-bundle record through reader's existing
  file-bundle record path.
- Rendering publishes through a staging directory so a failed render does not
  delete the previous successful bundle.
- Reader imports only public dnadesign surfaces.
- dnadesign exposes `dnadesign.baserender.sequence_panel.v1`,
  `promoter_compact_slide.v1`, public style helpers, and
  `render_sequence_panel_image`.

Remaining follow-up work:

- replace any remaining experiment-local historical scripts with the plugin
  path where appropriate
- broaden visual smoke coverage beyond the synthetic fixture
- decide whether the protocol default profile should eventually include this
  heavier bundle or keep it opt-in

## Problem

The current preview lives at:

```text
experiments/2026/20260501_sfxi_promoter_setpoint_scatter/
```

It successfully renders an SFXI review bundle, but it has contract drift from
the rest of `reader`:

- it is not discoverable as a first-class `reader/v7` experiment plot surface
- it writes a sidecar manifest instead of one canonical bundle record
- config validation is shallow and mostly top-level
- output cleanup happens before successful replacement
- BaseRender style details and image normalization live in experiment code
- dry-run is not a machine-readable preflight surface

The fix should not move SFXI scoring or sequence rendering into the wrong
package. The fix is to narrow and version the boundary.

## Goals

- Add a formal `reader` plot plugin:

  ```text
  plot/sfxi_triptych_sequence
  ```

- Persist one canonical figure bundle record containing all subplot outputs and
  provenance.
- Keep user-facing authoring semantic and protocol-owned, not plugin-shaped.
- Keep sequence rendering behind a public `dnadesign` contract.
- Fail fast when required data, package APIs, or contract versions are missing.
- Preserve the current preview output as the behavior and visual baseline unless
  intentionally revised.
- Make dry-run and validate useful for CI and agent workflows.
- Make failed renders preserve the last successful bundle.

## Non-Goals

- Do not change SFXI vec8 math or SFXI objective scoring.
- Do not introduce reader-side reimplementations of BaseRender.
- Do not import `dnadesign.*.src.*` from `reader`.
- Do not add hidden fallback rendering or hidden scoring compatibility modes.
- Do not generalize every future sequence visualization before this SFXI plot is
  stable.
- Do not hand-edit generated files under `experiments/**/outputs/`.

## Ownership Boundaries

| Surface | Owner | Notes |
| --- | --- | --- |
| Experiment config | `reader` | Public config remains `reader/v7`. |
| Protocol semantics | `reader` | `logic/sfxi_screen` decides when this plot is exposed. |
| Plate-reader traces | `reader` | OD600, YFP/CFP, snapshot selection, CIs, and labels. |
| Figure composition | `reader` | Multi-row triptych layout and bundle persistence. |
| Canonical artifact records | `reader` | One bundle record in `outputs/manifests/records.json`. |
| Sequence records | `dnadesign` | USR/DenseGen/GenBank source of sequence truth. |
| Sequence render semantics | `dnadesign` | BaseRender adapters, labels, styles, and render diagnostics. |
| Style profiles | `dnadesign` | Public named profile, not copied dict shims in `reader`. |
| SFXI scalar objective | `dnadesign` | Existing OPAL public scoring API remains authoritative. |

## Public Contract Names

Use stable, consumer-neutral names:

```text
dnadesign.baserender.sequence_panel.v1
promoter_compact_slide.v1
reader.sfxi_triptych_sequence_bundle.v1
```

Rationale:

- `sequence_panel.v1` describes the abstraction, not the current notebook or
  SFXI use case.
- `promoter_compact_slide.v1` describes the visual profile and can survive new
  assay consumers.
- `reader.sfxi_triptych_sequence_bundle.v1` describes the persisted figure
  bundle contract owned by `reader`.

Avoid names such as `notebook_render_contract`, `densegen_promoter_only`, or
`sfxi_baserender_hack`; those encode an implementation path instead of a stable
boundary.

## Proposed Reader Surfaces

### Domain Module

```text
src/reader/domains/logic/sfxi/triptych_sequence.py
```

Responsibilities:

- validate the typed triptych config
- resolve input artifacts and required columns
- build the row-level render plan
- join vec8 rows to sequence metadata
- call the optional dnadesign adapter
- compose each promoter figure
- build the canonical bundle manifest payload

It should not:

- own BaseRender feature layout rules
- parse private dnadesign record internals
- mutate generated outputs before successful render completion
- expose plugin IDs as public assay semantics

### Plot Plugin

```text
src/reader/plugins/plot/sfxi_triptych_sequence.py
```

Plugin id:

```text
plot/sfxi_triptych_sequence
```

Plugin responsibilities:

- declare input ports and file-bundle output port
- load `PluginConfig`/Pydantic config
- call the domain primitive
- participate in `reader plot --list`, `reader validate`, and dry-run surfaces
- report missing optional dependencies as actionable preflight issues

The plugin should stay a thin adapter. Plot mechanics and validation belong in
the domain module.

### Protocol Exposure

Expose through `logic/sfxi_screen` as a semantic plot output:

```yaml
protocol:
  id: logic/sfxi_screen
  outputs:
    plots:
      include:
        - sfxi_triptych_sequence
```

User config should not need to name internal plugin config fields unless using a
maintainer or expert override.

## Proposed dnadesign Surfaces

Expose a public sequence-panel contract from `dnadesign.baserender`.

Candidate public API:

```python
from dnadesign.baserender import (
    BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
    SequencePanelConfig,
    SequencePanelDiagnostics,
    list_style_presets,
    render_sequence_panel_image,
    resolve_style,
)
```

Required behavior:

- contract id: `dnadesign.baserender.sequence_panel.v1`
- default profile: `promoter_compact_slide.v1`
- input: public sequence render records or adapter-supported public records
- output: image/renderable plus diagnostics
- diagnostics include bounds, strand count, feature count, legend entries,
  profile id, and warnings
- invalid style or palette values fail during style resolution, not after render

This API should absorb the current need for reader-side raster crop thresholds,
manual canvas sizing, large copied style dictionaries, and source-string
sentinel knowledge.

## Canonical Bundle Artifact

The canonical output is one bundle record. Individual PNG/PDF/MP4/index files
are members of the bundle, not independent top-level records.

Record identity:

```yaml
artifact_type: figure_bundle
artifact_subtype: sfxi_triptych_sequence
contract_version: reader.sfxi_triptych_sequence_bundle.v1
```

Expected bundle members:

```text
outputs/plots/sfxi_triptych_sequence/<bundle_id>.png
outputs/plots/sfxi_triptych_sequence/<bundle_id>.pdf
outputs/plots/sfxi_triptych_sequence/<bundle_id>.mp4
outputs/exports/sfxi_triptych_sequence/<bundle_id>_index.csv
outputs/manifests/records.json
```

Required manifest fields:

| Field | Meaning |
| --- | --- |
| `bundle_id` | Stable id for this rendered bundle. |
| `contract_version` | `reader.sfxi_triptych_sequence_bundle.v1`. |
| `plot_id` | `sfxi_triptych_sequence`. |
| `protocol_id` | Expected `logic/sfxi_screen` when protocol-bound. |
| `source_experiment_id` | Reader experiment id. |
| `source_vec8_artifact` | Source vec8 artifact path or record id. |
| `row_count` | Number of promoter panels rendered. |
| `row_order` | Ordered promoter/design ids in the bundle. |
| `reference_rows` | Reference promoter ids and degraded/full status. |
| `vec8_selected_time_h` | Time used for vec8 derivation. |
| `snapshot_target_time_h` | Requested visual snapshot time. |
| `snapshot_observed_time_h` | Actual snapshot time used. |
| `snapshot_fell_back` | Boolean fallback flag. |
| `snapshot_fallback_delta_h` | Difference between target and observed time. |
| `dnadesign_contract_id` | `dnadesign.baserender.sequence_panel.v1`. |
| `dnadesign_contract_version` | Version reported by dnadesign. |
| `sequence_profile_id` | Example: `promoter_compact_slide.v1`. |
| `outputs` | Map of PNG/PDF/MP4/index paths. |
| `created_at` | Timestamp. |

## Config Shape

The plugin should use a typed config model. Public protocol config should stay
semantic; plugin config is a maintainer surface.

Sketch:

```yaml
analysis:
  sfxi_triptych_sequence:
    vec8_source: sfxi.vec8.v2
    sequence_source:
      provider: dnadesign.usr
      dataset: usr_sfxi_pdual10_densegen_promoters
      overlay: densegen_promoter_annotations
    references:
      include:
        - pDual-10-spyp
        - pDual-10-sulAp
    time:
      snapshot_target_time_h: 12.0
      induction_time_h: 12.0
    render:
      sequence_contract: dnadesign.baserender.sequence_panel.v1
      sequence_profile: promoter_compact_slide.v1
      movie_fps: 0.85
```

Validation rules:

- `snapshot_target_time_h` must be numeric and finite.
- `induction_time_h` must be numeric and finite when shown.
- `sequence_contract` must match a supported dnadesign public contract.
- `sequence_profile` must exist according to `dnadesign.baserender`.
- reference ids must be resolved explicitly or reported as degraded references.
- required vec8 columns must be present with canonical names.

## Runtime Lifecycle

1. Load and validate reader config.
2. Resolve plugin input records and declared artifacts.
3. Check optional `dnadesign` dependency.
4. Check `dnadesign.baserender.sequence_panel.v1` compatibility.
5. Validate SFXI vec8 input columns and row identity.
6. Resolve USR/DenseGen/GenBank sequence records.
7. Assert sequence equality where both reader and dnadesign provide sequence
   strings.
8. Build row-level render plan.
9. In dry-run mode, emit JSON plan and stop before rendering.
10. Render figures into a staging directory.
11. Verify expected PNG/PDF/MP4/index members exist.
12. Atomically publish the bundle.
13. Register one canonical bundle record.

## Failure and Degraded-Mode Contract

No silent fallback is allowed.

| Condition | Behavior |
| --- | --- |
| `dnadesign` not installed | Fail fast with `reader[dnadesign]` install/update guidance. |
| Missing sequence-panel API | Fail fast with expected public API and version. |
| Incompatible contract version | Fail fast with expected and actual version. |
| Missing candidate sequence overlay | Fail unless explicitly configured as optional. |
| Missing reference annotation | Mark reference as degraded, include reason in manifest, do not pretend full sequence evidence exists. |
| Snapshot target not present | Use nearest only under explicit policy, record fallback fields. |
| Render error | Keep last successful bundle; staging output is discarded. |
| Bad style/palette | Fail during config/style validation. |

## Atomic Publication

Rendering should never delete the previous good bundle before the new bundle is
complete.

Required approach:

1. Create a staging directory under `outputs/.staging/` or an equivalent temp
   location.
2. Render all row images and bundle outputs into staging.
3. Validate expected files and manifest payload.
4. Move or copy into final `outputs/plots`, `outputs/exports`, and
   `outputs/manifests` paths.
5. Clean staging only after success.

If any step fails before publication, final outputs remain untouched.

## Test Plan

### Reader Unit Tests

- typed config accepts valid minimal config
- typed config rejects unknown contract ids
- typed config rejects invalid time values
- vec8 validation rejects missing canonical columns
- row-order planner places references first when requested
- snapshot fallback metadata is recorded

### Reader Plugin Tests

- `reader plot --list` exposes `sfxi_triptych_sequence`
- missing dnadesign API appears in validate/dry-run preflight
- dry-run JSON includes bundle id, row count, row order, output paths, and
  contract versions
- full render writes one canonical bundle record
- interrupted render preserves previous outputs

### dnadesign Tests

- public root facade exports sequence-panel contract helpers
- `promoter_compact_slide.v1` resolves through public style APIs
- invalid palette values fail during style resolution
- near-feature labels render from public hints, not private source sentinels
- render diagnostics report two-strand state when requested

### Integration Smoke

- render a two-row fixture with one DenseGen promoter and one GenBank reference
- assert nonempty PNG/PDF outputs
- assert MP4 is produced when movie output is enabled
- assert canonical `records.json` contains one bundle record
- assert no `dnadesign.*.src.*` import appears in reader implementation

## Delivery Slices

### Slice 1: Reader Safety Wrapper

Goal: harden the current behavior without changing plot appearance.

In scope:

- typed config model
- machine-readable dry-run
- atomic staging and publication
- canonical bundle record bridge

Done when:

- the current preview output regenerates
- failed render preserves previous outputs
- dry-run reports planned row count and bundle paths
- `records.json` contains the bundle record

### Slice 2: Formal Plot Plugin

Goal: expose the plot through normal reader plugin and protocol surfaces.

In scope:

- `reader.domains.logic.sfxi.triptych_sequence`
- `reader.plugins.plot.sfxi_triptych_sequence`
- built-in plugin manifest registration
- `logic/sfxi_screen` semantic plot exposure
- `reader plot --list` and `reader validate` support

Done when:

- users can request the plot semantically from `logic/sfxi_screen`
- plugin internals remain thin
- current experiment script can be removed or reduced to a wrapper

### Slice 3: dnadesign Sequence-Panel Contract

Goal: remove copied BaseRender style and image-normalization logic from reader.

In scope:

- public `dnadesign.baserender.sequence_panel.v1`
- `promoter_compact_slide.v1` profile
- public style helpers
- early style/palette validation
- render diagnostics

Done when:

- reader calls only public dnadesign APIs
- reader selects a profile instead of passing low-level style dictionaries
- sequence-panel diagnostics are recorded in the bundle manifest

### Slice 4: Cleanup and Regression Hardening

Goal: remove parallel preview contracts.

In scope:

- remove or retire sidecar-only manifest behavior
- remove experiment-local raster crop/style shim code
- add smoke/golden sanity coverage
- update docs and dev journal

Done when:

- the formal plugin is the maintained path
- the old experiment preview is either deleted or documented as historical
- all validation commands for the slice pass

## Acceptance Criteria

- `reader plot --list` shows `sfxi_triptych_sequence`.
- `reader validate` catches missing dnadesign, missing contract, bad config, and
  missing required inputs.
- `reader plot --dry-run --format json` emits the planned bundle, row count,
  output paths, and contract versions.
- Full render produces one canonical bundle record.
- No reader import reaches into `dnadesign.*.src.*`.
- Failed render does not delete the previous good PNG/PDF/MP4.
- Current visual output remains the baseline unless a visual change is
  intentionally approved.
- The sequence panel uses a named dnadesign style profile rather than copied
  reader-side style dictionaries.

## Validation Commands

Docs-only changes to this spec:

```bash
uv run python tools/check_docs.py
git diff --check
```

Reader implementation slices:

```bash
uv run pytest -q src/reader/tests/domains/logic/sfxi
uv run pytest -q src/reader/tests/plugins/plot
uv run pytest -q src/reader/tests/cli/test_plot_export.py
uv run ruff check .
uv run ruff format . --check
git diff --check
```

dnadesign implementation slices:

```bash
uv run pytest -q src/dnadesign/baserender/tests
uv run ruff check src/dnadesign/baserender
uv run ruff format src/dnadesign/baserender --check
git diff --check
```

## Evidence Links

- `reader/ARCHITECTURE.md`: generated outputs, records, dry-run, and domain
  ownership expectations.
- `reader/DESIGN.md`: protocol-owned semantics and fail-fast behavior.
- `reader/docs/core/plugins.md`: plugin layering and registration rules.
- `reader/docs/lib/sfxi_vec8_in_reader.md`: SFXI vec8 ownership and OPAL
  handoff.
- `dnadesign/DESIGN.md`: cross-tool coupling through documented artifacts or
  public APIs only.
- `dnadesign/src/dnadesign/baserender/docs/reference.md`: public BaseRender
  boundary and private import warning.

## Open Risks

- The exact dnadesign API shape may need adjustment once implementation starts.
- A full image golden test may be brittle; prefer structural diagnostics plus a
  small nonblank render smoke test unless pixel stability is proven.
- Existing local experiment outputs are generated and should not be treated as
  source-of-truth code artifacts.
- If USR dataset resolution still depends on a sibling checkout path, a package
  resource or explicit dataset registry path will be needed for installed
  `reader[dnadesign]` workflows.
