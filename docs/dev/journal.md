# Dev Journal

## 2026-03-16: Explicit ReaderRuntime Composition Root Slice

Implemented the next breaking ontology cleanup after the typed-port cut exposed
the remaining ambient bootstrap lie: the semantic kernels were explicit, but the
built-in `reader` runtime was still being reassembled ad hoc in CLI, engine,
validation, notebook helpers, asset composition, and record discovery.

- Added `reader.runtime` as the single composition root for the built-in
  runtime.
- Added `ReaderRuntime` as the explicit assembled world containing:
  - contract catalog
  - plugin registry
  - unified asset catalog
  - record-store factory
- Added `builtin_runtime()` as the only built-in runtime constructor.
- Rewired engine run/explain/validate flows, CLI utilities, record discovery,
  and notebook SFXI helper code to consume `ReaderRuntime` instead of
  reconstructing contracts/plugins/stores inline.
- Removed the implicit fallback inside `build_workbench_asset_catalog()`; asset
  composition now requires an explicit plugin registry.
- Added regression coverage so direct built-in composition calls cannot quietly
  spread back outside `reader.runtime`.

The resulting placement rule is clearer:

- `contracts/` defines dataframe meaning
- `assets/` defines asset descriptors
- `workbench/registry.py` resolves plugin descriptors
- `runtime/` assembles the built-in `reader` world once
- CLI, engine, notebooks, and records consume that runtime instead of
  rebuilding it

Validation for this slice:

- targeted runtime/CLI/engine/records pytest matrix
- full pytest, lint, format, and compileall after final sweep

## 2026-03-16: Typed Plugin Port Kernel + Ingest Discovery Policy Slice

Implemented the next breaking ontology cleanup after the contract-kernel cut
exposed the remaining string-protocol lie: plugin I/O semantics were still
encoded through `?` suffixes, `"none"` sentinels, and the magic output label
`"files"` instead of one explicit port model.

- Added `reader.workbench.ports` as the single owner of plugin I/O ontology:
  - `InputPortSpec`
  - `OutputPortSpec`
  - `PortKind = dataframe | file_path | file_bundle`
- Replaced `input_contracts()` / `output_contracts()` with typed
  `input_ports()` / `output_ports()` across built-in plugins and registry
  validation.
- Replaced runtime string parsing in planning, validation, execution, and
  explain flows with typed port handling.
- Deleted the `?` suffix convention for optional inputs.
- Deleted the `"none"` sentinel from plugin port declarations.
- Deleted the magic `"files"` output convention; file outputs are now explicit
  `file_path` or `file_bundle` ports.
- Updated plot/export helpers and tests to use explicit output names such as
  `artifact` / `artifacts` without any engine special-casing.
- Moved raw ingest autodiscovery policy out of `workbench/resources/` and into
  `plugins/ingest/discovery_policy.py`.
- Deleted `workbench/resources/` and added a namespace guard so it cannot
  silently survive as a ghost package.

The resulting placement rule is clearer:

- `contracts/` owns dataframe meaning
- `graph/` owns workbench refs
- `ports/` owns plugin I/O ontology
- `plugins/` own implementations only
- `plugins/ingest/discovery_policy.py` owns raw-file autodiscovery policy

Validation for this slice:

- targeted engine/registry/explain/plugin/discovery pytest matrix
- compileall over `src/reader`
- full lint, format, repo validation, and smoke runs after final sweep

## 2026-03-16: Explicit Contract Catalog + Built-In Contract Kernel Slice

Implemented the next breaking ontology cleanup after the plugin-manifest and
experiment-semantics cuts exposed the remaining ambient-state lie: dataframe
contracts were still loaded by import-time mutation through a global `BUILTIN`
dict, and built-in declarations were still split across `contracts/` and
`domains/*/contracts.py`.

- Added `reader.contracts.catalog.ContractCatalog` as the explicit runtime
  contract kernel.
- Added typed contract identifiers in `reader.contracts.model` and moved
  `OutputContractSurface` into the contract kernel.
- Added `reader.contracts.builtins/` as the single declaration home for all
  built-in dataframe contracts:
  - `generic`
  - `plate_reader`
  - `logic`
  - `cytometry`
- Replaced import-time bootstrap and mutable global registry usage with the
  explicit built-in catalog constructor `builtin_contract_catalog()`.
- Deleted:
  - `reader.contracts.registry`
  - `reader.contracts.generic`
  - `reader.domains.plate_reader.contracts`
  - `reader.domains.logic.contracts`
  - `reader.domains.cytometry.contracts`
- Rewired workbench runtime, validation, persistence, CLI utilities, and
  contract-promotion plugins to consume explicit `ContractCatalog` instances
  instead of direct `BUILTIN[...]` lookups.
- Updated `RecordStore` and `Registry` so catalog ownership is explicit at the
  runtime boundary instead of hidden behind ambient imports.
- Updated regression tests to guard the new contract-loading surface and the
  removal of global bootstrap behavior.

The resulting placement rule is clearer:

- `contracts/` owns dataframe contract ontology entirely
- `contracts/builtins/` owns built-in contract declarations
- `domains/` owns domain logic and IO only
- `workbench/` consumes explicit contract catalogs; it no longer imports global
  registry truth

Validation for this slice:

- targeted contract/registry/records/CLI/plot asset pytest matrix
- full lint, compileall, repo validation, and smoke runs to follow after final
  sweep

## 2026-03-15: Explicit Plugin Manifest + Descriptor-Driven Registry Slice

Implemented the next breaking ontology cleanup after the experiment-semantics
cut exposed the remaining directory-level lie: built-in plugin discovery still
depended on scanning `reader.plugins`, so package layout remained part of the
runtime registry contract.

- Added an explicit built-in plugin manifest in
  `reader.workbench.assets.plugin_manifest`.
- Reworked `reader.workbench.registry` into a descriptor-driven registry:
  - no `pkgutil.walk_packages`
  - no `inspect.getmembers`
  - no “register every concrete subclass we imported” behavior
- Renamed the plugin loader surface from `load_entry_points()` to
  `load_plugin_catalog()`.
- Changed external plugin loading to one explicit `reader.plugins`
  entry-point group that resolves plugin descriptors rather than bare plugin
  classes.
- Moved built-in plugin identity and semantics out of implementation classes and
  into the manifest:
  - plugin id is singular and explicit
  - category is derived from the plugin id instead of duplicated across class
    attrs and semantics
- Removed built-in plugin class metadata such as `key`, `category`, and
  `semantics`; plugin classes now own behavior only.
- Bound plugin descriptors explicitly at execution time so plot helpers can use
  resolved plugin identity without turning class attributes back into registry
  truth.
- Updated tests and maintainer docs to the new manifest-driven contract.

The resulting placement rule is clearer:

- `workbench/assets/plugin_manifest.py` owns built-in plugin registration
- `workbench/registry.py` loads explicit descriptors only
- `plugins/` contains implementations only; its directory layout is no longer a
  runtime discovery surface
- external plugins use the same descriptor concept as built-ins

Validation for this slice:

- targeted registry/assets/CLI/explain/render-path pytest matrix
- lint, compileall, full repo validation, and smoke runs after final sweep

## 2026-03-15: Experiment Semantics Kernel + Explicit Resources Slice

Implemented the next breaking ontology cleanup after the declaration/graph
split exposed the remaining lie: experiment-local meaning still lived as raw
dicts plus hidden conventions instead of one explicit semantic kernel.

- Added `reader.workbench.experiment` as the single owner of experiment-local
  semantics:
  - typed assay labels / orders / collections / logic maps
  - explicit resource catalog
  - explicit output layout
- Replaced `WorkbenchDecl.assay: dict[str, Any]` and sibling `paths` /
  `resources` buckets with `WorkbenchDecl.experiment_semantics`.
- Reworked runtime consumers to use typed experiment semantics instead of
  `dict(...)` coercion:
  - plot partition resolution
  - assay label resolution
  - logic map resolution
  - snapshot-heatmap order refs
  - notebook SFXI scaffold config expansion
- Deleted `reader.workbench.semantics` and the logic-map helper module under
  `reader.domains.logic`, which were both acting as dict-normalization buckets.
- Removed hidden `sample_map` / `metadata` resource fallback from declaration
  building, graph normalization, and CLI override parsing.
- Replaced CLI/test `file:` / `resource:` shorthand parsing with structured
  bindings only at the override boundary.
- Extended persisted record provenance so records keep optional `source_recipe`
  metadata instead of dropping recipe origin during persistence.
- Migrated tracked experiment configs that depended on implicit `sample_map` or
  `metadata` handles to explicit `resources:` declarations.

The resulting placement rule is clearer:

- `workbench/config/` parses wire syntax only
- `workbench/decl/` builds typed declarations
- `workbench/experiment/` owns experiment-local semantics
- `workbench/graph/` owns executable runtime nodes and refs
- `workbench/engine/` consumes graph + experiment semantics only
- `workbench/records/` persists typed provenance, including recipe origin

Validation for this slice:

- targeted semantics/config/CLI/records/smoke pytest matrix
- compileall over `src/reader`
- full lint + full repo validation and smoke runs to follow after final docs sync

## 2026-03-15: Declaration Layer + `workbench/model` Deletion Slice

Implemented the next breaking ontology cleanup after the typed-graph cutover
exposed the remaining lie: `workbench/config/` was still doubling as the
internal authoring model, and `workbench/model/` had decayed into a vague
types-ish bucket.

- Added `reader.workbench.decl` as the explicit internal declaration layer for:
  - bound experiment metadata and output paths
  - resource declarations
  - plugin step declarations
  - recipe call declarations
  - notebook template call declarations
- Reworked `workbench/graph/` to normalize from declaration objects instead of
  round-tripping through config classes and raw dicts.
- Stopped reusing `PluginStepSpec` outside the `config -> decl` boundary:
  - recipe helpers now author `PluginStepDecl`
  - built-in recipe assets store declaration objects
  - notebook helpers consume resolved declarations/runtime nodes directly
- Moved workbench ontology types out of the deleted `workbench/model/` package
  and into `reader.workbench.ontology`.
- Split the old monolithic `reader.workbench.assets.py` into:
  - `workbench/assets/types.py`
  - `workbench/assets/builtins.py`
  - `workbench/assets/__init__.py`
- Deleted `src/reader/workbench/model/` as a source package and added a guard
  test so it cannot silently reappear as a namespace package.
- Updated tests and docs to reflect the new internal boundary:
  `config -> decl -> graph -> engine -> records`.

The resulting placement rule is clearer:

- `workbench/config/` is wire schema and YAML loading only
- `workbench/decl/` is the single internal authored IR
- `workbench/graph/` is runtime-only
- `workbench/assets/` is the asset kernel plus built-in asset inventory
- `workbench/ontology.py` owns shared workbench semantic types
- `workbench/model/` no longer exists

Validation for this slice:

- targeted declaration/graph/CLI/explain/setup/repo-config pytest matrix
- lint and compileall
- full repo validation and smoke runs to follow after the final sweep

## 2026-03-15: Typed Workbench Graph + `reader/v4` Slice

Implemented the next rooted cutover after the audit identified the remaining
architectural lie: the workbench graph was still encoded as strings and helper
conventions instead of one explicit graph/reference model.

- Added `reader.workbench.graph` as the normalization/runtime kernel for:
  - typed input refs (`RecordRef`, `FileRef`, `ResourceRef`)
  - typed output refs (`OutputRef`)
  - typed recipe provenance (`RecipeSource`)
  - normalized runtime nodes (`PluginStep`, `NotebookTemplateCall`, `Workbench`)
- Added typed record provenance under `reader.workbench.records.model` instead
  of persisting opaque input strings.
- Bumped the config schema to `reader/v4`.
- Replaced string-shaped `reads` / `writes` config bindings with explicit
  mapping forms:
  - `reads.foo: {record: "..."}`
  - `reads.foo: {file: "..."}`
  - `reads.foo: {resource: "..."}`
  - `writes.foo: {record: "..."}`
- Moved runtime normalization out of the old `workbench/model/specs.py` path
  and into `workbench/graph/`.
- Rewired execution, validation, planning, CLI overrides, explain/materialize,
  and records persistence to consume typed refs instead of string-prefix
  routing.
- Re-authored workbench recipe declarations as typed step entries instead of
  raw step dicts with late coercion.
- Deleted the dead `workbench/model/specs.py` and `workbench/model/records.py`
  surfaces after cutover.
- Migrated tracked experiment configs to `reader/v4` and updated docs to the
  explicit binding shape.

The resulting placement rule is clearer:

- `workbench/config/` parses YAML only
- `workbench/graph/` owns the typed workbench graph
- `workbench/engine/` consumes typed graph nodes
- `workbench/records/model.py` owns persisted artifact provenance
- `workbench/recipes/*` declares typed recipe steps rather than a private
  string protocol

Validation for this slice:

- targeted graph/config/contracts/records/CLI pytest matrix
- targeted assets/notebooks/explain/setup pytest matrix
- repo config validation and smoke runs to follow after final sweep

## 2026-03-15: Unified Workbench Asset Kernel Slice

Implemented the next rooted workbench cutover after the audit identified the
remaining semantic footgun: plugins, recipes, and notebook templates still
lived in parallel catalog systems, with notebook/defaulting behavior hardcoded
in CLI and planning logic.

- Added `reader.workbench.assets` as the single workbench asset kernel.
- Unified plugin, recipe, and notebook-template descriptors under one
  `AssetDescriptor` / `AssetCatalog` model with shared semantic fields.
- Added explicit asset capabilities for notebook template behavior:
  - default-selection rules
  - plot-filter support
  - plot-spec injection
  - declared applicability requirements
- Rewired plugin registry, recipe resolution, and notebook-template resolution
  onto the shared asset kernel instead of separate parallel registries.
- Replaced weak `recipe_meta` dict breadcrumbs with typed runtime
  `source_recipe` provenance on resolved plugin steps.
- Removed hardcoded notebook/template branching from workbench CLI, scaffold,
  and planning logic in favor of descriptor capabilities.
- Reworked the SFXI notebook scaffold to discover an SFXI-capable transform via
  plugin semantics/tags instead of a literal plugin id.

The resulting placement rule is clearer:

- `workbench/assets/` is the single semantic registry surface for workbench
  assets
- `workbench/registry.py` owns executable plugin discovery only
- `workbench/recipes/*` and `workbench/notebooks/templates.py` are now static
  asset declaration sources, not independent ontology systems
- operators consume capabilities from asset descriptors instead of switching on
  concrete ids

Validation for this slice:

- targeted notebook/template + workbench/spec + CLI + recipe registry pytest
  matrix
- targeted engine/config/records/repo-config pytest matrix
- lint + compileall + real CLI validation to follow after docs sync

## 2026-03-15: Workbench Asset Ontology Cutover Slice

Implemented the breaking workbench-asset cutover after the audit identified
the remaining root footgun: one overloaded `uses` string was still standing in
for three different kinds of thing.

- Replaced shared `uses` config/runtime references with explicit fields:
  - `plugin` for executable pipeline/plot/export steps
  - `recipe` for reusable multi-step workbench compositions
  - `template` for notebook scaffold artifacts
- Renamed `reader.workbench.presets` to `reader.workbench.recipes`.
- Added explicit recipe semantics (`domain`, `family`, `summary`, `tags`)
  instead of inferring recipe type from expanded step contents.
- Split the runtime model so plugin steps and notebook template calls are no
  longer collapsed into one `WorkbenchSpec` with a boolean patch.
- Replaced ambiguous record-producer metadata (`producer.uses`) with explicit
  `producer.plugin` / `producer.template` fields.
- Deleted the remaining ghost namespace buckets:
  - `src/reader/io`
  - `src/reader/lib`
  - `src/reader/domains/plate_reader/support`
  - mirrored legacy test dirs under `src/reader/tests/{io,lib}`
- Added a guard test to keep those legacy namespace packages from silently
  reappearing.

The resulting placement and identifier rule is now clearer:

- `plugin` means executable plugin id only
- `recipe` means reusable composition only
- `template` means notebook scaffold only
- workbench runtime surfaces materialize those asset kinds separately instead of
  normalizing them through one polymorphic string

Validation for this slice:

- targeted workbench-model + config + CLI + notebook + records pytest matrices
- compileall over `src/reader`
- repo config validation and smoke matrix to follow after docs sync

## 2026-03-15: Notebook Template Ontology Slice

Implemented the next ontology cleanup slice.

- Renamed the notebook workbench surface from "preset" terminology to
  "template" terminology in the CLI, docs, and notebook scaffolding helpers.
- Added explicit `NotebookTemplateSemantics` so notebook templates now carry the
  same kind of semantic fields as other workbench catalog surfaces:
  `domain`, `family`, `summary`, `tags`.
- Removed unused notebook-template alias normalization instead of preserving a
  second naming layer with no current value.
- Tightened `NotebookTemplateCatalog` to fail fast on duplicate `uses` keys
  instead of silently overwriting descriptors.

The resulting placement rule is clearer:

- `workbench/notebooks/catalog.py` is the semantic catalog for notebook
  templates, not a special-case "preset" registry
- `workbench/notebooks/scaffold.py` writes notebook files from template
  descriptors
- `reader notebook --template` is the explicit operator surface for notebook
  selection

Validation for this slice:

- targeted notebook + CLI + config pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, and cytometer
  scaffold experiments

## 2026-03-15: Shared Domain Semantics + Labeling Kernel Slice

Implemented the next ontology cleanup slice.

- Added `reader.domains.semantics` as the shared access surface for
  domain-specific semantic resolution and the canonical plugin-domain
  vocabulary.
- Made `PluginSemantics` validate domain values eagerly instead of allowing
  ad hoc buckets such as `assay`.
- Reclassified `transform/assay_labels` under the canonical `generic` domain.
- Moved reusable dataframe label-application mechanics out of
  `plugins/transform/_labeling.py` and into `reader.core.labeling`.
- Moved assay-label spec resolution into `reader.workbench.semantics` so
  plugins no longer parse `assay.labels` structures themselves.
- Rewired workbench validation, notebook SFXI template materialization, and
  logic plugins to use `reader.domains.semantics` instead of deep domain-module
  imports.

The resulting placement rule is clearer:

- `domains/semantics.py` is the only shared entrypoint for domain semantic
  resolution from outside a specific domain package
- `workbench/semantics.py` owns workbench-facing assay config resolution
- `core/labeling.py` owns generic dataframe label-application mechanics
- plugins remain adapters over domain/workbench/kernel helpers

Validation for this slice:

- targeted registry + assay-label + SFXI + explain/plot pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, and cytometer
  scaffold experiments

## 2026-03-15: Explicit Contract Bootstrap + Plate Reader Support Dissolution Slice

Implemented the next ontology cleanup slice.

- Replaced contract registration-by-import-side-effect with an explicit built-in
  manifest in `reader.contracts`.
- Moved logic-map assay resolution into `reader.domains.logic.semantics`
  instead of keeping it in generic workbench semantics.
- Deleted the vague `reader.domains.plate_reader.support` bucket.
- Split its former responsibilities into explicit owners:
  - `reader.domains.plate_reader.analysis.timepoints`
  - `reader.domains.plate_reader.ordering`
  - `reader.domains.plate_reader.plots.common`
  - `reader.domains.plate_reader.plots.grouping`

The resulting placement rule is clearer:

- `contracts` owns the shared registry kernel plus an explicit built-in
  bootstrap manifest
- `domains/logic/semantics.py` owns logic-specific assay semantic resolution
- `domains/plate_reader/analysis` owns time selection and derived summary logic
- `domains/plate_reader/ordering.py` owns plate-reader ordering semantics
- `domains/plate_reader/plots/*` owns only plotting helpers and renderers

Validation for this slice:

- targeted contract/bootstrap + logic + plate-reader plot pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, and cytometer
  scaffold experiments

## 2026-03-15: Plate Reader Analysis + Shared Plot Style Slice

Implemented the next ontology cleanup slice.

- Moved fold-change table construction into
  `reader.domains.plate_reader.analysis.fold_change`.
- Moved the `fold_change.v1` dataframe contract into
  `reader.domains.plate_reader.contracts`.
- Deleted the old shared `src/reader/contracts/analysis.py` sink instead of
  leaving one residual analysis bucket behind.
- Moved generic palette/style helpers into `reader.core.plot_style`.
- Rewired workbench palette resolution and plate-reader plotting modules to the
  new shared plotting-infra path.

The resulting placement rule is clearer:

- `domains/plate_reader/analysis` owns plate-reader derived summary logic
- `domains/plate_reader/contracts.py` owns plate-reader output contracts,
  including derived tables such as `fold_change.v1`
- `core/plot_style.py` owns shared plotting style/palette infrastructure
- plugins remain adapters only

Validation for this slice:

- targeted fold-change + plotting-style + engine/setup pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, and cytometer
  scaffold experiments

## 2026-03-15: Plate Reader Domain Slice

Implemented the first real domain migration slice.

- Moved the plate-reader contract into `reader.domains.plate_reader.contracts`.
- Moved the Synergy H1 parser into `reader.domains.plate_reader.io`.
- Moved the microplate support and plotting library into:
  - `reader.domains.plate_reader.support`
  - `reader.domains.plate_reader.plots`
- Rewired plate-reader plugins, workbench palette resolution, tests, and docs to
  the new domain paths.
- Deleted the old tracked `src/reader/lib/microplates/__init__.py` compatibility
  surface instead of leaving a stale alias bucket behind.

The resulting placement rule is clearer:

- `domains/plate_reader/io` owns raw workbook parsing
- `domains/plate_reader/analysis` and `domains/plate_reader/ordering.py` own
  reusable plate-reader semantics
- `domains/plate_reader/plots` owns plotting primitives and figure builders
- plugins remain adapters only

Validation for this slice:

- targeted domain + plugin pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, and sponge-screen experiments
- clean temp-copy smoke runs for plate-reader and SFXI experiments

## 2026-03-15: Logic Domain Slice

Implemented the next domain migration slice.

- Moved `reader.lib.sfxi` into `reader.domains.logic.sfxi`.
- Moved `reader.lib.logic_symmetry` into `reader.domains.logic.logic_symmetry`.
- Moved `reader.lib.crosstalk` into `reader.domains.logic.crosstalk`.
- Moved logic-owned contracts (`sfxi.vec8.*`, `logic_symmetry.v1`, `crosstalk_pairs.v1`) out of the shared `contracts/analysis.py` bucket and into `reader.domains.logic.contracts`.
- Rewired plugins, notebook templates, tests, and docs to the new logic-domain paths.

The resulting placement rule is clearer:

- `domains/logic/sfxi` owns SFXI config parsing, selection, math, reference handling, and vec8 writing
- `domains/logic/logic_symmetry` owns logic-symmetry preparation, metrics, overlays, and rendering
- `domains/logic/crosstalk` owns pairwise crosstalk ranking logic
- plugins remain thin adapters over those domain packages

Validation for this slice:

- targeted logic-domain + plugin pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, and sponge-screen experiments
- clean temp-copy smoke runs for plate-reader and SFXI experiments

## 2026-03-15: Cytometry Domain Slice

Implemented the next domain migration slice.

- Moved the FCS parser into `reader.domains.cytometry.io.fcs`.
- Moved the cytometry-specific dataframe contract into
  `reader.domains.cytometry.contracts`.
- Rewired the cytometry ingest plugin, tests, and docs to the new domain path.
- Deleted the moved legacy `src/reader/io/flow_cytometer.py` and
  `src/reader/contracts/cytometry.py` surfaces instead of leaving alias paths
  behind.

The resulting placement rule is clearer:

- `domains/cytometry/io` owns raw cytometry file parsing
- `domains/cytometry/contracts.py` owns cytometry-domain dataframe contracts
- plugins remain adapters only

Validation for this slice:

- targeted cytometry-domain + ingest pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, sponge-screen, and
  cytometer scaffold experiments

## 2026-03-15: Workbench Resource Cleanup Slice

Implemented the next ontology cleanup slice.

- Moved raw-file auto-discovery helpers into
  `reader.workbench.resources.discovery`.
- Moved the plate-map parser into `reader.domains.plate_reader.io.sample_map`.
- Rewired ingest and sample-map transform adapters to those new ownership
  roots.
- Deleted the moved legacy top-level `src/reader/io/` package surface instead of
  leaving it as a vague namespace.

The resulting placement rule is clearer:

- `workbench/resources` owns generic experiment resource discovery
- `domains/plate_reader/io` owns plate-reader specific resource parsing
- plugins remain adapters only

Validation for this slice:

- targeted workbench-resource + plate-reader io + ingest/transform pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, sponge-screen, and
  cytometer scaffold experiments

## 2026-03-15: Workbench Cutover Slice

Implemented the next destructive re-root slice on the workbench side.

- Moved config loading into `reader.workbench.config`.
- Moved execution/runtime/validation/planning into `reader.workbench.engine`.
- Moved record catalog operations into `reader.workbench.records`.
- Moved notebook scaffolding into `reader.workbench.notebooks`.
- Moved preset expansion into `reader.workbench.presets`.
- Moved experiment discovery, CLI, registry, run context, and config-semantics helpers into `reader.workbench`.
- Moved the workbench ontology/catalog/spec/record model into the workbench
  package.
- Deleted the moved `src/reader/core/{config,engine,notebooks,presets,records}` paths with no compat aliases.

The remaining `src/reader/core/` surface is now intentionally small and generic:

- `errors.py`
- `mpl.py`
- `plot_utils.py`
- `plot_sinks.py`

That is the current contract: workbench lifecycle code lives under `reader.workbench`; `core` is no longer a catch-all product bucket.

Validation for this slice:

- targeted workbench-heavy pytest matrix
- compileall over `src/reader`
- real config validation for template, plate-reader, SFXI, and sponge-screen experiments
- smoke runs remain part of the broader migration matrix

## 2026-03-15: Rooted Architecture Migration Plan

### Plan intent summary

Re-root `reader` around one primary architectural axis:

- `reader.workbench` owns experiment lifecycle, config, planning, execution, records, notebooks, presets, and CLI surfaces.
- `reader.domains` owns assay/data semantics, parsers, contracts, reusable math, plot-prep helpers, and domain-level semantics.
- `reader.plugins` stays as thin adapters that bind workbench config to domain operations and declared outputs.

This migration is intentionally breaking. There will be no backcompat shims for old package paths or mixed semantic surfaces. The goal is to remove the current ambiguity caused by `core/`, `lib/`, `io/`, and parallel semantic systems competing as first-class ontologies.

### Scope

In scope:

- Replace `src/reader/core/` as the main architectural bucket with `src/reader/workbench/` plus a very small shared-infra remainder only if truly generic.
- Rename `src/reader/lib/` to `src/reader/domains/` and reorganize by domain:
  - `domains/plate_reader/`
  - `domains/cytometry/`
  - `domains/logic/`
- Fold `src/reader/io/` into domain packages unless a parser is truly cross-domain.
- Move dataframe contracts out of `core/contracts/` into domain-owned contract packages, leaving only truly generic contracts in one tiny shared contract kernel.
- Split `core/semantics.py` into:
  - workbench-facing config semantics under `workbench`
  - domain-facing assay semantics under the relevant `domains/*`
- Keep plugin IDs and user config behavior stable only until the re-root lands; then delete old package paths entirely.
- Update tests, docs, and experiment smoke runs so the new package ontology is the only supported one.

Out of scope:

- New assay features or new experiment capabilities.
- Rewriting SFXI math, crosstalk math, or plotting math unless required by import relocation.
- A bigger ontology framework or metadata layer.
- Silent migration helpers, import aliases, or compat modules.
- Hand-editing generated `experiments/**/outputs/`.

### Ordered action plan

1. Lock the target architecture contract.
   - Add one short architecture note that defines the only valid placement rules:
     - `workbench` for experiment/workflow lifecycle.
     - `domains` for assay/data meaning.
     - `plugins` for thin adapters.
   - Define explicit “do not place here” rules for any residual shared package.
   - Treat this as the acceptance contract for all subsequent moves.

2. Create the new top-level package skeletons.
   - Create `src/reader/workbench/`.
   - Create `src/reader/domains/plate_reader/`, `src/reader/domains/cytometry/`, and `src/reader/domains/logic/`.
   - Create subpackages inside each domain for:
     - `contracts/`
     - `io/` only where raw parsing exists
     - `analysis/` or `transforms/` only where reusable domain logic exists
     - `plots/` only where plot-prep/render support is domain-specific

3. Move workbench-owned code out of `core/`.
   - Move config schema/load into `workbench/config/`.
   - Move experiment discovery, presets, notebooks, records, and CLI support into `workbench/`.
   - Move execution/planning/runtime/validation into `workbench/engine/`.
   - Move workbench catalog/spec/record ontology under `workbench/model/` or keep it inside `workbench/` with crisp ownership.
   - Leave behind only truly generic infrastructure, and delete `core/` entirely if no such remainder survives.

4. Move domain-owned code out of `lib/`, `io/`, and `core/contracts/`.
  - `lib/microplates/*`, plate-reader transforms/plot support, and plate-reader contracts move into `domains/plate_reader/`.
  - `io/synergy_h1.py` and plate-reader metadata readers move into `domains/plate_reader/io/`.
  - `io/flow_cytometer.py` and cytometer contracts move into `domains/cytometry/`.
  - `lib/sfxi/*`, `lib/logic_symmetry/*`, and logic-oriented contracts move into `domains/logic/`.
  - `plugins/merge/*` moves into `plugins/transform/*` as metadata-enrichment adapters.
  - Remove the top-level `lib/` and `io/` packages once imports are fully migrated.

5. Collapse semantic ownership so each concern has one home.
   - `assay.labels`, `assay.orders`, `assay.collections`, and `assay.logic_maps` should resolve through workbench config materialization, but domain-specific meaning should live in the relevant domain package.
   - Plugin semantic metadata should remain small and operational:
     - category
     - domain
     - family
     - summary
   - Dataframe contract lineage should be owned by domain contract packages, except for a tiny shared contract kernel for truly cross-domain tables.
   - Delete any semantic helper that mixes workbench concerns with domain meaning.

6. Thin all plugins to adapters.
   - Enforce a plugin rule:
     - validate config
     - load/read bound inputs
     - delegate to domain code
     - emit declared outputs
   - Remove `merge` as a first-class plugin category. Metadata joins and table enrichment live under `transform` because they are transforms over tidy data, not a separate execution ontology.
   - Any remaining orchestration shared across a plugin category can stay under `plugins/<category>/_*.py`, but only for adapter-level concerns.
   - If a plugin still contains domain math or domain semantics after this pass, move that logic into `domains/*`.

7. Rebuild imports and internal public boundaries.
   - Rewrite imports package by package instead of using alias layers.
   - Update `__init__.py` exports only for intended public surfaces.
   - Delete old package paths as soon as the new imports are green; do not leave re-export shims.

8. Reorganize tests to match the new ontology.
   - `tests/workbench/` should cover config, discovery, engine, records, notebooks, presets, and CLI integration.
   - `tests/domains/plate_reader/`, `tests/domains/cytometry/`, and `tests/domains/logic/` should own domain contracts, parsers, math, and plot-prep tests.
   - `tests/plugins/` should shrink to adapter contract tests and registry/discovery tests.
   - Delete tests whose only purpose was preserving the old package topology.

9. Update docs to match the rooted architecture.
   - Update user-facing docs only where behavior or commands changed.
   - Add one maintainer-facing architecture document that explains package placement rules with concrete examples.
   - Update plugin extension docs so new contributors know when code belongs in `plugins` versus `domains`.

10. Run the destructive cutover.
   - Remove `core/`, `lib/`, and `io/` once all imports, tests, and docs are migrated.
   - Remove any residual references to old package paths.
   - Confirm that no backcompat import surfaces remain.

### Suggested execution slices

Slice 1: architecture root and workbench migration

- Create `workbench/`.
- Move config, engine, records, notebooks, presets, CLI-adjacent helpers.
- Keep runtime green before touching domains.

Slice 2: plate-reader domain migration

- Migrate plate-reader contracts, parsers, microplate plotting support, and generic plate-reader transforms.
- This should be the tracer-bullet domain because it covers the broadest current experiment family.

Slice 3: logic domain migration

- Move SFXI and logic-symmetry code into `domains/logic/`.
- Keep plugin adapters stable and verify SFXI math outputs are byte-for-byte or dataframe-equivalent unchanged.

Slice 4: cytometry domain migration

- Move FCS parsing and cytometry contracts into `domains/cytometry/`.
- Keep missing-input and missing-optional-dependency behavior explicit.

Slice 5: topology cutover and deletion

- Remove old package trees.
- Rewrite docs and final imports.
- Run full repo validation and experiment smoke suite.

### Validation and risk handling

- Add characterization tests before moving each major package seam.
  - Import-level characterization for plugin registry discovery.
  - Behavior-level characterization for domain math and contract promotion.
  - CLI characterization for `reader ls`, `reader validate`, `reader explain`, `reader run`, `reader plot`, `reader export`, and `reader notebook`.

- Validate every migration slice with:
  - `uv run pytest -q`
  - `uv run ruff check .`
  - `uv run ruff format .`
  - `uv run python -m compileall src/reader`

- Keep real experiment coverage:
  - validate all repo experiment configs
  - smoke-run at least one plate-reader panel experiment
  - smoke-run at least one SFXI experiment
  - keep scaffold experiments allowed to fail only for explicit missing-input reasons

- Add architectural guardrails:
  - no imports from `plugins` into `domains`
  - no domain math inside plugin adapters
  - no domain contracts under `workbench`
  - no residual imports from deleted top-level packages after cutover
  - no residual `merge/*` plugin ids or `plugins/merge/` package after cutover

- Fail fast on ambiguity:
  - duplicate semantic ownership
  - duplicate contract IDs after migration
  - duplicate preset names
  - any remaining scan/fallback mode that is not explicit in CLI/config

### Main risks

- The migration is broad enough that package moves can create churn in tests and docs if done in one blast. Keep the work in bounded slices with green checkpoints.
- Plate-reader code is the widest surface and will expose the most hidden coupling first.
- `core/` currently hosts many cross-cutting concerns, so some utilities that look generic may actually encode workbench or domain semantics and need deliberate re-homing.
- Over-abstracting the new `domains/` structure would recreate the same problem under a new name. Keep domain packages concrete and minimal.

### Decisions recorded

- Keep a tiny shared contract kernel at `reader/contracts/` for truly cross-domain tabular contracts and contract infrastructure only.
  - This kernel should stay minimal: examples include `tidy.v1`, the contract model, lineage helpers, and validation primitives.
  - All domain-specific contracts move into `domains/*/contracts/`.
  - Rationale:
    - Pragmatically, `tidy.v1` is not owned by `plate_reader`, `cytometry`, or `logic`; forcing it into one domain would create a false ontology.
    - Ontologically, this is a shared upper vocabulary for tabular records, not an assay/domain vocabulary.
    - Keeping the shared kernel tiny avoids recreating `core/contracts/` as another junk drawer.

- Remove `merge` as a first-class plugin category and fold it into `transform`.
  - `sample_map` and `sample_metadata` become transform adapters such as `transform/sample_map` and `transform/sample_metadata`.
  - Rationale:
    - These steps are tidy-table enrichments with extra inputs, not a separate execution stage.
    - Keeping `merge` distinct increases category sprawl in registry, docs, tests, and mental model without giving the runtime a different contract shape.
    - Domain/family metadata can still distinguish metadata-enrichment transforms without preserving a redundant top-level category.

### Consequences of the decisions

- New target top-level ontology:
  - `reader/workbench/`
  - `reader/domains/`
  - `reader/plugins/`
  - `reader/contracts/` (tiny shared kernel only)

- Removed top-level architectural concepts:
  - `core/`
  - `lib/`
  - `io/`
  - `merge` as a plugin category

- Migration implications:
  - plugin registry categories become `ingest`, `transform`, `plot`, `export`, `validator`
  - config `uses:` values for merge steps must be rewritten with no compat shim
  - tests and docs must stop treating metadata joins as a separate plugin family at the category level
