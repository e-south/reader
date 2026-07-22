from __future__ import annotations

from functools import cache

from .model import DataClassSpec, DopRegistry, ReadySpec

DOP_SCHEMA = "reader.dop/v1"

BUILTIN_DATA_CLASSES: tuple[DataClassSpec, ...] = (
    DataClassSpec(
        id="plate_reader_screen",
        label="Plate-reader screen",
        summary="Well-level plate-reader assay data with explicit channel, treatment, control, and plate/well semantics.",
        decision_order=10,
        protocol_candidates=(
            "plate_reader/retron_sponge_screen",
            "plate_reader/dual_reporter_screen",
            "plate_reader/single_reporter_screen",
        ),
        minimum_capture=(
            "raw plate-reader workbook or export",
            "sample map with measured well coverage",
            "channel labels and denominator/ratio meaning",
            "treatment and control semantics",
            "plate, well, replicate, and design identifiers",
        ),
        stop_conditions=(
            "well coordinates or sample positions conflict",
            "treatment or control meaning is incomplete",
            "channel labels drift from the selected protocol",
            "nearest protocol would silently change control semantics",
        ),
        transfer_rules=(
            "stage the original workbook or export under inputs/",
            "bind metadata files through resources",
            "regenerate plots, exports, and records from source inputs",
        ),
        verification=(
            "config schema and protocol binding validate",
            "declared raw files and metadata resources exist",
            "records catalog captures generated dataframe evidence",
        ),
    ),
    DataClassSpec(
        id="flow_cytometry_panel",
        label="Flow-cytometry panel",
        summary="Cytometry panel data with explicit FCS roots, channel naming, and sample metadata.",
        decision_order=20,
        protocol_candidates=("cytometry/flow_panel",),
        minimum_capture=(
            "raw FCS roots or files",
            "channel naming field",
            "sample metadata",
            "required panel metadata columns",
        ),
        stop_conditions=(
            "FCS root or file mapping is ambiguous",
            "channel naming field is unknown",
            "sample metadata cannot be joined to events",
        ),
        transfer_rules=(
            "stage raw FCS material under inputs/",
            "bind FCS roots and metadata through resources",
            "keep generated review notebooks under outputs/notebooks/",
        ),
        verification=(
            "FCS resources resolve",
            "configured metadata columns are present",
            "records catalog captures panel dataframe evidence",
        ),
    ),
    DataClassSpec(
        id="logic_sfxi_analysis",
        label="Logic/SFXI analysis",
        summary="Logic-response or SFXI-style assay data with response/intensity channels and ordered states.",
        decision_order=30,
        protocol_candidates=("logic/sfxi_screen",),
        minimum_capture=(
            "raw assay files",
            "metadata map",
            "response and intensity channel choices",
            "reference design",
            "ordered 00/10/01/11 states",
        ),
        stop_conditions=(
            "reference design cannot be reconstructed",
            "ordered state values are missing or contradictory",
            "response or intensity channel choices are ambiguous",
        ),
        transfer_rules=(
            "stage raw files under inputs/",
            "encode ordered state spaces in annotations or metadata resources",
            "regenerate SFXI summaries from source inputs",
        ),
        verification=(
            "logic reference config validates",
            "ordered state-space annotations are present",
            "records catalog captures vec8 summary evidence",
        ),
    ),
    DataClassSpec(
        id="aggregate_review_workspace",
        label="Aggregate/review workspace",
        summary="Review material assembled from prior reader records, plots, exports, or hand-authored notes.",
        decision_order=40,
        protocol_candidates=("workbench/generic",),
        minimum_capture=(
            "source experiment ids",
            "record, plot, or export paths",
            "review purpose",
            "expected notebook template",
        ),
        stop_conditions=(
            "source experiment ids are unknown",
            "review material mixes generated outputs without source records",
            "notebook purpose is unclear",
        ),
        transfer_rules=(
            "reference source experiments instead of copying generated outputs",
            "keep hand-authored notes under notebooks/",
            "keep generated scaffolds under outputs/notebooks/",
        ),
        verification=(
            "source records or exports are identifiable",
            "review notebook template is explicit",
            "unresolved assumptions are visible in the handoff",
        ),
    ),
    DataClassSpec(
        id="unsupported_long_tail_assay",
        label="Unsupported long-tail assay",
        summary="Assay data that does not yet fit an existing executable protocol contract.",
        decision_order=50,
        protocol_candidates=("workbench/generic",),
        minimum_capture=(
            "raw source path",
            "intended analysis",
            "required metadata",
            "missing protocol decision",
            "owner for follow-up",
        ),
        stop_conditions=(
            "nearest protocol would change assay meaning",
            "required metadata is unknown",
            "execution contract is still being discovered",
        ),
        transfer_rules=(
            "keep the experiment draft or template until semantics are clear",
            "stage raw files without pretending they are runnable",
            "add a protocol only after the metadata and execution contract stabilize",
        ),
        verification=(
            "draft/template config shape validates with --no-files",
            "missing protocol or metadata contract is documented",
            "no generated outputs are treated as source material",
        ),
    ),
)

BUILTIN_READY_SPECS: tuple[ReadySpec, ...] = (
    ReadySpec(
        id="classified",
        label="Classified",
        summary="Dataset has a selected DOP data class and protocol candidate set.",
        required_evidence=(
            "DOP data class id",
            "candidate reader protocol ids",
            "reason the selected class fits the dataset",
        ),
        commands=("uv run reader dop classes",),
    ),
    ReadySpec(
        id="metadata_ready",
        label="Metadata ready",
        summary="Required semantics for the selected data class have been captured before execution.",
        required_evidence=(
            "dataset identity",
            "raw provenance",
            "assay semantics",
            "sample map",
            "control semantics",
            "canonical labels",
            "requested outputs when they differ from protocol defaults",
        ),
        commands=("uv run reader protocols <protocol-id> --example-config",),
    ),
    ReadySpec(
        id="staged",
        label="Staged",
        summary="Raw files, metadata resources, and hand-authored notes live in the standard experiment layout.",
        required_evidence=(
            "raw files under inputs/",
            "resources entries for consumed files or directories",
            "hand-authored notes under notebooks/ when present",
            "no copied generated outputs used as source material",
        ),
        commands=("uv run reader validate <config|dir|index> --no-files --format json",),
    ),
    ReadySpec(
        id="preflight_ok",
        label="Preflight OK",
        summary="Schema, protocol binding, declared files, and dependencies pass reader validation.",
        required_evidence=(
            "config schema is reader/v8",
            "protocol binding resolves",
            "declared files and resources exist",
            "runtime dependencies are available",
        ),
        accepted_readiness_states=("runnable", "uncataloged_outputs_present", "records_ready"),
        commands=("uv run reader validate <config|dir|index> --format json",),
    ),
    ReadySpec(
        id="runnable",
        label="Runnable",
        summary="The experiment is active and can run from authored source inputs.",
        required_evidence=(
            "reader readiness state allows run",
            "run capability is true",
            "next command is explicit",
        ),
        accepted_readiness_states=("runnable", "uncataloged_outputs_present", "records_ready"),
        required_capabilities=("run",),
        commands=("uv run reader run <config|dir|index> --dry-run --format json",),
    ),
    ReadySpec(
        id="records_ready",
        label="Records ready",
        summary="Generated dataframe and file-bundle evidence is present in the records catalog.",
        required_evidence=(
            "records catalog exists",
            "dataframe artifacts include contract ids",
            "records include input and config digests",
        ),
        accepted_readiness_states=("records_ready",),
        required_capabilities=("records",),
        commands=("uv run reader records <config|dir|index>",),
    ),
    ReadySpec(
        id="review_ready",
        label="Review ready",
        summary="Records exist and review surfaces such as plots or notebooks can be inspected deliberately.",
        required_evidence=(
            "records catalog exists",
            "selected plots or notebook scaffold are protocol-compatible",
            "unresolved metadata assumptions remain visible in the handoff",
        ),
        accepted_readiness_states=("records_ready",),
        required_capabilities=("records", "plot", "notebook_scaffold"),
        commands=(
            "uv run reader plot <config|dir|index> --list",
            "uv run reader notebook <config|dir|index> --mode none",
        ),
    ),
)


@cache
def builtin_dop_registry() -> DopRegistry:
    return DopRegistry(data_classes=BUILTIN_DATA_CLASSES, ready_specs=BUILTIN_READY_SPECS)
