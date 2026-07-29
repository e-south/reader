from __future__ import annotations

import inspect
import json
from dataclasses import fields, is_dataclass
from hashlib import sha256
from pathlib import Path

import pytest

from reader.protocols import ProtocolBinding
from reader.protocols.compiler import (
    compile_cytometry_flow_panel,
    compile_generic_protocol,
    compile_logic_sfxi_screen,
    compile_plate_reader_dual_reporter_screen,
    compile_plate_reader_single_reporter_screen,
)
from reader.runtime import builtin_runtime

COMPILED_PLAN_FIXTURES = {
    "cytometry/flow_panel": {
        "pipeline": ("ingest_cytometer", "merge_metadata"),
        "plots": (),
        "exports": (),
        "notebooks": ("notebook/cytometry",),
        "sha256": "ac959669362453c4d69d38ae0c2f93c4590cb8c964f84b3f716127d49d70aea8",  # pragma: allowlist secret
    },
    "workbench/generic": {
        "pipeline": (),
        "plots": (),
        "exports": (),
        "notebooks": ("notebook/basic",),
        "sha256": "5cbd7074a19e743fcda5da3507e4d51e30e136095bc1e36b10159ad86db5d422",  # pragma: allowlist secret
    },
    "logic/sfxi_screen": {
        "pipeline": (
            "ingest",
            "merge_map",
            "labels",
            "blank",
            "overflow",
            "ratio_yfp_cfp",
            "ratio_cfp_od600",
            "ratio_yfp_od600",
            "fold_change__yfp_over_cfp",
            "promote_to_tidy_plus_map",
            "sfxi_vec8",
        ),
        "plots": ("raw_kinetics", "endpoint_by_condition", "endpoint_by_design", "intensity_overview"),
        "exports": ("logic_summary_workbook",),
        "notebooks": ("notebook/sfxi_eda",),
        "sha256": "0c8c8f3438890e4e472cf798a9da5856d88851b575b6ed27a80e7648fafd9c29",  # pragma: allowlist secret
    },
    "plate_reader/dual_reporter_screen": {
        "pipeline": (
            "ingest",
            "merge_map",
            "labels",
            "blank",
            "overflow",
            "ratio_yfp_cfp",
            "ratio_cfp_od600",
            "ratio_yfp_od600",
            "fold_change__yfp_over_cfp",
        ),
        "plots": ("raw_kinetics", "endpoint_by_condition", "endpoint_by_design", "intensity_overview"),
        "exports": (),
        "notebooks": ("notebook/eda",),
        "sha256": "fd646d990e7f257980a9d2731d2e91dea1afd55a673a515f9c62a8ccbace0230",  # pragma: allowlist secret
    },
    "plate_reader/single_reporter_screen": {
        "pipeline": (
            "ingest",
            "merge_map",
            "labels",
            "blank",
            "overflow",
            "ratio_reporter_normalizer",
            "fold_change__single_reporter",
        ),
        "plots": ("raw_kinetics", "endpoint_by_condition", "endpoint_by_design", "intensity_overview"),
        "exports": (),
        "notebooks": ("notebook/eda",),
        "sha256": "655e4687c1faa0d0f4c4f399f8bf20101c385bf55f822479ec89c7451a425b23",  # pragma: allowlist secret
    },
}


def _normalized(value):
    if is_dataclass(value):
        return {field.name: _normalized(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _normalized(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_normalized(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _plan_digest(plan) -> str:
    payload = json.dumps(
        _normalized(plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return sha256(payload).hexdigest()


@pytest.mark.parametrize("protocol_id", sorted(COMPILED_PLAN_FIXTURES))
def test_builtin_compiled_plan_matches_characterization_fixture(protocol_id: str) -> None:
    runtime = builtin_runtime()
    plan = runtime.bind_protocol(ProtocolBinding(id=protocol_id)).compile()
    expected = COMPILED_PLAN_FIXTURES[protocol_id]

    assert tuple(step.id for step in plan.pipeline) == expected["pipeline"]
    assert tuple(step.id for step in plan.plots) == expected["plots"]
    assert tuple(step.id for step in plan.exports) == expected["exports"]
    assert tuple(step.template for step in plan.notebooks) == expected["notebooks"]
    assert _plan_digest(plan) == expected["sha256"]


@pytest.mark.parametrize(
    "compiler",
    [
        compile_cytometry_flow_panel,
        compile_generic_protocol,
        compile_logic_sfxi_screen,
        compile_plate_reader_dual_reporter_screen,
        compile_plate_reader_single_reporter_screen,
    ],
)
def test_public_compiler_signature_remains_one_protocol_argument(compiler) -> None:
    parameters = tuple(inspect.signature(compiler).parameters.values())

    assert len(parameters) == 1
    assert parameters[0].name == "protocol"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
