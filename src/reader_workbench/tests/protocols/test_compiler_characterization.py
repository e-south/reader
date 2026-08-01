from __future__ import annotations

import inspect
import json
from dataclasses import fields, is_dataclass
from hashlib import sha256
from pathlib import Path

import pytest

from reader_workbench.protocols import ProtocolBinding
from reader_workbench.protocols.compiler import (
    compile_cytometry_flow_panel,
    compile_generic_protocol,
    compile_logic_four_state_vector_collection,
    compile_logic_four_state_vector_screen,
    compile_plate_reader_dual_reporter_screen,
    compile_plate_reader_response_window,
    compile_plate_reader_single_reporter_screen,
)
from reader_workbench.runtime import builtin_runtime
from reader_workbench.tests.support.configs import cytometry_test_gating_policy

COMPILED_PLAN_FIXTURES = {
    "cytometry/flow_panel": {
        "pipeline": ("ingest_cytometer", "merge_metadata", "cytometry_gating"),
        "plots": ("gating_diagnostic",),
        "exports": ("gate_definition_table", "sample_stats_table", "group_stats_table", "qc_table"),
        "sha256": "e7f4b07b7be7f5a501fa2f490b4f5b49428c98c7847a170b5596cb72f5456b32",  # pragma: allowlist secret
    },
    "workbench/generic": {
        "pipeline": (),
        "plots": (),
        "exports": (),
        "sha256": "44224f270075c123c58bca16d9802acb533856b2c9eb3ed62cb177a435934398",  # pragma: allowlist secret
    },
    "logic/four_state_vector_collection": {
        "pipeline": ("four_state_vector_collection",),
        "plots": ("four_state_vector_heatmap",),
        "exports": ("vector_table",),
        "sha256": "a16bdb145fec593f20aa7adfcfd291b469c46240ba17d24e439c8aa3bb62d124",  # pragma: allowlist secret
    },
    "plate_reader/response_window": {
        "pipeline": ("response_window",),
        "plots": ("response_window_summary",),
        "exports": ("designs_table", "events_table"),
        "sha256": "8c05d7c289b87964300ba32eac10cb1eb4cbebe8d182d66338560a32436812fa",  # pragma: allowlist secret
    },
    "logic/four_state_vector_screen": {
        "pipeline": (
            "ingest",
            "merge_map",
            "labels",
            "blank",
            "overflow",
            "ratio_yfp_cfp",
            "ratio_cfp_od600",
            "ratio_yfp_od600",
            "promote_to_tidy_plus_map",
            "four_state_vector",
        ),
        "plots": ("raw_kinetics",),
        "exports": ("logic_summary_workbook",),
        "sha256": "189798460052f35ea53e7f9cc81c30ad911d4d4cb1333188e5699a375d0f8ac0",  # pragma: allowlist secret
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
        ),
        "plots": ("raw_kinetics", "value_distributions"),
        "exports": (),
        "sha256": "6926b5da669fa492527d836ea8235158780df833b509a37ea5d816a962bd12d5",  # pragma: allowlist secret
    },
    "plate_reader/single_reporter_screen": {
        "pipeline": (
            "ingest",
            "merge_map",
            "labels",
            "blank",
            "overflow",
            "ratio_reporter_normalizer",
            "sample_measurements",
        ),
        "plots": ("raw_kinetics", "value_distributions"),
        "exports": (),
        "sha256": "8b029f4232e05d4b264eff305c3256cba05784499aeba5e378b7b44db4e737fb",  # pragma: allowlist secret
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
    inputs = {"gating": cytometry_test_gating_policy()} if protocol_id == "cytometry/flow_panel" else {}
    plan = runtime.bind_protocol(ProtocolBinding(id=protocol_id, inputs=inputs)).compile()
    expected = COMPILED_PLAN_FIXTURES[protocol_id]

    assert tuple(step.id for step in plan.pipeline) == expected["pipeline"]
    assert tuple(step.id for step in plan.plots) == expected["plots"]
    assert tuple(step.id for step in plan.exports) == expected["exports"]
    assert _plan_digest(plan) == expected["sha256"]


@pytest.mark.parametrize(
    "compiler",
    [
        compile_cytometry_flow_panel,
        compile_generic_protocol,
        compile_logic_four_state_vector_screen,
        compile_logic_four_state_vector_collection,
        compile_plate_reader_dual_reporter_screen,
        compile_plate_reader_response_window,
        compile_plate_reader_single_reporter_screen,
    ],
)
def test_public_compiler_signature_remains_one_protocol_argument(compiler) -> None:
    parameters = tuple(inspect.signature(compiler).parameters.values())

    assert len(parameters) == 1
    assert parameters[0].name == "protocol"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
