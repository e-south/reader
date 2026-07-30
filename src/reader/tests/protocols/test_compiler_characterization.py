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
    compile_logic_sfxi_vec8_collection,
    compile_plate_reader_dual_reporter_screen,
    compile_plate_reader_response_window,
    compile_plate_reader_single_reporter_screen,
)
from reader.runtime import builtin_runtime
from reader.tests.support.configs import cytometry_test_gating_policy

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
        "sha256": "52c1d07c2c98b97b45bae1be25c9feb15324c8bbcbd50a58e3acde727d604e7d",  # pragma: allowlist secret
    },
    "logic/sfxi_vec8_collection": {
        "pipeline": ("collect_vec8",),
        "plots": ("vec8_collection_heatmap",),
        "exports": ("vec8_table",),
        "sha256": "a36f3c77035b4109821176f88a2b748d65a9e4cc3bc12bd6d4bf9073c5fd5791",  # pragma: allowlist secret
    },
    "plate_reader/response_window": {
        "pipeline": ("response_window",),
        "plots": ("response_window_summary",),
        "exports": ("designs_table", "events_table"),
        "sha256": "8c05d7c289b87964300ba32eac10cb1eb4cbebe8d182d66338560a32436812fa",  # pragma: allowlist secret
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
            "promote_to_tidy_plus_map",
            "sfxi_vec8",
        ),
        "plots": ("raw_kinetics",),
        "exports": ("logic_summary_workbook",),
        "sha256": "5f777e90d9e10d049cf857bc25fef3b0c6bcdd248eedefed415e53f6ce14802e",  # pragma: allowlist secret
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
        compile_logic_sfxi_screen,
        compile_logic_sfxi_vec8_collection,
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
