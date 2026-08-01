from __future__ import annotations

from pathlib import Path

import pytest

from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.errors import ConfigError
from reader_workbench.plugins.catalog import builtin_plugin_catalog
from reader_workbench.protocols import ProtocolBinding, builtin_protocol_catalog
from reader_workbench.runtime import builtin_runtime
from reader_workbench.workbench.cli.automation import request_from_argv
from reader_workbench.workbench.inspection.protocols import protocol_descriptor_payload

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_four_state_event_window_protocol_owns_canonical_runtime_ids() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="plate_reader/four_state_event_window"))

    compiled = protocol.compile()

    assert tuple(step.id for step in compiled.pipeline) == ("four_state_event_window",)
    assert tuple(step.plugin for step in compiled.pipeline) == ("transform/four_state_event_window",)
    assert tuple(step.id for step in compiled.plots) == ("four_state_event_window_summary",)
    assert tuple(step.plugin for step in compiled.plots) == ("plot/four_state_event_window_summary",)
    assert compiled.pipeline[0].writes["designs"].record_id == "four_state_event_window/designs"


def test_retired_event_window_protocol_does_not_resolve() -> None:
    retired_protocol = "plate_reader/" + "response_" + "window"

    with pytest.raises(ConfigError, match="Unknown protocol"):
        builtin_protocol_catalog().resolve(retired_protocol)


def test_four_state_event_window_protocol_is_json_inspectable() -> None:
    runtime = builtin_runtime()
    descriptor = runtime.protocols.resolve("plate_reader/four_state_event_window")

    payload = protocol_descriptor_payload(descriptor, runtime=runtime)

    compiled = payload["implementation"]["compiled"]
    assert compiled["pipeline"][0]["id"] == "four_state_event_window"
    assert compiled["pipeline"][0]["reads"][0]["ref"] == {"record_resources": []}


def test_four_state_event_window_plugins_replace_retired_plugin_ids() -> None:
    catalog = builtin_plugin_catalog()
    current = {
        "transform/four_state_event_window",
        "plot/four_state_event_window_summary",
        "plot/four_state_event_window_diagnostic",
    }
    retired = {
        "transform/" + "response_" + "window",
        "plot/" + "response_" + "window_summary",
        "plot/" + "response_" + "window_diagnostic",
    }

    assert current <= {descriptor.name for descriptor in catalog.all()}
    for plugin_id in retired:
        with pytest.raises(ConfigError, match="Unknown plugin"):
            catalog.resolve(plugin_id)


def test_four_state_event_window_contracts_replace_retired_contract_ids() -> None:
    catalog_ids = set(builtin_contract_catalog().ids())
    current = {
        "plate_reader.four_state_event_window.wells.v3",
        "plate_reader.four_state_event_window.designs.v4",
        "plate_reader.four_state_event_window.descriptive_resampling_draws.v3",
        "plate_reader.four_state_event_window.traces.v3",
        "plate_reader.four_state_event_window.events.v2",
    }
    retired_prefix = "plate_reader." + "response_" + "window."

    assert current <= catalog_ids
    assert not any(contract_id.startswith(retired_prefix) for contract_id in catalog_ids)


def test_cli_automation_has_no_retired_event_window_subcommand_special_case() -> None:
    retired_command = "response-" + "window"

    request = request_from_argv([retired_command, "build", "--format", "json"])

    assert request.command == retired_command


def test_retired_event_window_modules_are_not_kept_as_import_shims() -> None:
    retired_module = "response_" + "window"

    assert not (PACKAGE_ROOT / "contracts" / "builtins" / f"{retired_module}.py").exists()
    assert not (PACKAGE_ROOT / "plugins" / "transform" / f"{retired_module}.py").exists()
    assert not (PACKAGE_ROOT / "plugins" / "plot" / f"{retired_module}_summary.py").exists()
    assert not (PACKAGE_ROOT / "plugins" / "plot" / f"{retired_module}_diagnostic.py").exists()
    assert not (PACKAGE_ROOT / "domains" / "plate_reader" / "analysis" / retired_module).exists()
    assert not (PACKAGE_ROOT / "domains" / "plate_reader" / "plots" / retired_module).exists()
