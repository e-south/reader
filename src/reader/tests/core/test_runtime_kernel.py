from __future__ import annotations

from pathlib import Path

from reader.protocols import ProtocolBinding
from reader.runtime import builtin_runtime


def test_builtin_runtime_is_stable_and_composes_builtin_world(tmp_path: Path) -> None:
    runtime = builtin_runtime()

    assert builtin_runtime() is runtime
    assert runtime.plugins.contracts is runtime.contracts
    assert runtime.bind_protocol(ProtocolBinding(id="workbench/generic")).id == "workbench/generic"
    assert runtime.assets.resolve("transform/sample_map", kind="plugin").plugin_id == "transform/sample_map"

    store = runtime.record_store(tmp_path / "outputs", create=False)

    assert store.contracts is runtime.contracts


def test_bound_protocol_applies_executable_defaults() -> None:
    runtime = builtin_runtime()
    protocol = runtime.bind_protocol(
        ProtocolBinding(
            id="logic/sfxi_screen",
            parameters={
                "target_time_h": 10.0,
                "logic_map_ref": "induction_logic",
                "reference": {"design_id": "CUSTOM"},
            },
        )
    )

    cfg = protocol.effective_plugin_config(plugin_id="transform/sfxi", step_with={"reference": {"stat": "median"}})

    assert cfg["time_mode"] == "nearest"
    assert cfg["target_time_h"] == 10.0
    assert cfg["logic_map_ref"] == "induction_logic"
    assert cfg["reference"] == {"design_id": "CUSTOM", "stat": "median"}
    assert protocol.default_notebook_template == "notebook/sfxi_eda"


def test_runtime_composition_only_lives_in_runtime_package() -> None:
    code_root = Path(__file__).resolve().parents[2]
    allowed_builtin_contract_catalog = {
        code_root / "contracts" / "builtins" / "__init__.py",
        code_root / "runtime" / "builtin.py",
    }
    allowed_builtin_protocol_catalog = {
        code_root / "protocols" / "builtins.py",
        code_root / "runtime" / "builtin.py",
    }
    allowed_load_plugin_catalog = {
        code_root / "runtime" / "builtin.py",
        code_root / "workbench" / "registry.py",
    }

    builtin_violations: list[str] = []
    protocol_violations: list[str] = []
    plugin_violations: list[str] = []
    for path in code_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "builtin_contract_catalog(" in text and path not in allowed_builtin_contract_catalog:
            builtin_violations.append(str(path.relative_to(code_root)))
        if "builtin_protocol_catalog(" in text and path not in allowed_builtin_protocol_catalog:
            protocol_violations.append(str(path.relative_to(code_root)))
        if "load_plugin_catalog(" in text and path not in allowed_load_plugin_catalog:
            plugin_violations.append(str(path.relative_to(code_root)))

    assert not builtin_violations, f"builtin_contract_catalog() escaped runtime root: {builtin_violations}"
    assert not protocol_violations, f"builtin_protocol_catalog() escaped runtime root: {protocol_violations}"
    assert not plugin_violations, f"load_plugin_catalog() escaped runtime root: {plugin_violations}"
