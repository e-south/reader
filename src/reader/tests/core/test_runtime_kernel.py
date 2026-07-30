from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import reader
import reader.workbench
from reader.protocols import ProtocolBinding
from reader.runtime import builtin_runtime


def test_builtin_runtime_is_stable_and_composes_builtin_world(tmp_path: Path) -> None:
    runtime = builtin_runtime()

    assert builtin_runtime() is runtime
    assert runtime.plugins.contracts is runtime.contracts
    assert runtime.bind_protocol(ProtocolBinding(id="workbench/generic")).id == "workbench/generic"
    assert not hasattr(runtime, "assets")

    store = runtime.record_store(tmp_path / "outputs", create=False)

    assert store.contracts is runtime.contracts


def test_bound_protocol_applies_executable_defaults() -> None:
    runtime = builtin_runtime()
    protocol = runtime.bind_protocol(
        ProtocolBinding(
            id="logic/sfxi_screen",
            inputs={
                "target_time_h": 10.0,
                "state_map_ref": "induction_logic",
                "reference": {"design_id": "CUSTOM"},
            },
        )
    )

    cfg = protocol.effective_plugin_config(
        plugin_id="transform/sfxi",
        step_with={"reference": {"observation_stat": "median"}},
    )

    assert cfg["time_mode"] == "nearest"
    assert cfg["target_time_h"] == 10.0
    assert cfg["state_map_ref"] == "induction_logic"
    assert cfg["reference"] == {"design_id": "CUSTOM", "observation_stat": "median"}


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


def test_workbench_does_not_import_concrete_plugin_implementations() -> None:
    workbench_root = Path(__file__).resolve().parents[2] / "workbench"
    violations: list[str] = []

    for path in workbench_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = (node.module or "",)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("reader.plugins."):
                    violations.append(str(path.relative_to(workbench_root.parent)))
                continue
            else:
                continue
            if any(name == "reader.plugins" or name.startswith("reader.plugins.") for name in imported):
                violations.append(str(path.relative_to(workbench_root.parent)))

    assert not violations, "workbench imports concrete reader.plugins modules: " + ", ".join(sorted(violations))


def test_plot_registry_import_does_not_eager_load_snapshot_heatmap_domain_module() -> None:
    sys.modules.pop("reader.plugins.plot.snapshot_heatmap", None)
    sys.modules.pop("reader.domains.plate_reader.plots.snapshot_heatmap", None)

    importlib.import_module("reader.plugins.plot.snapshot_heatmap")

    assert "reader.domains.plate_reader.plots.snapshot_heatmap" not in sys.modules


def test_builtin_runtime_discovery_does_not_eager_load_matplotlib_pyplot() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from reader.runtime import builtin_runtime; "
                "builtin_runtime(); "
                "raise SystemExit(1 if 'matplotlib.pyplot' in sys.modules else 0)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def test_plotting_mpl_import_does_not_eager_load_plotting_style_module() -> None:
    sys.modules.pop("reader.plotting", None)
    sys.modules.pop("reader.plotting.mpl", None)
    sys.modules.pop("reader.plotting.style", None)

    importlib.import_module("reader.plotting.mpl")

    assert "reader.plotting.style" not in sys.modules


def test_runtime_import_does_not_eager_load_builtin_bootstrap() -> None:
    module_names = (
        "reader.runtime",
        "reader.runtime.builtin",
        "reader.workbench.records.store",
        "reader.workbench.registry",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    had_runtime_attr = hasattr(reader, "runtime")
    saved_runtime_attr = getattr(reader, "runtime", None)
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        if had_runtime_attr:
            delattr(reader, "runtime")

        importlib.import_module("reader.runtime")

        assert "reader.runtime.builtin" not in sys.modules
        assert "reader.workbench.records.store" not in sys.modules
        assert "reader.workbench.registry" not in sys.modules
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        if had_runtime_attr:
            reader.runtime = saved_runtime_attr
        elif hasattr(reader, "runtime"):
            delattr(reader, "runtime")


def test_cli_import_does_not_eager_load_protocol_or_notebook_bootstrap() -> None:
    module_names = (
        "reader.workbench.cli",
        "reader.protocols.model",
        "reader.protocols.builtins",
        "reader.workbench.notebooks",
        "reader.workbench.notebooks.scaffold",
        "yaml",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    saved_attrs = {
        "cli": getattr(reader.workbench, "cli", None),
        "notebooks": getattr(reader.workbench, "notebooks", None),
    }
    had_attrs = {name: hasattr(reader.workbench, name) for name in saved_attrs}
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, present in had_attrs.items():
            if present:
                delattr(reader.workbench, name)

        importlib.import_module("reader.workbench.cli")

        assert "reader.protocols.model" not in sys.modules
        assert "reader.protocols.builtins" not in sys.modules
        assert "reader.workbench.notebooks" not in sys.modules
        assert "reader.workbench.notebooks.scaffold" not in sys.modules
        assert "yaml" not in sys.modules
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        for name, present in had_attrs.items():
            if present:
                setattr(reader.workbench, name, saved_attrs[name])
            elif hasattr(reader.workbench, name):
                delattr(reader.workbench, name)
