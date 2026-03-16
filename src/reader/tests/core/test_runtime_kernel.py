from __future__ import annotations

from pathlib import Path

from reader.runtime import builtin_runtime


def test_builtin_runtime_is_stable_and_composes_builtin_world(tmp_path: Path) -> None:
    runtime = builtin_runtime()

    assert builtin_runtime() is runtime
    assert runtime.plugins.contracts is runtime.contracts
    assert runtime.assets.resolve("transform/sample_map", kind="plugin").plugin_id == "transform/sample_map"

    store = runtime.record_store(tmp_path / "outputs", create=False)

    assert store.contracts is runtime.contracts


def test_runtime_composition_only_lives_in_runtime_package() -> None:
    code_root = Path(__file__).resolve().parents[2]
    allowed_builtin_contract_catalog = {
        code_root / "contracts" / "builtins" / "__init__.py",
        code_root / "runtime" / "builtin.py",
    }
    allowed_load_plugin_catalog = {
        code_root / "runtime" / "builtin.py",
        code_root / "workbench" / "registry.py",
    }

    builtin_violations: list[str] = []
    plugin_violations: list[str] = []
    for path in code_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "builtin_contract_catalog(" in text and path not in allowed_builtin_contract_catalog:
            builtin_violations.append(str(path.relative_to(code_root)))
        if "load_plugin_catalog(" in text and path not in allowed_load_plugin_catalog:
            plugin_violations.append(str(path.relative_to(code_root)))

    assert not builtin_violations, f"builtin_contract_catalog() escaped runtime root: {builtin_violations}"
    assert not plugin_violations, f"load_plugin_catalog() escaped runtime root: {plugin_violations}"
