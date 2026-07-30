from __future__ import annotations

import ast
from pathlib import Path

READER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
API_ROOT = READER_ROOT / "api"
DOMAIN_ROOT = READER_ROOT / "domains"
WORKBENCH_NOTEBOOK_ROOT = READER_ROOT / "workbench" / "notebooks"
WORKBENCH_NOTEBOOK_COMPONENT_ROOT = WORKBENCH_NOTEBOOK_ROOT / "components"
CANONICAL_NOTEBOOK_SOURCE = WORKBENCH_NOTEBOOK_ROOT / "eda.marimo.py.txt"
FORBIDDEN_DOMAIN_DEPENDENCIES = (
    "reader_workbench.api",
    "reader_workbench.maintenance",
    "reader_workbench.plugins",
    "reader_workbench.protocols",
    "reader_workbench.runtime",
    "reader_workbench.workbench",
)
FORBIDDEN_WORKBENCH_NOTEBOOK_ANALYSIS_DEPENDENCIES = (
    "altair",
    "matplotlib",
    "numpy",
    "scipy",
    "seaborn",
    "sklearn",
    "statsmodels",
)
FORBIDDEN_API_DOMAIN_TERMS = (
    "crosstalk",
    "cytometry",
    "logic_symmetry",
    "plate_reader",
    "response_window",
    "sfxi",
    "vec8",
)


def test_domains_do_not_depend_on_orchestration_packages() -> None:
    violations: list[str] = []
    for path in sorted(DOMAIN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = _imported_modules(node)
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_DOMAIN_DEPENDENCIES):
                    violations.append(f"{path.relative_to(READER_ROOT)}:{node.lineno} imports {name}")

    assert violations == [], "Domain packages must not depend on orchestration:\n" + "\n".join(violations)


def test_public_api_does_not_own_domain_specific_policy_or_names() -> None:
    violations: list[str] = []
    for path in sorted(API_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8").casefold()
        for term in FORBIDDEN_API_DOMAIN_TERMS:
            if term in source:
                violations.append(f"{path.relative_to(READER_ROOT)} contains {term!r}")

    assert violations == [], "reader_workbench.api must remain domain-neutral:\n" + "\n".join(violations)


def test_shared_notebook_components_do_not_branch_on_domain_names() -> None:
    violations: list[str] = []
    for path in sorted(WORKBENCH_NOTEBOOK_COMPONENT_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8").casefold()
        for term in FORBIDDEN_API_DOMAIN_TERMS:
            if term in source:
                violations.append(f"{path.relative_to(READER_ROOT)} contains {term!r}")

    assert violations == [], "Shared notebook components must remain domain-neutral:\n" + "\n".join(violations)


def test_domain_operations_do_not_accept_runtime_objects() -> None:
    violations: list[str] = []
    runtime_parameter_names = {"ctx", "runtime", "store"}
    for path in sorted(DOMAIN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            accepted = sorted(parameter.arg for parameter in parameters if parameter.arg in runtime_parameter_names)
            if accepted:
                violations.append(
                    f"{path.relative_to(READER_ROOT)}:{node.lineno} accepts runtime parameter(s) {accepted}"
                )

    assert violations == [], (
        "Domain operations accept explicit data and parameters, not runtime objects:\n" + "\n".join(violations)
    )


def test_repository_has_no_parallel_work_product_roots() -> None:
    stray = [name for name in ("outputs", "tmp") if (REPO_ROOT / name).exists()]

    assert stray == [], (
        "Reader work products belong to experiments/<experiment>/outputs; "
        f"local scratch belongs in .tmp. Stray repository roots: {stray}"
    )


def test_runtime_plugin_namespace_is_a_packaged_boundary() -> None:
    marker = READER_ROOT / "plugins" / "__init__.py"

    assert marker.is_file(), (
        "reader_workbench.plugins must be a regular package so wheel builds include built-in plugins"
    )


def test_pypi_distribution_uses_only_the_reader_workbench_import_namespace() -> None:
    assert READER_ROOT.name == "reader_workbench"
    assert not (REPO_ROOT / "src" / "reader").exists()


def test_repo_local_skills_use_the_codex_discovery_root() -> None:
    skills_root = REPO_ROOT / ".agents" / "skills"

    assert skills_root.is_dir(), "repository skills must be discoverable under .agents/skills"
    assert not (REPO_ROOT / "skills").exists(), "do not maintain a parallel, undiscoverable skills root"


def test_domain_capabilities_do_not_create_parallel_public_lifecycles() -> None:
    assert not (READER_ROOT / "api" / "response_window").exists()
    assert not (READER_ROOT / "runtime" / "response_window.py").exists()
    assert not (READER_ROOT / "workbench" / "cli" / "response_window.py").exists()
    assert not (READER_ROOT / "runtime" / "sfxi_vec8_aggregate.py").exists()


def test_protocol_compilers_own_assay_step_composition() -> None:
    retired_recipe_surfaces = (
        READER_ROOT / "workbench" / "recipes",
        READER_ROOT / "workbench" / "recipes.py",
    )
    compiler_root = READER_ROOT / "protocols" / "compilers"
    stale_imports = [
        str(path.relative_to(READER_ROOT))
        for path in sorted(compiler_root.rglob("*.py"))
        if "reader_workbench.workbench.recipes" in path.read_text(encoding="utf-8")
    ]

    assert not any(path.exists() for path in retired_recipe_surfaces), (
        "Assay step composition belongs to protocol compiler support"
    )
    assert stale_imports == [], f"Protocol compilers still depend on the retired recipe registry: {stale_imports}"


def test_workbench_owns_plugin_domain_ontology() -> None:
    legacy_owner = DOMAIN_ROOT / "semantics.py"
    legacy_import = ".".join(("reader_workbench", "domains", "semantics"))
    stale_imports: list[str] = []
    for path in sorted(READER_ROOT.rglob("*.py")):
        if path == legacy_owner:
            continue
        if legacy_import in path.read_text(encoding="utf-8"):
            stale_imports.append(str(path.relative_to(READER_ROOT)))

    assert not legacy_owner.exists(), "Plugin-domain ontology belongs to reader_workbench.workbench.ontology"
    assert stale_imports == [], f"Reader modules still import the former domain ontology owner: {stale_imports}"


def test_workbench_notebooks_do_not_own_scientific_analysis() -> None:
    violations: list[str] = []
    for path in sorted(WORKBENCH_NOTEBOOK_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for name in _imported_modules(node):
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in FORBIDDEN_WORKBENCH_NOTEBOOK_ANALYSIS_DEPENDENCIES
                ):
                    violations.append(f"{path.relative_to(READER_ROOT)}:{node.lineno} imports {name}")

    legacy_owner = WORKBENCH_NOTEBOOK_ROOT / "dual_reporter_triptych.py"
    assert not legacy_owner.exists(), (
        "Triptych analysis and plotting belong to reader_workbench.domains.plate_reader.plots"
    )
    assert violations == [], (
        "Workbench notebook modules must compose domain operations, not own analysis:\n" + "\n".join(violations)
    )


def test_generated_notebooks_use_only_canonical_dataframe_records() -> None:
    violations: list[str] = []
    notebook_sources = sorted(READER_ROOT.rglob("*.marimo.py.txt"))
    assert notebook_sources == [CANONICAL_NOTEBOOK_SOURCE], "Reader must package exactly one fixed notebook scaffold"
    for path in notebook_sources:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        api_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "reader_workbench.api"
            for alias in node.names
        }
        missing = {"read_dataframe", "records"} - api_imports
        if missing:
            violations.append(f"{path.name} omits reader_workbench.api imports: {sorted(missing)}")
        if "reader_workbench.workbench.records" in source:
            violations.append(f"{path.name} imports the internal record store")
        if "reader_workbench.workbench.notebooks" in source:
            violations.append(f"{path.name} imports internal notebook orchestration")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else None
                if name in {"read_parquet", "scan_parquet", "scan_event_table"}:
                    violations.append(f"{path.name}:{node.lineno} calls {name}")

    assert not (READER_ROOT / "workbench" / "records" / "discovery.py").exists()
    assert not (WORKBENCH_NOTEBOOK_ROOT / "context.py").exists()
    assert not (WORKBENCH_NOTEBOOK_ROOT / "artifacts.py").exists()
    for path in sorted(WORKBENCH_NOTEBOOK_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"read_csv", "read_excel", "read_parquet", "scan_parquet"}:
                violations.append(f"{path.relative_to(READER_ROOT)}:{node.lineno} calls {node.func.attr}")
    assert violations == [], (
        "Generated notebooks must catalog and digest-verify dataframes through reader_workbench.api:\n"
        + "\n".join(violations)
    )


def test_domain_operations_do_not_publish_files() -> None:
    violations: list[str] = []
    sink_parameters = {"destination", "destination_dir", "out_dir", "output_dir", "output_path"}
    sink_calls = {
        "mkdir",
        "save_figure",
        "savefig",
        "to_csv",
        "to_excel",
        "to_json",
        "to_parquet",
        "to_pickle",
        "write_bytes",
        "write_text",
    }
    for path in sorted(DOMAIN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                accepted = sorted(parameter.arg for parameter in parameters if parameter.arg in sink_parameters)
                if accepted:
                    violations.append(
                        f"{path.relative_to(READER_ROOT)}:{node.lineno} accepts publication path(s) {accepted}"
                    )
            if isinstance(node, ast.Call):
                call_name = _call_name(node)
                if call_name in sink_calls:
                    violations.append(f"{path.relative_to(READER_ROOT)}:{node.lineno} calls {call_name}")

    assert violations == [], (
        "Domain operations must return in-memory values; publication belongs to runtime adapters:\n"
        + "\n".join(violations)
    )


def test_logic_symmetry_domain_has_no_private_output_sink() -> None:
    assert not (DOMAIN_ROOT / "logic" / "logic_symmetry" / "io.py").exists()


def _imported_modules(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return (node.module,)
    return ()


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None
