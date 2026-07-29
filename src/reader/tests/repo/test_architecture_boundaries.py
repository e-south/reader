from __future__ import annotations

import ast
from pathlib import Path

READER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
DOMAIN_ROOT = READER_ROOT / "domains"
FORBIDDEN_DOMAIN_DEPENDENCIES = (
    "reader.api",
    "reader.maintenance",
    "reader.plugins",
    "reader.protocols",
    "reader.runtime",
    "reader.workbench",
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


def test_repository_has_no_parallel_work_product_roots() -> None:
    stray = [name for name in ("outputs", "tmp") if (REPO_ROOT / name).exists()]

    assert stray == [], (
        "Reader work products belong to experiments/<year>/<experiment>/outputs; "
        f"local scratch belongs in .tmp. Stray repository roots: {stray}"
    )


def test_runtime_plugin_namespace_is_a_packaged_boundary() -> None:
    marker = READER_ROOT / "plugins" / "__init__.py"

    assert marker.is_file(), "reader.plugins must be a regular package so wheel builds include built-in plugins"


def test_repo_local_skills_use_the_codex_discovery_root() -> None:
    skills_root = REPO_ROOT / ".agents" / "skills"

    assert skills_root.is_dir(), "repository skills must be discoverable under .agents/skills"
    assert not (REPO_ROOT / "skills").exists(), "do not maintain a parallel, undiscoverable skills root"


def test_domain_capabilities_do_not_create_parallel_public_lifecycles() -> None:
    assert not (READER_ROOT / "api" / "response_window").exists()
    assert not (READER_ROOT / "runtime" / "response_window.py").exists()
    assert not (READER_ROOT / "workbench" / "cli" / "response_window.py").exists()
    assert not (READER_ROOT / "runtime" / "sfxi_vec8_aggregate.py").exists()


def _imported_modules(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return (node.module,)
    return ()
