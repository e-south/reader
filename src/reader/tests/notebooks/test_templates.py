"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_notebook_templates.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import ast

import pytest

from reader.errors import ConfigError
from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.workbench.assets import AssetCapabilities
from reader.workbench.templates import (
    NotebookTemplateCatalog,
    NotebookTemplateDescriptor,
    builtin_notebook_template_catalog,
    compatible_notebook_templates,
    require_notebook_template_for_protocol,
    resolve_notebook_template_descriptor,
    select_default_notebook_template,
)


def _is_app_cell(dec: ast.AST) -> bool:
    return (
        isinstance(dec, ast.Attribute)
        and isinstance(dec.value, ast.Name)
        and dec.value.id == "app"
        and dec.attr == "cell"
    )


class _Collector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name)


def _find_duplicates(template: str) -> set[str]:
    tree = ast.parse(template)
    seen: set[str] = set()
    dupes: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(_is_app_cell(dec) for dec in node.decorator_list):
            collector = _Collector()
            for stmt in node.body:
                collector.visit(stmt)
            for name in collector.names:
                if name.startswith("_"):
                    continue
                if name in seen:
                    dupes.add(name)
                seen.add(name)
    return dupes


def test_notebook_templates_no_duplicate_globals() -> None:
    for descriptor in builtin_notebook_template_catalog().all():
        name = descriptor.template
        template = descriptor.load_body()
        dupes = sorted(_find_duplicates(template))
        assert not dupes, f"{name} defines the same non-private name in multiple cells: {dupes}"


def test_notebook_templates_parse() -> None:
    for descriptor in builtin_notebook_template_catalog().all():
        name = descriptor.template
        template = descriptor.load_body()
        try:
            ast.parse(template)
        except SyntaxError as exc:  # pragma: no cover - explicit failure path
            raise AssertionError(f"{name} template has invalid syntax: {exc}") from exc


def test_notebook_template_uses_explicit_record_scan_placeholder() -> None:
    template = resolve_notebook_template_descriptor("notebook/basic").load_body()
    assert "pl.read_parquet" in template
    assert "pd.read_parquet" not in template
    assert "Polars is required to read parquet" in template
    assert "discover_dataframe_records" in template
    assert "allow_scan=__ALLOW_RECORD_SCAN__" in template


def test_notebook_template_catalog_exposes_domain_semantics() -> None:
    descriptors = {item.template: item for item in builtin_notebook_template_catalog().all()}
    assert descriptors["notebook/eda"].domain == "generic"
    assert descriptors["notebook/microplate"].domain == "plate_reader"
    assert descriptors["notebook/cytometry"].domain == "cytometry"
    assert descriptors["notebook/sfxi_eda"].domain == "logic"
    assert descriptors["notebook/eda"].capabilities.supports_plot_filters is True
    assert descriptors["notebook/eda"].capabilities.inject_plot_specs is True


def test_notebook_template_default_selection_uses_protocol_policy() -> None:
    catalog = builtin_protocol_catalog()
    assert (
        select_default_notebook_template(
            protocol=catalog.bind(ProtocolBinding(id="plate_reader/dual_reporter_screen"))
        ).template
        == "notebook/eda"
    )
    assert (
        select_default_notebook_template(protocol=catalog.bind(ProtocolBinding(id="cytometry/flow_panel"))).template
        == "notebook/cytometry"
    )
    assert (
        select_default_notebook_template(protocol=catalog.bind(ProtocolBinding(id="workbench/generic"))).template
        == "notebook/basic"
    )


def test_notebook_template_catalog_filters_by_protocol() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="logic/sfxi_screen"))
    templates = [item.template for item in compatible_notebook_templates(protocol=protocol)]
    assert templates == ["notebook/sfxi_eda", "notebook/eda", "notebook/basic"]
    descriptor = require_notebook_template_for_protocol("notebook/sfxi_eda", protocol=protocol)
    assert descriptor.template == "notebook/sfxi_eda"
    with pytest.raises(ConfigError, match="does not allow notebook template"):
        require_notebook_template_for_protocol("notebook/cytometry", protocol=protocol)


def test_notebook_template_catalog_rejects_duplicate_templates() -> None:
    descriptor = NotebookTemplateDescriptor(
        template="notebook/eda",
        domain="generic",
        family="record_explorer",
        summary="x",
        source_package="reader.workbench.templates.builtins",
        source_name="basic.marimo.py",
        capabilities=AssetCapabilities(),
    )
    with pytest.raises(ConfigError, match="Duplicate template"):
        NotebookTemplateCatalog([descriptor, descriptor])
