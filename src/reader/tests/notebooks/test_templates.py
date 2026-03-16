"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_notebook_templates.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import ast

import pytest

from reader.core.errors import ConfigError
from reader.workbench.assets import AssetCapabilities, select_default_notebook_template
from reader.workbench.notebooks import notebook_template_catalog
from reader.workbench.notebooks import templates as notebook_templates
from reader.workbench.notebooks.catalog import NotebookTemplateCatalog, NotebookTemplateDescriptor


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
    templates = {
        "notebook/eda": notebook_templates.EXPERIMENT_NOTEBOOK_EDA_TEMPLATE,
        "notebook/basic": notebook_templates.EXPERIMENT_EDA_BASIC_TEMPLATE,
        "notebook/microplate": notebook_templates.EXPERIMENT_EDA_MICROPLATE_TEMPLATE,
        "notebook/cytometry": notebook_templates.EXPERIMENT_EDA_CYTOMETRY_TEMPLATE,
        "notebook/sfxi_eda": notebook_templates.EXPERIMENT_SFXI_EDA_TEMPLATE,
    }
    for name, template in templates.items():
        dupes = sorted(_find_duplicates(template))
        assert not dupes, f"{name} defines the same non-private name in multiple cells: {dupes}"


def test_notebook_templates_parse() -> None:
    templates = {
        "notebook/eda": notebook_templates.EXPERIMENT_NOTEBOOK_EDA_TEMPLATE,
        "notebook/basic": notebook_templates.EXPERIMENT_EDA_BASIC_TEMPLATE,
        "notebook/microplate": notebook_templates.EXPERIMENT_EDA_MICROPLATE_TEMPLATE,
        "notebook/cytometry": notebook_templates.EXPERIMENT_EDA_CYTOMETRY_TEMPLATE,
        "notebook/sfxi_eda": notebook_templates.EXPERIMENT_SFXI_EDA_TEMPLATE,
    }
    for name, template in templates.items():
        try:
            ast.parse(template)
        except SyntaxError as exc:  # pragma: no cover - explicit failure path
            raise AssertionError(f"{name} template has invalid syntax: {exc}") from exc


def test_notebook_template_uses_explicit_record_scan_placeholder() -> None:
    template = notebook_templates.EXPERIMENT_EDA_BASIC_TEMPLATE
    assert "pl.read_parquet" in template
    assert "pd.read_parquet" not in template
    assert "Polars is required to read parquet" in template
    assert "discover_dataframe_records" in template
    assert "allow_scan=__ALLOW_RECORD_SCAN__" in template


def test_notebook_template_catalog_exposes_domain_semantics() -> None:
    descriptors = {item.template: item for item in notebook_template_catalog().all()}
    assert descriptors["notebook/eda"].domain == "generic"
    assert descriptors["notebook/microplate"].domain == "plate_reader"
    assert descriptors["notebook/cytometry"].domain == "cytometry"
    assert descriptors["notebook/sfxi_eda"].domain == "logic"
    assert descriptors["notebook/eda"].capabilities.supports_plot_filters is True
    assert descriptors["notebook/eda"].capabilities.inject_plot_specs is True


def test_notebook_template_default_selection_uses_capabilities() -> None:
    assert select_default_notebook_template(has_plots=True, has_cytometry=False).template == "notebook/eda"
    assert select_default_notebook_template(has_plots=False, has_cytometry=True).template == "notebook/cytometry"
    assert select_default_notebook_template(has_plots=False, has_cytometry=False).template == "notebook/basic"


def test_notebook_template_catalog_rejects_duplicate_templates() -> None:
    descriptor = NotebookTemplateDescriptor(
        kind="template",
        name="notebook/eda",
        domain="generic",
        family="record_explorer",
        summary="x",
        body="print('x')",
        capabilities=AssetCapabilities(),
    )
    with pytest.raises(ConfigError, match="Duplicate template"):
        NotebookTemplateCatalog([descriptor, descriptor])
