"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_notebook_templates.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from reader.errors import ConfigError
from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.workbench.assets import AssetCapabilities
from reader.workbench.notebooks.scaffold import write_experiment_notebook
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


def test_notebook_templates_render_through_scaffold_and_pass_marimo_check(tmp_path: Path) -> None:
    run_marimo_check = importlib.util.find_spec("marimo") is not None
    rendered_paths: list[Path] = []
    for descriptor in builtin_notebook_template_catalog().all():
        target = tmp_path / descriptor.template.replace("/", "__")
        target = target.with_suffix(".py")

        rendered_path, changed = write_experiment_notebook(
            target,
            template=descriptor.template,
            overwrite=True,
            plot_specs=[],
            allow_record_scan=False,
        )
        content = rendered_path.read_text(encoding="utf-8")

        assert changed is True
        assert "__ALLOW_RECORD_SCAN__" not in content
        assert "__PLOT_SPECS__" not in content
        if not run_marimo_check:
            continue
        rendered_paths.append(rendered_path)

    if not run_marimo_check:
        return
    assert rendered_paths, "Built-in notebook template catalog must not be empty."
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "check", *map(str, rendered_paths)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Rendered notebooks failed marimo check:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_notebook_template_uses_explicit_record_scan_placeholder() -> None:
    template = resolve_notebook_template_descriptor("notebook/basic").load_body()
    assert "pl.read_parquet" in template
    assert "pd.read_parquet" not in template
    assert "Polars is required to read parquet" in template
    assert "discover_dataframe_records" in template
    assert "allow_scan=__ALLOW_RECORD_SCAN__" in template


def test_triptych_notebook_templates_debounce_snapshot_time_slider() -> None:
    for template_name in ("notebook/dual_reporter_triptych", "notebook/sfxi_eda"):
        template = resolve_notebook_template_descriptor(template_name).load_body()
        assert "debounce=True" in template
        assert "chart_selection=False" in template
        assert "legend_selection=False" in template
        assert "min-height" in template
        assert "mo.output.replace(_chart_panel)" in template
        assert "Selected design" in template
        assert "Triptych context" not in template
        assert "Design alias" in template
        assert "Sequence" in template


def test_sfxi_notebook_uses_protocol_bound_transform_config() -> None:
    template = resolve_notebook_template_descriptor("notebook/sfxi_eda").load_body()

    assert "bind_protocol(decl.experiment_semantics.protocol)" in template
    assert "effective_plugin_config(" in template


def test_sfxi_notebook_surfaces_deliverables_with_progressive_disclosure() -> None:
    template = resolve_notebook_template_descriptor("notebook/sfxi_eda").load_body()

    assert "collect_notebook_deliverables" in template
    assert "render_notebook_deliverables_panel" in template
    assert "render_notebook_deliverables_panel(mo, deliverables)" in template
    assert "render_notebook_overview_panel" in template
    assert "data_ready" not in template
    assert "sfxi_raw_data_panel = mo.accordion" in template
    assert "Raw records and generated outputs" in template
    assert "def _(eda_base_panel)" not in template


def test_sfxi_notebook_export_panel_requires_valid_vec8_result() -> None:
    template = resolve_notebook_template_descriptor("notebook/sfxi_eda").load_body()

    assert (
        "def _(Path, design_select, exports_dir, mo, sfxi_cfg, time_selected_h, time_target_h, vec8_result)" in template
    )
    assert '"Reference anchor"' in template
    assert '"Snapshot time used (h)"' in template


def test_sfxi_notebook_treatment_condition_labels_respect_case_insensitive_config() -> None:
    template = resolve_notebook_template_descriptor("notebook/sfxi_eda").load_body()

    assert "if case_sensitive:" in template
    assert "return text.strip().casefold()" in template
    assert "_treatment_key = _raw_treatment.str.strip().str.casefold()" in template


def test_sfxi_notebook_triptych_uses_closed_corner_condition_labels() -> None:
    template = resolve_notebook_template_descriptor("notebook/sfxi_eda").load_body()

    assert "sfxi_condition_order" in template
    assert 'f"{_corner}: {sfxi_cfg.treatment_map[_corner]}"' in template
    assert 'sfxi_triptych_treatment_col = "sfxi_condition"' in template
    assert "sfxi_triptych_rows[sfxi_triptych_treatment_col].isin(sfxi_condition_order)" in template
    assert "treatment_order=sfxi_condition_order" in template


def test_notebook_template_catalog_exposes_domain_semantics() -> None:
    descriptors = {item.template: item for item in builtin_notebook_template_catalog().all()}
    assert descriptors["notebook/eda"].domain == "generic"
    assert descriptors["notebook/microplate"].domain == "plate_reader"
    assert descriptors["notebook/dual_reporter_triptych"].domain == "plate_reader"
    assert descriptors["notebook/retron_sponge"].domain == "plate_reader"
    assert descriptors["notebook/retron_sponge_aggregate"].domain == "generic"
    assert descriptors["notebook/cytometry"].domain == "cytometry"
    assert descriptors["notebook/sfxi_eda"].domain == "logic"
    assert descriptors["notebook/eda"].capabilities.supports_plot_filters is True
    assert descriptors["notebook/eda"].capabilities.inject_plot_specs is True
    assert descriptors["notebook/retron_sponge"].family == "screen_review"
    assert descriptors["notebook/retron_sponge"].capabilities.supports_plot_filters is True
    assert descriptors["notebook/retron_sponge"].capabilities.inject_plot_specs is True
    assert descriptors["notebook/retron_sponge_aggregate"].family == "screen_review"


def test_notebook_template_default_selection_uses_protocol_policy() -> None:
    catalog = builtin_protocol_catalog()
    assert (
        select_default_notebook_template(
            protocol=catalog.bind(ProtocolBinding(id="plate_reader/retron_sponge_screen"))
        ).template
        == "notebook/retron_sponge"
    )
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
    assert templates == ["notebook/sfxi_eda", "notebook/dual_reporter_triptych", "notebook/eda", "notebook/basic"]
    descriptor = require_notebook_template_for_protocol("notebook/sfxi_eda", protocol=protocol)
    assert descriptor.template == "notebook/sfxi_eda"
    with pytest.raises(ConfigError, match="does not allow notebook template"):
        require_notebook_template_for_protocol("notebook/cytometry", protocol=protocol)


def test_retron_notebook_template_catalog_filters_by_protocol() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="plate_reader/retron_sponge_screen"))
    templates = [item.template for item in compatible_notebook_templates(protocol=protocol)]
    assert templates == ["notebook/retron_sponge", "notebook/eda", "notebook/microplate", "notebook/basic"]


def test_generic_notebook_template_catalog_includes_retron_aggregate_review() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="workbench/generic"))
    templates = [item.template for item in compatible_notebook_templates(protocol=protocol)]
    assert templates == [
        "notebook/basic",
        "notebook/retron_sponge_aggregate",
        "notebook/eda",
        "notebook/microplate",
        "notebook/dual_reporter_triptych",
        "notebook/cytometry",
        "notebook/sfxi_eda",
    ]


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
