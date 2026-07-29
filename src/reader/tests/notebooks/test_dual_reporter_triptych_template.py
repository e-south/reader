from __future__ import annotations

from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.workbench.templates import (
    compatible_notebook_templates,
    resolve_notebook_template_descriptor,
)


def test_dual_reporter_triptych_template_is_protocol_neutral() -> None:
    descriptor = resolve_notebook_template_descriptor("notebook/dual_reporter_triptych")

    assert descriptor.domain == "plate_reader"
    assert "sfxi" not in descriptor.tags
    body = descriptor.load_body()
    assert "Dual-reporter triptych" in body
    assert "debounce=True" in body
    assert "chart_selection=False" in body
    assert "mo.output.replace(_chart_panel)" in body
    assert "Selected design" in body
    assert "Triptych context" not in body
    assert "summarize_design_context" in body
    assert "bootstrap CI" in body
    assert "Export 8-vector" not in body


def test_triptych_templates_consume_the_domain_owner() -> None:
    for template_name in ("notebook/dual_reporter_triptych", "notebook/sfxi_eda"):
        body = resolve_notebook_template_descriptor(template_name).load_body()
        assert "reader.domains.plate_reader.plots.dual_reporter_triptych" in body
        assert "reader.workbench.notebooks.dual_reporter_triptych" not in body


def test_dual_reporter_screen_allows_triptych_without_sfxi_vec8() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="plate_reader/dual_reporter_screen"))
    templates = [item.template for item in compatible_notebook_templates(protocol=protocol)]

    assert "notebook/dual_reporter_triptych" in templates
    assert "notebook/sfxi_eda" not in templates
