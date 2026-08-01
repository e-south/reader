from __future__ import annotations

from pathlib import Path

import pytest

from reader_workbench.workbench.notebooks.components.overview import (
    build_notebook_overview,
    render_notebook_overview_panel,
)


class _FakeUi:
    def table(self, rows, *, page_size):
        return {"kind": "table", "rows": list(rows), "page_size": page_size}


class _FakeMarimo:
    def __init__(self) -> None:
        self.ui = _FakeUi()
        self.accordion_calls = []

    def md(self, text):
        return {"kind": "markdown", "text": text}

    def accordion(self, sections, *, multiple, lazy):
        self.accordion_calls.append({"sections": sections, "multiple": multiple, "lazy": lazy})
        return {"kind": "accordion", "sections": sections}

    def vstack(self, items):
        return {"kind": "vstack", "items": list(items)}


def test_build_notebook_overview_requires_identity_and_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="experiment_id is required"):
        build_notebook_overview(
            experiment_id=" ",
            experiment_title="demo",
            protocol_id="logic/four_state_vector_screen",
            experiment_root=tmp_path,
            outputs_dir=tmp_path / "outputs",
            notebooks_dir=tmp_path / "outputs" / "notebooks",
            pipeline_step_ids=(),
        )

    with pytest.raises(ValueError, match="protocol_id is required"):
        build_notebook_overview(
            experiment_id="exp",
            experiment_title="demo",
            protocol_id="",
            experiment_root=tmp_path,
            outputs_dir=tmp_path / "outputs",
            notebooks_dir=tmp_path / "outputs" / "notebooks",
            pipeline_step_ids=(),
        )


def test_build_notebook_overview_humanizes_compiled_identity_fallback(tmp_path: Path) -> None:
    experiment_id = "20260102_four_state_vector_four-state-panel"
    overview = build_notebook_overview(
        experiment_id=experiment_id,
        experiment_title=experiment_id,
        protocol_id="logic/four_state_vector_screen",
        experiment_root=tmp_path,
        outputs_dir=tmp_path / "outputs",
        notebooks_dir=tmp_path / "outputs" / "notebooks",
        pipeline_step_ids=(),
    )

    assert overview.experiment_id == experiment_id
    assert overview.experiment_title == "2026-01-02 · Four State Vector Four State Panel"


def test_render_notebook_overview_panel_is_assay_neutral_and_uses_lazy_accordion(tmp_path: Path) -> None:
    mo = _FakeMarimo()
    overview = build_notebook_overview(
        experiment_id="generic_run",
        experiment_title="Generic run",
        protocol_id="workbench/generic",
        experiment_root=tmp_path,
        outputs_dir=tmp_path / "outputs",
        notebooks_dir=tmp_path / "outputs" / "notebooks",
        pipeline_step_ids=("ingest", "normalize"),
    )

    panel = render_notebook_overview_panel(mo, overview)

    assert panel["kind"] == "vstack"
    assert panel["items"][0] == {"kind": "markdown", "text": "# Generic run"}
    assert panel["items"][1]["rows"] == [
        {"Field": "Experiment ID", "Value": "generic_run"},
        {"Field": "Protocol", "Value": "workbench/generic"},
        {"Field": "Pipeline steps", "Value": 2},
    ]
    assert mo.accordion_calls == [
        {
            "sections": {
                "Pipeline": {
                    "kind": "table",
                    "rows": [{"Order": 1, "Step ID": "ingest"}, {"Order": 2, "Step ID": "normalize"}],
                    "page_size": 2,
                },
                "Paths": {
                    "kind": "table",
                    "rows": [
                        {"Path": "Experiment root", "Value": "."},
                        {"Path": "Outputs", "Value": "outputs"},
                        {"Path": "Generated notebooks", "Value": "outputs/notebooks"},
                    ],
                    "page_size": 3,
                },
            },
            "multiple": True,
            "lazy": True,
        }
    ]


def test_render_notebook_overview_panel_accepts_semantic_detail_sections(tmp_path: Path) -> None:
    mo = _FakeMarimo()
    overview = build_notebook_overview(
        experiment_id="exp",
        experiment_title="Experiment",
        protocol_id="workbench/generic",
        experiment_root=tmp_path,
        outputs_dir=tmp_path / "outputs",
        notebooks_dir=tmp_path / "outputs" / "notebooks",
        pipeline_step_ids=(),
    )
    scope = mo.md("Protocol-owned context")

    render_notebook_overview_panel(
        mo,
        overview,
        detail_sections={"Assay context": scope},
    )

    sections = mo.accordion_calls[-1]["sections"]
    assert list(sections) == ["Assay context", "Pipeline", "Paths"]
    assert sections["Assay context"] is scope
