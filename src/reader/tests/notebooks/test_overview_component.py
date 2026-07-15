from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reader.workbench.notebooks.components.overview import (
    build_design_treatment_summary_rows,
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


def test_build_design_treatment_summary_rows_lists_vocabularies_without_pairing() -> None:
    df = pd.DataFrame(
        {
            "design_id": ["pDual-10-SECG-B0-ETH-02", "pDual-10-SECG-B0-ETH-01", None],
            "treatment": ["EtOH_3_percent_0nM_cipro", "EtOH_0_percent_0nM_cipro", "EtOH_3_percent_0nM_cipro"],
        }
    )

    rows, note = build_design_treatment_summary_rows(df)

    assert note == ""
    assert rows == (
        {"Category": "Design ID", "Value": "pDual-10-SECG-B0-ETH-01"},
        {"Category": "Design ID", "Value": "pDual-10-SECG-B0-ETH-02"},
        {"Category": "Treatment", "Value": "EtOH_0_percent_0nM_cipro"},
        {"Category": "Treatment", "Value": "EtOH_3_percent_0nM_cipro"},
    )


def test_build_notebook_overview_requires_identity_and_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="experiment_id is required"):
        build_notebook_overview(
            experiment_id=" ",
            experiment_title="demo",
            protocol_id="logic/sfxi_screen",
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
    experiment_id = "20260706_sfxi_sensor-panel-m9-glu-secg"
    overview = build_notebook_overview(
        experiment_id=experiment_id,
        experiment_title=experiment_id,
        protocol_id="logic/sfxi_screen",
        experiment_root=tmp_path,
        outputs_dir=tmp_path / "outputs",
        notebooks_dir=tmp_path / "outputs" / "notebooks",
        pipeline_step_ids=(),
    )

    assert overview.experiment_id == experiment_id
    assert overview.experiment_title == "2026-07-06 · SFXI Sensor Panel M9 Glu SECG"


def test_render_notebook_overview_panel_uses_lazy_accordion(tmp_path: Path) -> None:
    mo = _FakeMarimo()
    overview = build_notebook_overview(
        experiment_id="exp_sfxi",
        experiment_title="SFXI run",
        protocol_id="logic/sfxi_screen",
        experiment_root=tmp_path,
        outputs_dir=tmp_path / "outputs",
        notebooks_dir=tmp_path / "outputs" / "notebooks",
        pipeline_step_ids=("ingest", "sfxi_vec8"),
    )

    panel = render_notebook_overview_panel(
        mo,
        overview,
        design_treatment_rows=({"Category": "Design ID", "Value": "pDual-10"},),
    )

    assert panel["kind"] == "vstack"
    assert panel["items"][0] == {"kind": "markdown", "text": "# SFXI run"}
    assert panel["items"][1]["rows"] == [
        {"Field": "Experiment ID", "Value": "exp_sfxi"},
        {"Field": "Protocol", "Value": "logic/sfxi_screen"},
        {"Field": "Pipeline steps", "Value": 2},
    ]
    assert mo.accordion_calls == [
        {
            "sections": {
                "Design/treatment scope": {
                    "kind": "table",
                    "rows": [{"Category": "Design ID", "Value": "pDual-10"}],
                    "page_size": 1,
                },
                "Pipeline": {
                    "kind": "table",
                    "rows": [{"Order": 1, "Step ID": "ingest"}, {"Order": 2, "Step ID": "sfxi_vec8"}],
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
