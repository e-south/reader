from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.plugins.plot.logic_symmetry import LogicSymmetryPlot, LogicSymmetryPlotCfg
from reader_workbench.plugins.transform.logic_symmetry import LogicSymmetryCfg, LogicSymmetryTransform
from reader_workbench.workbench import PluginSemantics
from reader_workbench.workbench.assets import build_plugin_asset
from reader_workbench.workbench.experiment import AnnotationSemantics, OrderedStateSpaces, OrderedStateSpaceSpec


def _logic_symmetry_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["A1", "A2", "A3", "A4"],
            "time": [1.0, 1.0, 1.0, 1.0],
            "channel": ["signal", "signal", "signal", "signal"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "design_id": ["design_a", "design_a", "design_a", "design_a"],
            "batch": [0, 0, 0, 0],
            "condition": ["off", "input_a", "input_b", "both"],
        }
    )


def _context(*, state_order=("00", "10", "01", "11"), plots_dir: Path | None = None):
    source_values = {"00": "off", "10": "input_a", "01": "input_b", "11": "both"}
    annotations = AnnotationSemantics(
        ordered_state_spaces=OrderedStateSpaces(
            by_id={
                "states": OrderedStateSpaceSpec(
                    column="condition",
                    state_order=state_order,
                    source_values={state_id: source_values[state_id] for state_id in state_order},
                )
            }
        )
    )
    return SimpleNamespace(
        experiment=SimpleNamespace(annotations=annotations),
        palette_book=None,
        plots_dir=plots_dir,
    )


def _summary_table() -> pd.DataFrame:
    result = LogicSymmetryTransform().run(
        _context(),
        {"df": _logic_symmetry_input()},
        LogicSymmetryCfg(
            response_channel="signal",
            design_by=["design_id"],
            state_map_ref="states",
        ),
    )
    return result["table"]


def test_logic_symmetry_transform_rejects_noncanonical_state_space() -> None:
    cfg = LogicSymmetryCfg(response_channel="signal", state_map_ref="states")

    with pytest.raises(
        ValueError, match="Logic-symmetry state space must declare exactly 00, 10, 01, 11 in that order"
    ):
        LogicSymmetryTransform().run(
            _context(state_order=("00", "01", "10", "11")),
            {"df": _logic_symmetry_input()},
            cfg,
        )


def test_logic_symmetry_transform_owns_the_summary_record() -> None:
    table = _summary_table()

    assert set(LogicSymmetryTransform.output_ports()) == {"table"}
    assert table["design_id"].tolist() == ["design_a"]
    assert table["baseline_corner"].tolist() == ["00"]


def test_logic_symmetry_plot_consumes_the_summary_record() -> None:
    cfg = LogicSymmetryPlotCfg(format=["png", "pdf"], dpi=72, figsize=(4, 3))

    figures = LogicSymmetryPlot().render(_context(), {"table": _summary_table()}, cfg)

    assert set(LogicSymmetryPlot.input_ports()) == {"table"}
    assert set(LogicSymmetryPlot.output_ports()) == {"artifacts"}
    assert [(item.filename, item.ext, item.dpi) for item in figures] == [
        ("logic_symmetry", "png", 72),
        ("logic_symmetry", "pdf", 72),
    ]
    plt.close(figures[0].fig)


def test_logic_symmetry_plot_publishes_with_the_shared_sink(tmp_path: Path) -> None:
    plugin = LogicSymmetryPlot()
    plugin.bind_runtime(
        descriptor=build_plugin_asset(
            plugin_id="plot/logic_symmetry",
            semantics=PluginSemantics(
                domain="logic",
                family="geometry_plot",
                summary="Render logic symmetry geometry.",
            ),
            plugin_cls=LogicSymmetryPlot,
        ),
        contracts=builtin_contract_catalog(),
    )

    result = plugin.run(
        _context(plots_dir=tmp_path),
        {"table": _summary_table()},
        LogicSymmetryPlotCfg(format=["png", "pdf"], dpi=72, figsize=(4, 3)),
    )
    paths = [Path(item.path if hasattr(item, "path") else item) for item in result["artifacts"]]

    assert [path.name for path in paths] == ["logic_symmetry.png", "logic_symmetry.pdf"]
    assert all(path.is_file() for path in paths)
