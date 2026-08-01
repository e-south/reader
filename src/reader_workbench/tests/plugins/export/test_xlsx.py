from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from reader_workbench.errors import ExecutionError
from reader_workbench.plugins.export.csv import ExportCsv, ExportCsvCfg
from reader_workbench.plugins.export.xlsx import ExportXlsx, ExportXlsxCfg
from reader_workbench.workbench.context import RunContext


def _ctx(tmp_path: Path) -> RunContext:
    outputs = tmp_path / "outputs"
    return RunContext(
        exp_dir=tmp_path,
        outputs_dir=outputs,
        artifacts_dir=outputs / "artifacts",
        plots_dir=outputs / "plots",
        exports_dir=outputs / "exports",
        records_path=outputs / "manifests" / "records.json",
        logger=logging.getLogger("test"),
        palette_book=None,
    )


def test_export_xlsx_writes_readable_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3.0, 4.0]})
    cfg = ExportXlsxCfg(path="vector.xlsx", sheet_name="vector", index=False)

    out = ExportXlsx().run(ctx, {"df": df}, cfg)
    out_path = Path(out["artifact"])
    assert out_path.exists()

    back = pd.read_excel(out_path, sheet_name="vector")
    assert list(back.columns) == ["col_a", "col_b"]
    assert back.shape == (2, 2)


@pytest.mark.parametrize(
    ("plugin", "config_type", "suffix"),
    [
        (ExportCsv(), ExportCsvCfg, ".csv"),
        (ExportXlsx(), ExportXlsxCfg, ".xlsx"),
    ],
)
def test_exports_reject_absolute_and_parent_escape_paths(tmp_path, plugin, config_type, suffix) -> None:
    ctx = _ctx(tmp_path)
    frame = pd.DataFrame({"value": [1.0]})
    outside_absolute = tmp_path / f"outside-absolute{suffix}"
    outside_parent = ctx.outputs_dir / f"outside-parent{suffix}"

    for raw_path, outside in (
        (str(outside_absolute), outside_absolute),
        (f"../{outside_parent.name}", outside_parent),
    ):
        with pytest.raises(ExecutionError, match="Export paths must"):
            plugin.run(ctx, {"df": frame}, config_type(path=raw_path))
        assert not outside.exists()


@pytest.mark.parametrize(
    ("plugin", "config_type", "suffix"),
    [
        (ExportCsv(), ExportCsvCfg, ".csv"),
        (ExportXlsx(), ExportXlsxCfg, ".xlsx"),
    ],
)
def test_exports_reject_symlink_escape(tmp_path, plugin, config_type, suffix) -> None:
    ctx = _ctx(tmp_path)
    ctx.exports_dir.mkdir(parents=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    link = ctx.exports_dir / "escape"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ExecutionError, match="stay under the experiment exports directory"):
        plugin.run(ctx, {"df": pd.DataFrame({"value": [1.0]})}, config_type(path=f"escape/result{suffix}"))
    assert not (outside_dir / f"result{suffix}").exists()
