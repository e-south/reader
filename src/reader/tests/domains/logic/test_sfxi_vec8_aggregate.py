from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import reader.domains.logic.sfxi.vec8_aggregate.writer as aggregate_writer
from reader.contracts import builtin_contract_catalog
from reader.domains.logic.sfxi.vec8_aggregate import (
    VEC8_CHANNELS,
    load_sfxi_vec8_sources,
    write_sfxi_vec8_aggregate,
)
from reader.domains.logic.sfxi.vec8_aggregate.render import render_sfxi_vec8_heatmap
from reader.errors import SFXIError
from reader.tests.support import base_reader_config, write_config
from reader.workbench.cli import app
from reader.workbench.records import RecordStore


def _vec8_df(*, design_prefix: str, v11: float, delta: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": [f"{design_prefix}-01", f"{design_prefix}-02"],
            "sequence": ["ACGT", "TGCA"],
            "id": [f"{design_prefix}-seq01", f"{design_prefix}-seq02"],
            "time_selected_h": [18.0, 18.0],
            "reference_design_id": ["pDual-10", "pDual-10"],
            "intensity_log2_offset_delta": [delta, delta],
            "r_logic": [4.0, 5.0],
            "v00": [0.0, 0.1],
            "v10": [1.0, 0.9],
            "v01": [0.2, 0.3],
            "v11": [v11, v11 / 2.0],
            "y00_star": [0.0, 0.1],
            "y10_star": [1.0, 0.8],
            "y01_star": [0.2, 0.3],
            "y11_star": [1.4, 0.7],
            "flat_logic": [False, False],
        }
    )


def _write_experiment_with_vec8(
    tmp_path: Path, *, experiment_id: str, design_prefix: str, v11: float, delta: float = 0.0
) -> Path:
    root = tmp_path / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = write_config(
        root,
        base_reader_config(
            experiment_id=experiment_id,
            lifecycle="active",
            protocol_id="logic/sfxi_screen",
            protocol_analysis={"include_vec8": True},
        ),
    )
    store = RecordStore(root / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="sfxi_vec8",
        producer_plugin="transform/sfxi",
        out_name="vec8",
        record_id="sfxi_vec8/vec8",
        df=_vec8_df(design_prefix=design_prefix, v11=v11, delta=delta),
        contract_id="sfxi.vec8.v3",
        inputs=[],
        config_digest=f"sha256:{experiment_id}",
    )
    return cfg_path


def test_load_sfxi_vec8_sources_reads_experiment_record_tables(tmp_path: Path) -> None:
    exp_a = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    exp_b = _write_experiment_with_vec8(tmp_path, experiment_id="20260707_sfxi", design_prefix="and", v11=0.5)

    aggregate = load_sfxi_vec8_sources([exp_a, exp_b.parent])

    assert [source.source_id for source in aggregate.sources] == ["20260706_sfxi", "20260707_sfxi"]
    assert aggregate.frame["source_id"].tolist() == [
        "20260706_sfxi",
        "20260706_sfxi",
        "20260707_sfxi",
        "20260707_sfxi",
    ]
    assert aggregate.frame["row_label"].tolist() == [
        "20260706_sfxi :: eth-01",
        "20260706_sfxi :: eth-02",
        "20260707_sfxi :: and-01",
        "20260707_sfxi :: and-02",
    ]
    assert list(VEC8_CHANNELS) == ["v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star"]


def test_write_sfxi_vec8_aggregate_writes_heatmap_tidy_csv_and_manifest(tmp_path: Path) -> None:
    exp_a = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    exp_b = _write_experiment_with_vec8(tmp_path, experiment_id="20260707_sfxi", design_prefix="and", v11=0.5)
    out_dir = tmp_path / "review" / "sfxi_vec8"

    artifacts = write_sfxi_vec8_aggregate(
        sources=[exp_a, exp_b],
        out_dir=out_dir,
        title="Measured SFXI vec8 aggregate",
    )

    assert artifacts.heatmap_path == out_dir / "sfxi_vec8_heatmap.png"
    assert artifacts.tidy_path == out_dir / "sfxi_vec8_heatmap_tidy.csv"
    assert artifacts.manifest_path == out_dir / "sfxi_vec8_heatmap_manifest.json"
    assert artifacts.heatmap_path.exists()
    tidy = pd.read_csv(artifacts.tidy_path)
    assert tidy["channel"].tolist()[:8] == list(VEC8_CHANNELS)
    assert set(tidy["source_id"]) == {"20260706_sfxi", "20260707_sfxi"}
    assert len(tidy) == 4 * len(VEC8_CHANNELS)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "reader.sfxi_vec8_aggregate.v1"
    assert manifest["summary"] == {"sources": 2, "rows": 4, "channels": 8}
    assert manifest["channels"] == list(VEC8_CHANNELS)
    assert manifest["intensity_log2_offset_deltas"] == [0.0]
    assert manifest["mixed_intensity_log2_offset_delta"] is False
    source_payload = manifest["sources"][0]
    assert source_payload["record_id"] == "sfxi_vec8/vec8"
    assert source_payload["record"]["contract_id"] == "sfxi.vec8.v3"
    assert source_payload["record"]["content_digest"].startswith("sha256:")
    assert source_payload["record"]["config_digest"] == "sha256:20260706_sfxi"
    assert source_payload["record"]["producer"] == {
        "kind": "pipeline",
        "id": "sfxi_vec8",
        "plugin": "transform/sfxi",
    }
    assert set(tidy["intensity_log2_offset_delta"]) == {0.0}


def test_write_sfxi_vec8_aggregate_reports_mixed_delta_in_manifest(tmp_path: Path) -> None:
    exp_a = _write_experiment_with_vec8(
        tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0, delta=0.0
    )
    exp_b = _write_experiment_with_vec8(
        tmp_path, experiment_id="20260707_sfxi", design_prefix="and", v11=0.5, delta=0.25
    )

    artifacts = write_sfxi_vec8_aggregate(sources=[exp_a, exp_b], out_dir=tmp_path / "aggregate")

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["intensity_log2_offset_deltas"] == [0.0, 0.25]
    assert manifest["mixed_intensity_log2_offset_delta"] is True


def test_load_sfxi_vec8_sources_rejects_missing_intensity_delta(tmp_path: Path) -> None:
    path = tmp_path / "incomplete_vec8.csv"
    _vec8_df(design_prefix="incomplete", v11=1.0).drop(columns=["intensity_log2_offset_delta"]).to_csv(
        path, index=False
    )

    with pytest.raises(SFXIError, match="requires column 'intensity_log2_offset_delta'"):
        load_sfxi_vec8_sources([path])


def test_render_sfxi_vec8_heatmap_uses_compact_display_labels(tmp_path: Path) -> None:
    cfg_path = _write_experiment_with_vec8(
        tmp_path,
        experiment_id="20260706_sfxi_sensor-panel-m9-glu-secg",
        design_prefix="pDual-10-SECG-B0-ETH",
        v11=1.0,
    )
    aggregate = load_sfxi_vec8_sources([cfg_path])

    fig = render_sfxi_vec8_heatmap(aggregate.frame, title="Measured SFXI vec8 aggregate")
    try:
        ax = fig.axes[0]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "v00",
            "v10",
            "v01",
            "v11",
            "y00*",
            "y10*",
            "y01*",
            "y11*",
        ]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "20260706 t=18.00h :: ETH-01",
            "20260706 t=18.00h :: ETH-02",
        ]
        annotation_labels = {text.get_text() for text in ax.texts}
        assert "Logic\npattern" in annotation_labels
        assert "Fluor.\nintensity" in annotation_labels
        assert r"vec8 = concat($v$, $y^\star$)" in annotation_labels
        colorbar_labels = {axis.xaxis.label.get_text() for axis in fig.axes[1:]}
        assert "$v_i$ normalized response" in colorbar_labels
        assert "$y_i^\\star$ log2 intensity" in colorbar_labels
        heatmap_box = ax.get_position()
        colorbar_boxes = [axis.get_position() for axis in fig.axes[1:]]
        assert all(box.width > box.height for box in colorbar_boxes)
        assert all(box.y1 < heatmap_box.y0 for box in colorbar_boxes)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        channel_boxes = [
            tick.get_window_extent(renderer=renderer)
            for tick in ax.get_xticklabels()
            if tick.get_visible() and tick.get_text()
        ]
        rendered_colorbar_boxes = [axis.get_tightbbox(renderer=renderer) for axis in fig.axes[1:]]
        assert min(box.y0 for box in channel_boxes) > max(box.y1 for box in rendered_colorbar_boxes) + 2.0
        title = next(text for text in fig.texts if text.get_text() == "Measured SFXI vec8 aggregate")
        assert title.get_position()[0] == pytest.approx((heatmap_box.x0 + heatmap_box.x1) / 2)
    finally:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.close(fig)


def test_render_sfxi_vec8_heatmap_sorts_controls_then_natural_design_ids() -> None:
    frame = _vec8_df(design_prefix="unused", v11=1.0).iloc[:0].copy()
    for design_id in ("pDual-10-ES10p", "pDual-10-spyp", "pDual-10-ES2p", "pDual-10-sulAp"):
        row = _vec8_df(design_prefix=design_id, v11=1.0).iloc[[0]].copy()
        row["design_id"] = design_id
        row["source_id"] = "20260706_sfxi_sensor-panel-m9-glu-secg"
        row["row_label"] = "20260706_sfxi_sensor-panel-m9-glu-secg :: " + design_id
        frame = pd.concat([frame, row], ignore_index=True)

    fig = render_sfxi_vec8_heatmap(frame, title="Measured SFXI vec8 aggregate")
    try:
        assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == [
            "20260706 t=18.00h :: spyP",
            "20260706 t=18.00h :: sulA",
            "20260706 t=18.00h :: ES2p",
            "20260706 t=18.00h :: ES10p",
        ]
    finally:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.close(fig)


def test_render_sfxi_vec8_heatmap_keeps_dense_tick_labels_from_overlapping() -> None:
    frame = _vec8_df(design_prefix="unused", v11=1.0).iloc[:0].copy()
    for index in range(72):
        design_id = f"pDual-10-SECG-B0-AND-{index + 1:02d}"
        row = _vec8_df(design_prefix=design_id, v11=1.0).iloc[[0]].copy()
        row["design_id"] = design_id
        row["source_id"] = "20260707_sfxi_sensor-panel-m9-glu-secg"
        row["row_label"] = "20260707_sfxi_sensor-panel-m9-glu-secg :: " + design_id
        frame = pd.concat([frame, row], ignore_index=True)

    fig = render_sfxi_vec8_heatmap(frame, title="Measured SFXI vec8 aggregate")
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = [tick for tick in fig.axes[0].get_yticklabels() if tick.get_visible() and str(tick.get_text()).strip()]
        boxes = sorted((label.get_window_extent(renderer=renderer) for label in labels), key=lambda box: box.y0)

        assert len(labels) == 72
        assert min(label.get_fontsize() for label in labels) >= 8.5
        assert all(left.y1 <= right.y0 + 1.0 for left, right in zip(boxes, boxes[1:], strict=False))
    finally:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.close(fig)


def test_sfxi_vec8_aggregate_rejects_missing_vec8_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_vec8.csv"
    _vec8_df(design_prefix="bad", v11=1.0).drop(columns=["v11"]).to_csv(path, index=False)

    with pytest.raises(SFXIError, match="requires columns: v11"):
        load_sfxi_vec8_sources([path])


def test_sfxi_vec8_aggregate_requires_design_id_without_genotype_alias(tmp_path: Path) -> None:
    path = tmp_path / "bad_vec8.csv"
    _vec8_df(design_prefix="bad", v11=1.0).rename(columns={"design_id": "genotype"}).to_csv(path, index=False)

    with pytest.raises(SFXIError, match="requires columns: design_id"):
        load_sfxi_vec8_sources([path])


def test_direct_table_sources_accept_optional_time_metadata(tmp_path: Path) -> None:
    path = tmp_path / "vec8_without_time.csv"
    _vec8_df(design_prefix="valid", v11=1.0).drop(columns=["time_selected_h"]).to_csv(path, index=False)

    aggregate = load_sfxi_vec8_sources([path])

    assert "time_selected_h" not in aggregate.frame.columns
    assert aggregate.frame["design_id"].tolist() == ["valid-01", "valid-02"]


def test_direct_table_sources_accept_nullable_time_metadata(tmp_path: Path) -> None:
    path = tmp_path / "vec8_with_nullable_time.csv"
    frame = _vec8_df(design_prefix="valid", v11=1.0)
    frame.loc[0, "time_selected_h"] = float("nan")
    frame.to_csv(path, index=False)

    aggregate = load_sfxi_vec8_sources([path])

    assert pd.isna(aggregate.frame.loc[0, "time_selected_h"])
    assert aggregate.frame.loc[1, "time_selected_h"] == pytest.approx(18.0)


@pytest.mark.parametrize("missing_column", ["reference_design_id", "r_logic", "flat_logic"])
def test_direct_table_sources_require_vec8_v3_provenance_columns(tmp_path: Path, missing_column: str) -> None:
    path = tmp_path / "bad_vec8.csv"
    _vec8_df(design_prefix="bad", v11=1.0).drop(columns=[missing_column]).to_csv(path, index=False)

    with pytest.raises(SFXIError, match=f"requires columns: {missing_column}"):
        load_sfxi_vec8_sources([path])


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("reference_design_id", "", "non-empty labels"),
        ("intensity_log2_offset_delta", -0.1, "nonnegative values"),
        ("r_logic", float("nan"), "finite numeric values"),
        ("r_logic", -1.0, "nonnegative values"),
        ("flat_logic", "not-a-bool", "boolean values"),
    ],
)
def test_direct_table_sources_validate_vec8_v3_provenance_values(
    tmp_path: Path, column: str, bad_value: object, message: str
) -> None:
    path = tmp_path / "bad_vec8.csv"
    df = _vec8_df(design_prefix="bad", v11=1.0)
    if column == "flat_logic":
        df[column] = df[column].astype(object)
    df.loc[0, column] = bad_value
    df.to_csv(path, index=False)

    with pytest.raises(SFXIError, match=message):
        load_sfxi_vec8_sources([path])


def test_direct_table_sources_reject_duplicate_design_ids(tmp_path: Path) -> None:
    path = tmp_path / "bad_vec8.csv"
    df = _vec8_df(design_prefix="bad", v11=1.0)
    df.loc[1, "design_id"] = df.loc[0, "design_id"]
    df.to_csv(path, index=False)

    with pytest.raises(SFXIError, match="design_id values must be unique within each source"):
        load_sfxi_vec8_sources([path])


def test_direct_workbook_sources_require_vec8_sheet(tmp_path: Path) -> None:
    path = tmp_path / "bad_vec8.xlsx"
    _vec8_df(design_prefix="bad", v11=1.0).to_excel(path, sheet_name="not_vec8", index=False)

    with pytest.raises(SFXIError, match="must include a 'vec8' sheet"):
        load_sfxi_vec8_sources([path])


def test_writer_rejects_nonpositive_dpi(tmp_path: Path) -> None:
    exp_a = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)

    with pytest.raises(SFXIError, match="dpi must be a positive integer"):
        write_sfxi_vec8_aggregate(sources=[exp_a], out_dir=tmp_path / "aggregate", dpi=0)


def test_writer_refuses_to_overwrite_existing_bundle_without_explicit_flag(tmp_path: Path) -> None:
    exp_a = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    out_dir = tmp_path / "aggregate"

    first = write_sfxi_vec8_aggregate(sources=[exp_a], out_dir=out_dir)

    with pytest.raises(SFXIError, match="already exists"):
        write_sfxi_vec8_aggregate(sources=[exp_a], out_dir=out_dir)

    second = write_sfxi_vec8_aggregate(sources=[exp_a], out_dir=out_dir, overwrite=True)
    assert second.heatmap_path == first.heatmap_path


def test_writer_preserves_output_created_during_render_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    exp_a = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    out_dir = tmp_path / "aggregate"
    heatmap_path = out_dir / "sfxi_vec8_heatmap.png"
    concurrent_content = b"published by another writer"
    render = aggregate_writer.render_sfxi_vec8_heatmap

    def _render_after_concurrent_publish(*args, **kwargs):
        heatmap_path.write_bytes(concurrent_content)
        return render(*args, **kwargs)

    monkeypatch.setattr(aggregate_writer, "render_sfxi_vec8_heatmap", _render_after_concurrent_publish)

    with pytest.raises(SFXIError, match="already exists"):
        write_sfxi_vec8_aggregate(sources=[exp_a], out_dir=out_dir)

    assert heatmap_path.read_bytes() == concurrent_content
    assert not (out_dir / "sfxi_vec8_heatmap_tidy.csv").exists()
    assert not (out_dir / "sfxi_vec8_heatmap_manifest.json").exists()


def test_experiment_source_requires_vec8_record_without_export_fallback(tmp_path: Path) -> None:
    root = tmp_path / "20260706_sfxi"
    root.mkdir()
    cfg_path = write_config(
        root,
        base_reader_config(
            experiment_id="20260706_sfxi",
            lifecycle="active",
            protocol_id="logic/sfxi_screen",
            protocol_analysis={"include_vec8": True},
        ),
    )
    export_path = root / "outputs" / "exports" / "sfxi" / "vec8.xlsx"
    export_path.parent.mkdir(parents=True)
    _vec8_df(design_prefix="eth", v11=1.0).to_excel(export_path, sheet_name="vec8", index=False)

    with pytest.raises(SFXIError, match="could not find 'sfxi_vec8/vec8'"):
        load_sfxi_vec8_sources([cfg_path])

    direct = load_sfxi_vec8_sources([export_path])
    assert direct.sources[0].source_kind == "table"
    assert direct.frame["source_id"].tolist() == ["20260706_sfxi", "20260706_sfxi"]


def test_direct_workbook_sources_use_experiment_dir_source_ids(tmp_path: Path) -> None:
    paths = []
    for experiment_id, design_prefix in (("20260706_sfxi", "eth"), ("20260707_sfxi", "and")):
        root = tmp_path / experiment_id
        root.mkdir()
        write_config(
            root,
            base_reader_config(
                experiment_id=experiment_id,
                lifecycle="active",
                protocol_id="logic/sfxi_screen",
                protocol_analysis={"include_vec8": True},
            ),
        )
        export_path = root / "outputs" / "exports" / "sfxi" / "vec8.xlsx"
        export_path.parent.mkdir(parents=True)
        _vec8_df(design_prefix=design_prefix, v11=1.0).to_excel(export_path, sheet_name="vec8", index=False)
        paths.append(export_path)

    aggregate = load_sfxi_vec8_sources(paths)

    assert [source.source_id for source in aggregate.sources] == ["20260706_sfxi", "20260707_sfxi"]
    assert aggregate.frame["row_label"].tolist() == [
        "20260706_sfxi :: eth-01",
        "20260706_sfxi :: eth-02",
        "20260707_sfxi :: and-01",
        "20260707_sfxi :: and-02",
    ]


def test_direct_table_sources_reject_duplicate_source_ids(tmp_path: Path) -> None:
    path_a = tmp_path / "a" / "vec8.csv"
    path_b = tmp_path / "b" / "vec8.csv"
    path_a.parent.mkdir()
    path_b.parent.mkdir()
    _vec8_df(design_prefix="a", v11=1.0).to_csv(path_a, index=False)
    _vec8_df(design_prefix="b", v11=0.5).to_csv(path_b, index=False)

    with pytest.raises(SFXIError, match="source_id values must be unique: vec8"):
        load_sfxi_vec8_sources([path_a, path_b])


def test_cli_aggregate_sfxi_vec8_wraps_stale_record_artifact_errors(tmp_path: Path) -> None:
    cfg_path = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    store = RecordStore(cfg_path.parent / "outputs", contracts=builtin_contract_catalog(), create=False)
    record = store.latest_dataframe("sfxi_vec8/vec8")
    assert record is not None
    record.path.unlink()

    result = CliRunner().invoke(app, ["aggregate-sfxi-vec8", str(cfg_path), "--out-dir", str(tmp_path / "aggregate")])

    assert result.exit_code == 1
    assert "could not load 'sfxi_vec8/vec8' dataframe artifact" in result.output


def test_cli_aggregate_sfxi_vec8_wraps_mutated_record_artifact_errors(tmp_path: Path) -> None:
    cfg_path = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    store = RecordStore(cfg_path.parent / "outputs", contracts=builtin_contract_catalog(), create=False)
    record = store.latest_dataframe("sfxi_vec8/vec8")
    assert record is not None
    _vec8_df(design_prefix="eth", v11=9.9).to_parquet(record.path, index=False)

    result = CliRunner().invoke(app, ["aggregate-sfxi-vec8", str(cfg_path), "--out-dir", str(tmp_path / "aggregate")])

    assert result.exit_code == 1
    assert "could not load 'sfxi_vec8/vec8' dataframe artifact" in result.output


def test_writer_does_not_leave_partial_bundle_when_render_fails(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    out_dir = tmp_path / "aggregate"

    def _fail_render(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("render failed")

    monkeypatch.setattr(aggregate_writer, "render_sfxi_vec8_heatmap", _fail_render)

    with pytest.raises(SFXIError, match="could not write artifact bundle"):
        write_sfxi_vec8_aggregate(sources=[cfg_path], out_dir=out_dir)

    assert list(out_dir.iterdir()) == []


def test_writer_rejects_non_file_existing_targets_before_overwrite(tmp_path: Path) -> None:
    cfg_path = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    out_dir = tmp_path / "aggregate"
    first = write_sfxi_vec8_aggregate(sources=[cfg_path], out_dir=out_dir)
    heatmap_bytes = first.heatmap_path.read_bytes()
    first.tidy_path.unlink()
    first.tidy_path.mkdir()

    with pytest.raises(SFXIError, match="output paths must be files"):
        write_sfxi_vec8_aggregate(sources=[cfg_path], out_dir=out_dir, overwrite=True)

    assert first.heatmap_path.read_bytes() == heatmap_bytes
    assert first.tidy_path.is_dir()


def test_cli_aggregate_sfxi_vec8_writes_json_summary(tmp_path: Path) -> None:
    exp_a = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    exp_b = _write_experiment_with_vec8(tmp_path, experiment_id="20260707_sfxi", design_prefix="and", v11=0.5)
    out_dir = tmp_path / "aggregate"

    result = CliRunner().invoke(
        app,
        [
            "aggregate-sfxi-vec8",
            str(exp_a),
            str(exp_b),
            "--out-dir",
            str(out_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"] == {"sources": 2, "rows": 4, "channels": 8}
    assert Path(payload["artifacts"]["heatmap"]).exists()
    assert Path(payload["artifacts"]["tidy"]).exists()
    assert Path(payload["artifacts"]["manifest"]).exists()


def test_cli_aggregate_sfxi_vec8_requires_overwrite_for_existing_bundle(tmp_path: Path) -> None:
    exp_a = _write_experiment_with_vec8(tmp_path, experiment_id="20260706_sfxi", design_prefix="eth", v11=1.0)
    out_dir = tmp_path / "aggregate"
    runner = CliRunner()
    command = [
        "aggregate-sfxi-vec8",
        str(exp_a),
        "--out-dir",
        str(out_dir),
        "--format",
        "json",
    ]

    first = runner.invoke(app, command)
    second = runner.invoke(app, command)
    third = runner.invoke(app, [*command, "--overwrite"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 1
    assert "already exists" in second.output
    assert third.exit_code == 0, third.output
