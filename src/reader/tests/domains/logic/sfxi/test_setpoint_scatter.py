from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.domains.logic.sfxi.setpoint_scatter import score_sfxi_setpoints
from reader.errors import SFXIError


def _vec8_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": ["p01", "p02"],
            "sequence": ["AAAA", "CCCC"],
            "experiment_id": ["exp_a", "exp_b"],
            "experiment_date": ["20260101", "20260102"],
            "time_selected_h": [10.0, 10.0],
            "intensity_log2_offset_delta": [0.0, 0.0],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 1.0],
            "y00_star": [0.0, 0.0],
            "y10_star": [0.0, 0.0],
            "y01_star": [0.0, 0.0],
            "y11_star": [1.0, 0.0],
            "r_logic": [8.0, 4.0],
            "flat_logic": [False, False],
        }
    )


def _install_fake_dnadesign_api(monkeypatch) -> None:
    class _FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeResult:
        api_version = "1"
        objective_name = "sfxi_v1"

        def to_records(self):
            return [
                {
                    "objective_name": self.objective_name,
                    "api_version": self.api_version,
                    "state_order": ["00", "10", "01", "11"],
                    "setpoint_vector": [0.0, 0.0, 0.0, 1.0],
                    "denom_percentile": 95,
                    "denom_used": 2.0,
                    "logic_fidelity": 1.0,
                    "effect_raw": 2.0,
                    "effect_scaled": 1.0,
                    "sfxi": 1.0,
                    "clip_lo_mask": False,
                    "clip_hi_mask": True,
                    "intensity_disabled": False,
                },
                {
                    "objective_name": self.objective_name,
                    "api_version": self.api_version,
                    "state_order": ["00", "10", "01", "11"],
                    "setpoint_vector": [0.0, 0.0, 0.0, 1.0],
                    "denom_percentile": 95,
                    "denom_used": 2.0,
                    "logic_fidelity": 1.0,
                    "effect_raw": 1.0,
                    "effect_scaled": 0.5,
                    "sfxi": 0.5,
                    "clip_lo_mask": False,
                    "clip_hi_mask": False,
                    "intensity_disabled": False,
                },
            ]

    fake_api = SimpleNamespace(
        SFXI_API_VERSION="1",
        SFXIScoringConfig=_FakeConfig,
        score_vec8=lambda *args, **kwargs: _FakeResult(),
    )
    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None):
        if name == "dnadesign.opal.api.sfxi":
            return fake_api
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)


def test_score_sfxi_setpoints_uses_canonical_objective_names(monkeypatch) -> None:
    _install_fake_dnadesign_api(monkeypatch)

    scored = score_sfxi_setpoints(
        _vec8_df(),
        setpoints={"and": [0.0, 0.0, 0.0, 1.0]},
        scaling_min_n=1,
    )

    assert list(scored["design_id"]) == ["p01", "p02"]
    assert list(scored["experiment_id"]) == ["exp_a", "exp_b"]
    assert list(scored["experiment_date"]) == ["20260101", "20260102"]
    assert list(scored["intensity_log2_offset_delta"]) == [0.0, 0.0]
    assert list(scored["setpoint_name"]) == ["and", "and"]
    assert scored["logic_fidelity"].tolist() == pytest.approx([1.0, 1.0])
    assert scored["effect_scaled"].tolist() == pytest.approx([1.0, 0.5])
    assert scored["sfxi"].tolist() == pytest.approx([1.0, 0.5])
    assert "score" not in scored.columns
    assert "f_logic" not in scored.columns
    assert "e_scaled" not in scored.columns


def test_score_sfxi_setpoints_preserves_aggregate_source_metadata(monkeypatch) -> None:
    _install_fake_dnadesign_api(monkeypatch)
    vec8 = _vec8_df()
    vec8["source_id"] = ["exp_a_source", "exp_b_source"]
    vec8["source_path"] = ["/tmp/exp_a/config.yaml", "/tmp/exp_b/config.yaml"]
    vec8["table_path"] = ["/tmp/exp_a/vec8.parquet", "/tmp/exp_b/vec8.parquet"]
    vec8["source_kind"] = ["record", "record"]
    vec8["source_row_index"] = [0, 1]
    vec8["row_label"] = ["exp_a :: p01", "exp_b :: p02"]

    scored = score_sfxi_setpoints(
        vec8,
        setpoints={"and": [0.0, 0.0, 0.0, 1.0]},
        scaling_min_n=1,
    )

    assert list(scored["source_id"]) == ["exp_a_source", "exp_b_source"]
    assert list(scored["source_kind"]) == ["record", "record"]
    assert list(scored["source_row_index"]) == [0, 1]
    assert list(scored["row_label"]) == ["exp_a :: p01", "exp_b :: p02"]


def test_score_sfxi_setpoints_reports_missing_public_dnadesign_api(monkeypatch) -> None:
    real_import = importlib.import_module

    def _blocked(name: str, package: str | None = None):
        if name == "dnadesign.opal.api.sfxi":
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _blocked)

    with pytest.raises(SFXIError, match=r"reader\[dnadesign\]"):
        score_sfxi_setpoints(_vec8_df(), setpoints={"and": [0.0, 0.0, 0.0, 1.0]}, scaling_min_n=1)


def test_score_sfxi_setpoints_backfills_missing_intensity_delta_provenance(monkeypatch) -> None:
    _install_fake_dnadesign_api(monkeypatch)
    vec8 = _vec8_df().drop(columns=["intensity_log2_offset_delta"])

    scored = score_sfxi_setpoints(vec8, setpoints={"and": [0.0, 0.0, 0.0, 1.0]}, scaling_min_n=1)

    assert list(scored["intensity_log2_offset_delta"]) == [0.0, 0.0]


def test_score_sfxi_setpoints_rejects_missing_intensity_delta_when_expected_nonzero() -> None:
    vec8 = _vec8_df().drop(columns=["intensity_log2_offset_delta"])

    with pytest.raises(SFXIError, match="intensity_log2_offset_delta mismatch"):
        score_sfxi_setpoints(
            vec8,
            setpoints={"and": [0.0, 0.0, 0.0, 1.0]},
            scaling_min_n=1,
            intensity_log2_offset_delta=0.25,
        )


def test_score_sfxi_setpoints_rejects_intensity_delta_mismatch() -> None:
    vec8 = _vec8_df()
    vec8["intensity_log2_offset_delta"] = [0.0, 0.25]

    with pytest.raises(SFXIError, match="intensity_log2_offset_delta mismatch"):
        score_sfxi_setpoints(
            vec8,
            setpoints={"and": [0.0, 0.0, 0.0, 1.0]},
            scaling_min_n=1,
            intensity_log2_offset_delta=0.0,
        )


def test_score_sfxi_setpoints_wraps_transitive_public_api_import_failures(monkeypatch) -> None:
    def _blocked(name: str, package: str | None = None):
        if name == "dnadesign.opal.api.sfxi":
            raise ImportError("missing transitive dependency")
        raise AssertionError((name, package))

    monkeypatch.setattr(importlib, "import_module", _blocked)

    with pytest.raises(SFXIError, match=r"reader\[dnadesign\]") as exc_info:
        score_sfxi_setpoints(_vec8_df(), setpoints={"and": [0.0, 0.0, 0.0, 1.0]}, scaling_min_n=1)

    assert isinstance(exc_info.value.__cause__, ImportError)


def test_score_sfxi_setpoints_rejects_unsupported_public_api_version(monkeypatch) -> None:
    fake_api = SimpleNamespace(SFXI_API_VERSION="2", SFXIScoringConfig=object, score_vec8=lambda *args, **kwargs: None)
    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None):
        if name == "dnadesign.opal.api.sfxi":
            return fake_api
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    with pytest.raises(SFXIError, match="Unsupported dnadesign SFXI API version"):
        score_sfxi_setpoints(_vec8_df(), setpoints={"and": [0.0, 0.0, 0.0, 1.0]}, scaling_min_n=1)
