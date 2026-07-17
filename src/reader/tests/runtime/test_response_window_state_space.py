from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from reader.runtime import response_window as runtime_module


def _resolved(*, state_ids: tuple[str, ...] = ("00", "10", "01", "11"), case_sensitive: bool = False):
    return SimpleNamespace(
        state_ids=state_ids,
        column="treatment",
        source_values={state_id: f"source-{state_id}" for state_id in state_ids},
        case_sensitive=case_sensitive,
    )


def test_response_window_accepts_only_its_declared_four_state_contract() -> None:
    state_space = runtime_module._require_response_window_state_space(_resolved())

    assert state_space.state_ids == ("00", "10", "01", "11")


@pytest.mark.parametrize(
    "state_ids",
    [
        ("00", "10", "01"),
        ("00", "10", "01", "11", "extra"),
        ("00", "01", "10", "11"),
    ],
)
def test_response_window_rejects_missing_extra_or_reordered_states(state_ids: tuple[str, ...]) -> None:
    with pytest.raises(
        ValueError, match="response-window state space must declare exactly 00, 10, 01, 11 in that order"
    ):
        runtime_module._require_response_window_state_space(_resolved(state_ids=state_ids))


def test_response_window_resolves_metric_neutral_ordered_state_space(monkeypatch, tmp_path: Path) -> None:
    resolved = _resolved(case_sensitive=False)

    class Annotations:
        def resolve_ordered_state_space(self, *, ref: str):
            assert ref == "stress_states"
            return resolved

    declaration = SimpleNamespace(
        experiment=SimpleNamespace(id="20260001_test"),
        experiment_semantics=SimpleNamespace(
            annotations=Annotations(),
            layout=SimpleNamespace(outputs_dir=tmp_path / "outputs"),
        ),
    )
    experiment_dir = tmp_path / "experiments" / "2026" / "20260001_test"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text("schema: reader/v8\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "load_workbench_decl", lambda *_args, **_kwargs: declaration)

    class Record:
        path = tmp_path / "record.parquet"
        contract_id = "plate_reader.annotated.v1"
        content_digest = "digest"

        def verify_content_digest(self) -> None:
            return None

    class Store:
        records_path = tmp_path / "records.json"

        def catalog_exists(self) -> bool:
            return True

        def read_dataframe(self, _record_id: str) -> Record:
            return Record()

    runtime = SimpleNamespace(
        protocols=SimpleNamespace(),
        contracts=SimpleNamespace(),
        record_store=lambda *_args, **_kwargs: Store(),
    )
    source_spec = SimpleNamespace(
        state_map_ref="stress_states",
        response_record_id="response",
        magnitude_record_id="magnitude",
        trajectory_record_id="trajectory",
    )
    captured = {}

    def fake_load_experiment_source(resolved_source, **_kwargs):
        captured["resolved"] = resolved_source
        return SimpleNamespace()

    monkeypatch.setattr(runtime_module, "load_experiment_source", fake_load_experiment_source)

    runtime_module._load_source(
        "20260001_test",
        source_spec,
        SimpleNamespace(),
        reader_root=tmp_path,
        runtime=runtime,
    )

    assert captured["resolved"].state_column == "treatment"
    assert captured["resolved"].treatment_map == resolved.source_values
    assert captured["resolved"].state_values_case_sensitive is False
