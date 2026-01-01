from __future__ import annotations

from pathlib import Path

import yaml

from reader.core.config_model import ReaderSpec


def _write_cfg(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_presets_expand_and_override(tmp_path: Path) -> None:
    cfg = {
        "experiment": {"outputs": "./outputs"},
        "overrides": {"ingest": {"with": {"mode": "mixed", "channels": ["OD600"]}}},
        "steps": [
            {"preset": "plate_reader/synergy_h1"},
            {"preset": "plate_reader/sample_map"},
            {
                "id": "aliases",
                "uses": "transform/alias",
                "reads": {"df": "merge_map/df"},
                "with": {"aliases": {"design_id": {}}, "in_place": False, "case_insensitive": True},
            },
        ],
    }
    spec = ReaderSpec.load(_write_cfg(tmp_path / "config.yaml", cfg))
    ids = [s.id for s in spec.steps]
    assert ids == ["ingest", "merge_map", "aliases"]
    ingest = spec.steps[0]
    assert ingest.with_.get("mode") == "mixed"
    assert ingest.with_.get("channels") == ["OD600"]
