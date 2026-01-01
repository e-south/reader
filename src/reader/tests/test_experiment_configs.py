from __future__ import annotations

from pathlib import Path

import pytest

from reader.core.config_model import ReaderSpec


def test_experiment_configs_load() -> None:
    root = Path(__file__).resolve().parents[3]
    exp_dir = root / "experiments"
    if not exp_dir.exists():
        pytest.skip("experiments/ not present in this checkout")

    configs = sorted(exp_dir.rglob("config.yaml"))
    if not configs:
        pytest.skip("no experiment configs found")

    for cfg in configs:
        try:
            ReaderSpec.load(cfg)
        except Exception as e:
            raise AssertionError(f"Failed to load config: {cfg}\n{e}") from e
