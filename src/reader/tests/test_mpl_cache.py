import os
from pathlib import Path

import pytest

from reader.core.errors import ConfigError
from reader.core.mpl import ensure_mpl_cache_dir


def test_ensure_mpl_cache_dir_env_points_to_file(tmp_path, monkeypatch):
    bad = tmp_path / "not_a_dir"
    bad.write_text("x", encoding="utf-8")
    monkeypatch.setenv("MPLCONFIGDIR", str(bad))
    with pytest.raises(ConfigError):
        ensure_mpl_cache_dir()


def test_ensure_mpl_cache_dir_creates_reader_env(tmp_path, monkeypatch):
    target = tmp_path / "mpl"
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setenv("READER_MPLCONFIGDIR", str(target))
    path = ensure_mpl_cache_dir()
    assert Path(os.environ["MPLCONFIGDIR"]) == target
    assert path == target
    assert target.is_dir()
