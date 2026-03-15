from __future__ import annotations

import pytest

from reader.core.errors import ConfigError
from reader.core.presets.registry import _build_preset_registry


def test_build_preset_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ConfigError, match="Duplicate preset 'shared/preset'"):
        _build_preset_registry(
            ("first", {"shared/preset": {"description": "one", "steps": []}}),
            ("second", {"shared/preset": {"description": "two", "steps": []}}),
        )
