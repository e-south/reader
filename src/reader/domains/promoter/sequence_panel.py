"""Metric-neutral Reader adapter for public BaseRender promoter sequence panels."""

from __future__ import annotations

import importlib
from typing import Any

from reader.domains.promoter.candidate_bindings import (
    BASERENDER_CONTRACT_ID,
    BASERENDER_CONTRACT_VERSION,
    PromoterCandidateBinding,
)


class PromoterSequencePanelError(RuntimeError):
    """Raised when the public BaseRender contract cannot render a bound candidate."""


def require_baserender_api() -> Any:
    """Return the compatible public BaseRender module without importing study internals."""

    try:
        module = importlib.import_module("dnadesign.baserender")
    except ImportError as exc:
        raise PromoterSequencePanelError(
            "Promoter sequence panels require the public dnadesign BaseRender API; "
            "run `uv sync --locked --group dnadesign`."
        ) from exc
    if (
        getattr(module, "BASERENDER_SEQUENCE_PANEL_CONTRACT_ID", None) != BASERENDER_CONTRACT_ID
        or str(getattr(module, "BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION", None)) != BASERENDER_CONTRACT_VERSION
        or not callable(getattr(module, "render_sequence_panel_image", None))
    ):
        raise PromoterSequencePanelError(
            "Promoter sequence panels require the supported public BaseRender sequence-panel contract."
        )
    return module


def render_candidate_sequence_panel(
    binding: PromoterCandidateBinding,
    *,
    style_profile: str,
    style_overrides: dict[str, Any] | None = None,
    target_width_px: int,
    target_height_px: int,
    vertical_anchor: str,
    canvas_top_pad_px: int,
) -> Any:
    """Render exactly the sequence record projected by the study-owned binding artifact."""

    baserender = require_baserender_api()
    rendered = baserender.render_sequence_panel_image(
        binding.baserender_record,
        adapter_kind=binding.baserender_adapter_kind,
        style_profile=style_profile,
        style_overrides=dict(style_overrides or {}),
        target_width_px=target_width_px,
        target_height_px=target_height_px,
        vertical_anchor=vertical_anchor,
        canvas_top_pad_px=canvas_top_pad_px,
    )
    diagnostics = getattr(rendered, "diagnostics", None)
    if (
        getattr(diagnostics, "contract_id", None) != BASERENDER_CONTRACT_ID
        or str(getattr(diagnostics, "contract_version", None)) != BASERENDER_CONTRACT_VERSION
        or getattr(diagnostics, "adapter_kind", None) != binding.baserender_adapter_kind
        or int(getattr(diagnostics, "sequence_length_bp", -1)) != len(binding.canonical_sequence)
    ):
        raise PromoterSequencePanelError(
            "BaseRender sequence-panel diagnostics disagree with the study-issued candidate binding."
        )
    return rendered


__all__ = [
    "PromoterSequencePanelError",
    "render_candidate_sequence_panel",
    "require_baserender_api",
]
