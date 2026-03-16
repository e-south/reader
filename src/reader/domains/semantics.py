from __future__ import annotations

from typing import Literal

PluginDomain = Literal["generic", "plate_reader", "cytometry", "logic"]
KNOWN_PLUGIN_DOMAINS: tuple[PluginDomain, ...] = ("generic", "plate_reader", "cytometry", "logic")


def validate_plugin_domain(domain: str) -> PluginDomain:
    normalized = str(domain).strip()
    if normalized in KNOWN_PLUGIN_DOMAINS:
        return normalized  # type: ignore[return-value]
    options = ", ".join(KNOWN_PLUGIN_DOMAINS)
    raise ValueError(f"Unknown plugin domain '{domain}'. Expected one of: {options}")


__all__ = ["KNOWN_PLUGIN_DOMAINS", "PluginDomain", "validate_plugin_domain"]
