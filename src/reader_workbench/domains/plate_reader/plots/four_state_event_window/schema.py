"""Shared four-state event-window plot coordinates."""

STATE_ORDER = ("00", "10", "01", "11")
RESPONSE_COLUMNS = tuple(f"r{state}" for state in STATE_ORDER)
MAGNITUDE_COLUMNS = tuple(f"b{state}" for state in STATE_ORDER)
COMPONENT_COLUMNS = (*RESPONSE_COLUMNS, *MAGNITUDE_COLUMNS)

__all__ = ["COMPONENT_COLUMNS", "MAGNITUDE_COLUMNS", "RESPONSE_COLUMNS", "STATE_ORDER"]
