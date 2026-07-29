"""Public parser surface for BioTek Synergy H1 workbooks."""

from ._parser import parse_kinetic_only, parse_snapshot_and_timeseries
from ._shared import probe_synergy_workbook

__all__ = ["parse_kinetic_only", "parse_snapshot_and_timeseries", "probe_synergy_workbook"]
