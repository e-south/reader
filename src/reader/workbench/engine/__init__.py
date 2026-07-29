from __future__ import annotations

from reader.workbench.registry import load_plugin_catalog

from .contracts import (
    _assert_input_ports,
    _assert_output_ports,
    _resolve_output_labels,
    _resolve_runtime_output_ports,
)
from .execution import execute_step, run_steps
from .inputs import _resolve_inputs
from .invocations import ExecutionResult, ProducedRecordRevision, SelectedSteps
from .planning import build_next_steps, explain
from .runtime import run_job, run_spec
from .setup import build_run_context, configure_logger, normalize_log_level, resolve_palette_book, slice_pipeline_steps
from .validation import validate, validation_summary

__all__ = [
    "_assert_input_ports",
    "_assert_output_ports",
    "_resolve_inputs",
    "_resolve_output_labels",
    "_resolve_runtime_output_ports",
    "build_run_context",
    "build_next_steps",
    "configure_logger",
    "ExecutionResult",
    "execute_step",
    "explain",
    "load_plugin_catalog",
    "normalize_log_level",
    "ProducedRecordRevision",
    "resolve_palette_book",
    "run_job",
    "run_steps",
    "run_spec",
    "SelectedSteps",
    "slice_pipeline_steps",
    "validate",
    "validation_summary",
]
