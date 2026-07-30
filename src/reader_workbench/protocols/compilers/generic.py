from __future__ import annotations

from typing import Any

from reader_workbench.protocols.model import CompiledProtocolPlan


def compile_generic_protocol(protocol: Any):
    return CompiledProtocolPlan(
        pipeline=(),
        plots=(),
        exports=(),
        semantic_program=protocol.descriptor.semantic_program(),
    )
