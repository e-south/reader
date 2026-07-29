from __future__ import annotations

from typing import Any

from reader.protocols.model import CompiledProtocolPlan

from .common import default_notebook_call


def compile_generic_protocol(protocol: Any):
    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        pipeline=(),
        plots=(),
        exports=(),
        notebooks=(default_notebook_call(template),),
        semantic_program=protocol.descriptor.semantic_program(),
    )
