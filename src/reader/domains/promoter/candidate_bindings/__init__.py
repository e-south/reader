"""Public Reader consumer surface for study-issued promoter candidate bindings."""

from .contract import (
    BASERENDER_CONTRACT_ID,
    BASERENDER_CONTRACT_VERSION,
    BINDING_SCHEMA_ID,
    BINDING_SCHEMA_VERSION,
    BINDING_STUDY_ID,
    READER_ALIAS_NAMESPACE,
    PromoterCandidateBinding,
    PromoterCandidateBindings,
)
from .loader import load_promoter_candidate_bindings

__all__ = [
    "BASERENDER_CONTRACT_ID",
    "BASERENDER_CONTRACT_VERSION",
    "BINDING_SCHEMA_ID",
    "BINDING_SCHEMA_VERSION",
    "BINDING_STUDY_ID",
    "READER_ALIAS_NAMESPACE",
    "PromoterCandidateBinding",
    "PromoterCandidateBindings",
    "load_promoter_candidate_bindings",
]
