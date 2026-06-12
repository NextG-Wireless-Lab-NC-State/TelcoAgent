"""3GPP ontology + KG loader."""

from . import kg_loader
from .core import (
    CausalChain,
    DirectionalRule,
    KPIDefinition,
    KPIRelationship,
    ThreeGPPOntology,
    get_default_ontology,
)

__all__ = [
    "ThreeGPPOntology",
    "KPIDefinition",
    "KPIRelationship",
    "CausalChain",
    "DirectionalRule",
    "get_default_ontology",
    "kg_loader",
]
