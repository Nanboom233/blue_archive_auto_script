# Re-export public API from the internal navigator module
from .navigator import (
    NodeSpec,
    Index,
    compile_specs,
    lookup_next_hop,
    fallback_next_hop,
)

__all__ = [
    "NodeSpec",
    "Index",
    "compile_specs",
    "lookup_next_hop",
    "fallback_next_hop",
]

