"""Stage-local gradient graph transformations, separate from AOT capture."""

from .accumulation import accumulate_gradient_outputs
from .precision import cast_gradient_outputs
from .storage import materialize_gradient_outputs

__all__ = [
    "accumulate_gradient_outputs",
    "cast_gradient_outputs",
    "materialize_gradient_outputs",
]
