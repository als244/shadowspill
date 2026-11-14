"""Which of a stage's outputs a backward can be taken through.

One rule, in one place, because two of them decide the same question: the
automatic partition folds away a stage that has no differentiable output,
and stage capture refuses one that reaches it anyway. A second copy of the
rule would let the two disagree, and the disagreement would read as a model
that cannot be trained.
"""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten


def differentiable_output_positions(output: object) -> tuple[int, ...]:
    """Return the flattened output positions a gradient can flow back through.

    A value qualifies when it carries a gradient and is continuous. Integer
    and boolean values are control: they select kernels and index memory, and
    nothing differentiates them.
    """

    leaves, _ = tree_flatten(output)
    return tuple(
        position
        for position, value in enumerate(leaves)
        if isinstance(value, torch.Tensor)
        and value.requires_grad
        and (value.is_floating_point() or value.is_complex())
    )


__all__ = ["differentiable_output_positions"]
