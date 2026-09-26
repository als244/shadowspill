"""The operation a backward adds a matrix product into a running gradient with."""

from __future__ import annotations

import torch

#: The ATen overloads that add a product into a tensor in place, at that
#: tensor's dtype, by the tensor's rank: a matrix, or a batch of matrices.
ADDING_INTO = {
    2: torch.ops.aten.addmm.dtype_out,
    3: torch.ops.aten.baddbmm.dtype_out,
}


@torch.library.custom_op(
    "shadowspill::accumulate_matmul_", mutates_args=("accumulator",)
)
def accumulate_matmul_(
    accumulator: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> None:
    """Add ``left @ right`` into ``accumulator`` in place, at its dtype.

    The multiply adds the accumulator as it writes its result -- cuBLAS's
    ``C = A @ B + C`` -- so the product is never written anywhere first and
    read back. ATen's ``addmm`` and ``baddbmm`` do this when their ``out`` is
    their ``self``; this operation exists because the graphs ShadowSpill
    compiles are traced again through decompositions that do not take
    ``out``.
    """

    ADDING_INTO[accumulator.dim()](
        accumulator, left, right, accumulator.dtype, out=accumulator
    )


@accumulate_matmul_.register_fake
def _accumulate_matmul_fake(
    accumulator: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> None:
    del accumulator, left, right


#: The operation as a graph calls it.
ACCUMULATE_MATMUL = torch.ops.shadowspill.accumulate_matmul_.default

__all__ = ["ACCUMULATE_MATMUL", "ADDING_INTO", "accumulate_matmul_"]
