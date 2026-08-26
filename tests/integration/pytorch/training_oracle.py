"""Explicit execution of one captured objective forward/backward graph pair."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import AotGraphPair, ObjectiveSchema


@dataclass(frozen=True, slots=True)
class ObjectivePairResult:
    """Tensor results and input-aligned gradients from one microbatch."""

    loss: torch.Tensor
    metrics: object
    gradients: tuple[torch.Tensor | None, ...]
    saved_values: tuple[torch.Tensor, ...]


class ObjectivePairExecutor:
    """Execute explicit AOT forward/backward functions without autograd state."""

    def __init__(
        self,
        pair: AotGraphPair,
        schema: ObjectiveSchema,
        forward: Callable[..., object] | None = None,
        backward: Callable[..., object] | None = None,
    ) -> None:
        self.pair = pair
        self.schema = schema
        self.forward = forward or pair.forward.graph_module
        self.backward = backward or pair.backward.graph_module
        self._public_tensor_count = 1 + len(schema.tensor_metric_positions)
        if (
            pair.forward.output_count
            != self._public_tensor_count + pair.saved_value_count
        ):
            raise CaptureError(
                "AOT forward output count differs from objective and residual contract"
            )

    def __call__(self, arguments: Sequence[object]) -> ObjectivePairResult:
        if len(arguments) != self.pair.forward.argument_count:
            raise CaptureError(
                "AOT forward argument count changed: "
                f"expected {self.pair.forward.argument_count}, got {len(arguments)}"
            )
        raw_forward = self.forward(*arguments)
        forward_leaves, _ = tree_flatten(raw_forward)
        if len(forward_leaves) != self.pair.forward.output_count:
            raise CaptureError("AOT forward output count changed during execution")
        public = forward_leaves[: self._public_tensor_count]
        if any(not isinstance(value, torch.Tensor) for value in public):
            raise CaptureError("AOT objective outputs must remain tensors")
        loss = public[0]
        assert isinstance(loss, torch.Tensor)
        metric_tensors = tuple(public[1:])
        assert all(isinstance(value, torch.Tensor) for value in metric_tensors)
        saved = tuple(forward_leaves[self._public_tensor_count :])
        if any(not isinstance(value, torch.Tensor) for value in saved):
            raise CaptureError("AOT saved values must remain tensors")
        backward_arguments: tuple[torch.Tensor, ...]
        if self.pair.specialized_unit_tangent_count:
            if self.pair.specialized_unit_tangent_count != 1:
                raise CaptureError("objective graph has multiple specialized tangents")
            backward_arguments = saved
        else:
            backward_arguments = (*saved, torch.ones_like(loss))
        raw_backward = self.backward(*backward_arguments)
        gradients, _ = tree_flatten(raw_backward)
        if len(gradients) != self.pair.backward.output_count:
            raise CaptureError("AOT backward output count changed during execution")
        if any(
            value is not None and not isinstance(value, torch.Tensor)
            for value in gradients
        ):
            raise CaptureError("AOT backward returned a non-tensor gradient")
        return ObjectivePairResult(
            loss=loss,
            metrics=self.schema.rebuild_metrics(metric_tensors),
            gradients=tuple(gradients),
            saved_values=saved,
        )


__all__ = ["ObjectivePairExecutor", "ObjectivePairResult"]
