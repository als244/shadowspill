"""Serial ready-frame consumption for the existing Executor input_loader API."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Self

from shadowspill.training.inputs.spool import SpooledFrame
from shadowspill.training.preparation.capture.binding import (
    OperationBinding,
    control_tensors,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.controls import Scalar
from shadowspill_client.protocol.payload import BatchSnapshot


class ReadyInputLoader:
    """Keep one decoded physical frame until its synchronous spill copy finishes.

    This runs in the worker. Model/content preprocessing already finished on the
    controller. Restored CPU views are read-only sources for guards and spill
    copies; neither model execution nor mutable user code receives these views.
    Remote workers supply their own admitted node-local read budget.
    """

    def __init__(
        self,
        frames: Sequence[tuple[SpooledFrame, OperationBinding]],
        loss_params: Mapping[str, Scalar],
        budget: BufferBudget,
    ) -> None:
        self.budget = budget
        self.loss_params = loss_params
        self._frames = tuple(frames)
        if not self._frames:
            raise ValueError("worker rank has no prepared input frames")
        self._snapshot: BatchSnapshot | None = None
        self._next = 0
        self.closed = False

    def __call__(self, index: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
        import torch

        if self.closed:
            raise RuntimeError("ready input loader is closed")
        if type(index) is not int or index != self._next or index >= len(self._frames):
            raise ValueError("consume prepared input frames exactly once in order")
        if self._snapshot is not None:
            self._snapshot.close()
            self._snapshot = None
        frame, binding = self._frames[index]
        snapshot = self._snapshot = frame.read(self.budget)
        arrays = frame.binding.frame_arrays(snapshot)
        trees: dict[str, dict[str, Any]] = {"inputs": {}, "loss_inputs": {}}
        # The immutable snapshot is deliberately borrowed for read-only input
        # staging. Executor copies every consumed value before advancing this
        # loader. Never hand these tensors to an eager mutable computation.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable.*",
                category=UserWarning,
            )
            for path, array in sorted(arrays.items()):
                tensor = torch.from_numpy(array)
                if frame.binding.dtypes[path] == "bfloat16":
                    tensor = tensor.view(torch.bfloat16)
                tree = trees[path[0]]
                for name in path[1:-1]:
                    tree = tree.setdefault(name, {})
                tree[path[-1]] = tensor
        controls = control_tensors(binding.contract.controls, self.loss_params)
        self._next += 1
        return (trees["inputs"], trees["loss_inputs"], controls), {}

    def close(self) -> None:
        if self._snapshot is not None:
            self._snapshot.close()
            self._snapshot = None
        self._frames = ()
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
