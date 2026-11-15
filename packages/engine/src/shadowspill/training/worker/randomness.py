"""Bounded default training/evaluation RNG streams for one admitted worker.

Model-owned generators remain explicit registered state. These default Python,
NumPy and Torch generators are process-local, with logical identities independent
of the physical CUDA ordinal. Checkpoint metadata uses the existing value codec.
"""

import json
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from shadowspill.artifacts.state import StateId
from shadowspill.checkpoint import metadata_values
from shadowspill.checkpoint.random_streams import (
    RandomState,
    RandomStream,
    map_random_streams,
    read_snapshot,
    snapshot,
)
from shadowspill.pytorch._rng import restore_rng, rng_state


class WorkerRandomness:
    """One training stream set and one separate advancing evaluation stream set.

    Construction is inert. Fresh initialization or restore must finish before
    invocation. Restore validates both complete sets using isolated generators
    before touching the active defaults. Capture/restore happen only at quiescent
    worker boundaries; the caller owns synchronization and memory admission.
    """

    def __init__(self, device: torch.device, logical_rank: int) -> None:
        if device.type not in ("cpu", "cuda"):
            raise ValueError("default worker RNG supports CPU or CUDA")
        self.device = device
        representations = {
            "python": "python.random@3",
            "numpy": "numpy.MT19937@1",
            "torch": "torch.cpu@" + torch.__version__,
        }
        if device.type == "cuda":
            representations["accelerator"] = "torch.cuda@" + torch.__version__
        self.streams = tuple(
            RandomStream(
                StateId("rng", "worker.default", (str(logical_rank), purpose, name)),
                representation,
                logical_rank,
            )
            for purpose in ("training", "evaluation")
            for name, representation in representations.items()
        )
        self.root_seed: int | None = None
        self._evaluation: dict[str, Any] | None = None
        self._evaluating = False

    def initialize(self, root_seed: int) -> None:
        if self.root_seed is not None:
            raise RuntimeError("worker random streams are already initialized")
        mapped = map_random_streams(
            [],
            list(self.streams),
            root_seed=root_seed,
            restore_id="fresh-default-streams-v1",
            initializers={stream.representation for stream in self.streams},
        )
        self.restore(snapshot(root_seed, mapped.states))

    def _encoded(self, purpose: str, values: Mapping[str, Any]) -> list[RandomState]:
        return [
            RandomState(
                stream,
                json.dumps(
                    metadata_values.encode(values[stream.identity.component_path[-1]]),
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
            for stream in self.streams
            if stream.identity.component_path[-2] == purpose
        ]

    def capture(self) -> dict[str, Any]:
        if self.root_seed is None or self._evaluation is None or self._evaluating:
            raise RuntimeError("RNG capture requires a ready training boundary")
        return snapshot(
            self.root_seed,
            self._encoded("training", rng_state(self.device))
            + self._encoded("evaluation", self._evaluation),
        )

    def _decode(self, state: RandomState) -> Any:
        """Validate state in independent generators, without advancing defaults."""
        backend = state.stream.identity.component_path[-1]
        value: Any = (
            None
            if state.payload is None
            else metadata_values.decode(json.loads(state.payload))
        )
        if backend == "python":
            python = random.Random(0 if state.seed is None else state.seed)
            if state.seed is None:
                python.setstate(value)
            return python.getstate()
        if backend == "numpy":
            numpy = np.random.RandomState(
                0 if state.seed is None else state.seed % (1 << 32)
            )
            if state.seed is None:
                numpy.set_state(value)
            return numpy.get_state()
        if backend not in ("torch", "accelerator"):
            raise ValueError("unknown default generator backend")
        generator = torch.Generator(
            device=self.device if backend == "accelerator" else "cpu"
        )
        if state.seed is None:
            if not isinstance(value, torch.Tensor) or (
                value.dtype != torch.uint8
                or value.device.type != "cpu"
                or value.ndim != 1
                or not value.is_contiguous()
            ):
                raise ValueError("Torch generator state requires contiguous CPU bytes")
            generator.set_state(value)
        else:
            generator.manual_seed(state.seed)
        return generator.get_state()

    def prepare(
        self, value: Mapping[str, Any]
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Validate a complete checkpoint without mutating any active RNG stream."""
        root_seed, states = read_snapshot(value)
        if {state.stream for state in states} != set(self.streams):
            raise ValueError("RNG snapshot requires logical stream conversion")
        training: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        for state in states:
            purpose, name = state.stream.identity.component_path[-2:]
            target = training if purpose == "training" else evaluation
            target[name] = self._decode(state)
        return root_seed, training, evaluation

    def restore(self, value: Mapping[str, Any]) -> None:
        if self._evaluating:
            raise RuntimeError("cannot restore during evaluation")
        root_seed, training, evaluation = self.prepare(value)
        restore_rng(training, self.device)
        self.root_seed, self._evaluation = root_seed, evaluation

    @contextmanager
    def evaluation(self) -> Iterator[None]:
        """Advance evaluation randomness while preserving every training stream."""
        if self._evaluation is None or self._evaluating:
            raise RuntimeError("evaluation requires an initialized idle RNG owner")
        training = rng_state(self.device)
        self._evaluating = True
        try:
            restore_rng(self._evaluation, self.device)
            yield
        finally:
            try:
                self._evaluation = rng_state(self.device)
            finally:
                restore_rng(training, self.device)
                self._evaluating = False
