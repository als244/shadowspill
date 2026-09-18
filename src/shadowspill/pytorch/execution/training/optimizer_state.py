"""The optimizer's state as the plan holds it: whether it exists yet, how a
checkpoint reads and writes it through the spill pool, and how the plan lets it
go."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch.utils._pytree import tree_map

from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
)
from shadowspill.pytorch.materialization.training import TrainingMaterializedState
from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
    current_optimizer_bindings,
    opaque_optimizer_outputs,
    restore_optimizer_checkpoint_structure,
)
from shadowspill.pytorch.runtime_adapter.boundaries import PublishedStorage
from shadowspill.pytorch.spill import (
    read_spill_tensor,
    write_spill_tensor,
)
from shadowspill.pytorch.state.optimizer import release_optimizer_state_from_plan
from shadowspill.runtime.plan import (
    RuntimeBridge,
)

from ..records import (
    ExecutionTaskRecord as _ExecutionTaskRecord,
)
from .values import ExposedOptimizerTensor, TensorLayout


class OptimizerState:
    """The optimizer and the plan's account of its state.

    `initialized` says whether the state the optimizer keeps exists yet -- a
    lazy optimizer creates it on its first step, which the initial plan runs;
    `available` whether the traced update may run over it. Both are the
    executor's to read at a task boundary and this class's to change.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        state: TrainingMaterializedState,
        bridge: RuntimeBridge,
        lowered: LoweredTrainingProgram,
        *,
        has_initial_plan: bool,
        was_lazy: bool,
        preinitialized: bool,
    ) -> None:
        self.optimizer = optimizer
        self._state = state
        self._bridge = bridge
        self._lowered = lowered
        self._has_initial_plan = has_initial_plan
        self.initialized = not was_lazy
        self.available = preinitialized or not was_lazy
        self._size_by_alias = {
            item.alias_group_id: item.size_bytes
            for item in lowered.program.alias_groups
        }

    def set_initialized(self, value: bool) -> None:
        """Select the recurrent plan after a checkpoint restores lazy state."""

        if value and not self._has_initial_plan:
            self.initialized = True
            self.available = True
            return
        self.initialized = value
        self.available = value

    def state_dict(self) -> dict[str, object]:
        """Synchronously snapshot optimizer state without stale CUDA pointers.

        The snapshot is independent: each tensor is its own compact host
        allocation, aliasing neither runtime storage nor the other entries,
        so a caller can serialize it while training continues. The pool keeps
        the authoritative copy throughout and every alias is read out of it
        into a buffer, so an alias costs two until the snapshot is built. On a
        large model that is the biggest transient the frontend asks for, so
        budget for it.
        """

        if not self.initialized:
            raw = self.optimizer.state_dict()
            return {
                "state": {},
                "param_groups": copy.deepcopy(raw["param_groups"]),
            }

        exposed = self.expose_cpu()
        try:
            raw = self.optimizer.state_dict()
            return cast(
                dict[str, object],
                tree_map(
                    lambda value: (
                        value.detach().cpu().clone()
                        if isinstance(value, torch.Tensor)
                        else copy.deepcopy(value)
                    ),
                    raw,
                ),
            )
        finally:
            self.restore_spill_only(exposed)

    def load(self, value: Mapping[str, object]) -> bool:
        """Restore optimizer metadata and write tensor bytes into spill storage."""

        exposed = self.expose_cpu()
        initialized = False
        try:
            restored = restore_optimizer_checkpoint_structure(
                dict(self._state.model.named_parameters()),
                self.optimizer,
                value,
            )
            tensors = {item.name: item for item in restored.tensors}
            current = self.current_bindings()
            planned = self._lowered.optimizer_objects
            present = {item.name for item in planned if item.name in current}
            required_created = {
                item.name for item in planned if item.created_on_first_step
            }
            if present and present != {item.name for item in planned}:
                missing = sorted({item.name for item in planned} - present)
                raise RuntimeError(
                    f"optimizer checkpoint has incomplete planned state: {missing}"
                )
            initialized = not required_created or (
                restored.initialized and required_created.issubset(present)
            )
            if initialized:
                self._write_restored_tensors(
                    planned,
                    current,
                    tensors,
                )
        finally:
            self.restore_spill_only(exposed)

        if not initialized:
            aliases = tuple(
                self._bridge.objects.alias_for_object(item.object_id)
                for item in planned
            )
            self._bridge.objects.unregister(aliases)
            for item in planned:
                alias_id = self._bridge.objects.alias_for_object(item.object_id)
                self._state.object_store.pop(alias_id, None)
                self._state.object_tensors.pop(item.object_id, None)
            self.initialized = False
            return False

        self.initialized = True
        return True

    def _write_restored_tensors(
        self,
        planned: Sequence[Any],
        current: Mapping[str, Any],
        tensors: Mapping[str, Any],
    ) -> None:
        """Copy a checkpoint into existing spill-backed optimizer aliases."""

        for name, restored in tensors.items():
            destination = restored.destination
            source = restored.source.detach()
            if destination.device.type != "cpu":
                raise RuntimeError(
                    f"optimizer checkpoint destination {name!r} is not CPU exposed"
                )
            destination.copy_(source.to(device="cpu"))
        written: set[str] = set()
        for item in planned:
            restored = tensors.get(item.name)
            actual = current.get(item.name)
            if restored is None or actual is None:
                raise RuntimeError(
                    f"optimizer checkpoint lacks planned tensor {item.name!r}"
                )
            alias_id = self._bridge.objects.alias_for_object(item.object_id)
            if alias_id in written:
                continue
            write_spill_tensor(self._bridge.objects, alias_id, actual.tensor)
            written.add(alias_id)

    def release(self) -> None:
        """Drop optimizer state with the plan that owns its spill storage.

        Nothing is copied: a caller who wants the state takes the checkpoint
        while the callable is open. The state tensors view storage the plan is
        about to reclaim, so they are cleared rather than left dangling -- the
        same ownership restoration a planning rollback performs. State the
        caller imported is left alone; the plan was lent it.
        """

        if not release_optimizer_state_from_plan(
            self.optimizer,
            runtime=self._state.runtime,
        ):
            return
        self.optimizer.state.clear()
        self.initialized = False
        self.available = False

    def created_state(
        self,
        record: _ExecutionTaskRecord,
    ) -> tuple[
        tuple[PublishedStorage, ...],
        tuple[tuple[str, torch.Tensor, str], ...],
    ]:
        artifact = record.artifact
        if not isinstance(artifact, OpaqueOptimizerArtifact):
            raise RuntimeError("initial optimizer state requires an opaque artifact")
        outputs = {
            binding.name: binding.tensor
            for binding in opaque_optimizer_outputs(
                artifact,
                self.optimizer,
                device_ordinal=self._state.device.index or 0,
            )
        }
        produced: set[str] = set()
        adopted: list[PublishedStorage] = []
        bound: list[tuple[str, torch.Tensor, str]] = []
        for item in record.optimizer_outputs:
            tensor = outputs.get(item.name)
            if tensor is None:
                raise RuntimeError(
                    f"optimizer did not create planned state {item.name!r}"
                )
            if item.alias_id not in produced and item.publication_ordinal is not None:
                adopted.append(
                    PublishedStorage(
                        tensor,
                        item.alias_id,
                        item.publication_ordinal,
                    )
                )
                produced.add(item.alias_id)
            bound.append((item.object_id, tensor, item.alias_id))
        return tuple(adopted), tuple(bound)

    def current_bindings(self) -> dict[str, Any]:
        return {
            item.name: item
            for item in current_optimizer_bindings(
                dict(self._state.model.named_parameters()), self.optimizer
            )
        }

    def _copied_alias_buffer(self, alias_id: str) -> torch.Tensor:
        """Return a writable host buffer holding one alias's current bytes."""

        owner = torch.empty(
            self._size_by_alias[alias_id],
            dtype=torch.uint8,
            device="cpu",
        )
        read_spill_tensor(self._bridge.objects, alias_id, owner)
        return owner

    def expose_cpu(self) -> tuple[ExposedOptimizerTensor, ...]:
        """Point live optimizer state at host copies of its pool bytes.

        Each alias group is read out of the pool into a writable buffer rather
        than viewed in place. A pool's memory is not always in this address
        space, so copying is the one way that works for every kind -- the same
        reason state enters a pool by copying, and the reason there is no
        second path here that would work only sometimes.
        """

        self._bridge.wait_until_idle()
        current = self.current_bindings()
        exposed: list[ExposedOptimizerTensor] = []
        owners: dict[str, torch.Tensor] = {}
        for item in self._lowered.optimizer_objects:
            actual = current.get(item.name)
            if actual is None:
                continue
            tensor = actual.tensor
            alias_id = self._bridge.objects.alias_for_object(item.object_id)
            owner = owners.get(alias_id)
            if owner is None:
                owner = self._copied_alias_buffer(alias_id)
                owners[alias_id] = owner
            device_placeholder = tensor.data
            layout = TensorLayout(
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset()),
                tensor.dtype,
            )
            tensor.data = self._view(owner, layout)
            exposed.append(ExposedOptimizerTensor(tensor, device_placeholder))
        return tuple(exposed)

    def restore_spill_only(self, exposed: tuple[ExposedOptimizerTensor, ...]) -> None:
        # Exposing state never changes the neutral runtime object. Restore the
        # exact dematerialized CUDA views that were present before the CPU
        # snapshot; manufacturing temporary device allocations here would add
        # no information and can exceed the execution pool for large AdamW
        # inventories even though every individual task is feasible.
        for item in exposed:
            item.tensor.data = item.device_placeholder

    @staticmethod
    def _view(owner: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
        return torch.empty(0, dtype=layout.dtype, device=owner.device).set_(
            owner.untyped_storage(),
            layout.storage_offset,
            layout.shape,
            layout.stride,
        )


__all__ = ["OptimizerState"]
