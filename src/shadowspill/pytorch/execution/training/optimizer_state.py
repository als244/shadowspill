"""The optimizer's state as the plan holds it: how a checkpoint reads and
writes it through the spill pool, and how the plan lets it go."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch.utils._pytree import tree_map

from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
)
from shadowspill.pytorch.materialization.training import TrainingMaterializedState
from shadowspill.pytorch.optimizer import (
    OptimizerTensorRole,
    current_optimizer_bindings,
    restore_optimizer_checkpoint_structure,
)
from shadowspill.pytorch.spill import (
    read_spill_tensor,
    spill_view,
    write_spill_tensor,
)
from shadowspill.pytorch.state.checkpoint import PoolCheckpoint
from shadowspill.pytorch.state.optimizer import release_optimizer_state_from_plan
from shadowspill.runtime.plan import (
    RuntimeBridge,
)

from .values import ExposedOptimizerTensor, TensorLayout


class OptimizerState:
    """The optimizer and the plan's account of its state, which exists in the
    spill pool from planning on.

    ``optimizer_parameters`` are the optimizer's parameters by the model's
    names, and ``master_names`` the names whose parameter is a master copy of
    the weights rather than the weights: the masters are the optimizer's to
    keep, and a checkpoint reads and writes them beside its state.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        state: TrainingMaterializedState,
        bridge: RuntimeBridge,
        lowered: LoweredTrainingProgram,
        optimizer_parameters: Mapping[str, torch.nn.Parameter],
    ) -> None:
        self.optimizer = optimizer
        self.optimizer_parameters = dict(optimizer_parameters)
        self.master_names = tuple(
            item.name
            for item in lowered.optimizer_objects
            if item.role is OptimizerTensorRole.PARAMETER
        )
        self._state = state
        self._bridge = bridge
        self._lowered = lowered
        self._objects = tuple(
            item
            for item in lowered.optimizer_objects
            if item.role is not OptimizerTensorRole.COMPUTE_COPY
        )
        self._size_by_alias = {
            item.alias_group_id: item.size_bytes
            for item in lowered.program.alias_groups
        }

    def state_dict(self) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
        """Synchronously snapshot optimizer state without stale CUDA pointers,
        with the masters by name.

        The snapshot is independent: each tensor is its own compact host
        allocation, aliasing neither runtime storage nor the other entries,
        so a caller can serialize it while training continues. The pool keeps
        the authoritative copy throughout and every alias is read out of it
        into a buffer, so an alias costs two until the snapshot is built. On a
        large model that is the biggest transient the frontend asks for, so
        budget for it.
        """

        exposed = self.expose_cpu()
        try:
            raw = self.optimizer.state_dict()
            snapshot = cast(
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
            return snapshot, {
                name: self.optimizer_parameters[name].detach().cpu().clone()
                for name in self.master_names
            }
        finally:
            self.restore_spill_only(exposed)

    def checkpoint_state(
        self,
        writer: PoolCheckpoint,
    ) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
        """Describe state without exposing all its pool payloads on the host."""
        bindings = self.current_bindings()
        aliases = {
            id(bindings[item.name].tensor): self._bridge.objects.alias_for_object(
                item.object_id
            )
            for item in self._objects
            if item.name in bindings
        }

        def describe(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            alias = aliases.get(id(value))
            if alias is None:
                return writer.value(value)
            owner = writer.alias(alias, self._size_by_alias[alias])
            return writer.view(owner, value)

        return tree_map(describe, self.optimizer.state_dict()), {
            name: describe(self.optimizer_parameters[name])
            for name in self.master_names
        }

    def load(
        self,
        value: Mapping[str, object],
        masters: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Restore optimizer metadata and write tensor bytes into spill storage,
        each master from ``masters``, cast to its dtype.

        The checkpoint has to hold every entry the plan keeps, and ``masters``
        every master; one that lacks any is refused before anything changes.
        """

        given = dict(masters or {})
        missing = sorted(set(self.master_names) - set(given))
        if missing:
            raise RuntimeError(f"checkpoint lacks the values of masters {missing}")
        self._bridge.wait_runtime_idle()
        planned = tuple(
            item for item in self._objects if item.name not in self.master_names
        )
        restored = restore_optimizer_checkpoint_structure(
            self.optimizer_parameters,
            self.optimizer,
            value,
            required=tuple(item.name for item in planned),
        )
        objects = {item.name: item for item in self._objects}
        for entry in restored:
            item = objects.get(entry.name)
            if item is None:
                if entry.destination.device.type != "cpu":
                    raise RuntimeError(
                        f"optimizer checkpoint tensor {entry.name!r} has no pool object"
                    )
                entry.destination.copy_(entry.source.detach().to(device="cpu"))
            else:
                self._write_tensor(item.object_id, entry.destination, entry.source)
        current = self.current_bindings()
        for item in self._objects:
            if item.name in self.master_names:
                self._write_tensor(
                    item.object_id, current[item.name].tensor, given[item.name]
                )

    def _write_tensor(
        self, object_id: str, template: torch.Tensor, source: torch.Tensor
    ) -> None:
        """Restore one alias at a time, including dtype conversion for masters."""
        alias = self._bridge.objects.alias_for_object(object_id)
        owner = spill_view(self._bridge.objects, alias)
        if owner is None:
            owner = self._copied_alias_buffer(alias)
        destination = self._view(
            owner,
            TensorLayout(
                tuple(template.shape),
                tuple(template.stride()),
                int(template.storage_offset()),
                template.dtype,
            ),
        )
        with torch.no_grad():
            destination.copy_(source.detach().to(device="cpu"))
        write_spill_tensor(self._bridge.objects, alias, owner)

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

    def current_bindings(self) -> dict[str, Any]:
        return {
            item.name: item
            for item in current_optimizer_bindings(
                self.optimizer_parameters, self.optimizer
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

    def expose_cpu(
        self, *, in_place: bool = False
    ) -> tuple[ExposedOptimizerTensor, ...]:
        """Point live optimizer state at host copies of its pool bytes.

        Each alias group is read out of the pool into a writable buffer: a
        pool's memory is not always in this address space, so copying is the
        one way that works for every kind. ``in_place`` views an alias group
        where it is instead, wherever the pool allows it, for a caller that
        only reads the state and is done before the next step.
        """

        self._bridge.wait_until_idle()
        current = self.current_bindings()
        exposed: list[ExposedOptimizerTensor] = []
        owners: dict[str, torch.Tensor] = {}
        for item in self._objects:
            actual = current.get(item.name)
            if actual is None:
                continue
            tensor = actual.tensor
            alias_id = self._bridge.objects.alias_for_object(item.object_id)
            owner = owners.get(alias_id)
            if owner is None:
                owner = spill_view(self._bridge.objects, alias_id) if in_place else None
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
