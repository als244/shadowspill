"""Persistent frontend views participating in a storage replacement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from shadowspill.pytorch.spill import read_spill_tensor, write_spill_tensor
from shadowspill.runtime.plan import RuntimeBridge


@dataclass(frozen=True, slots=True)
class ReplacementStorageViews:
    """Persistent frontend views rebound when one logical object is overwritten."""

    alias_id: str
    tensors: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.alias_id:
            raise ValueError("replacement alias must be non-empty")
        if not self.tensors:
            raise ValueError("replacement must name at least one frontend view")


class MaterializedState:
    """What a materialized model's state does regardless of what it is for.

    Forward and training states differ in what they hold, not in how they
    read an alias back from spill, how they take a CPU view of one, or how
    they rebind a replaced view.
    """

    bridge: RuntimeBridge
    object_store: dict[str, torch.Tensor]
    #: The model's state entries by name, as ``state_dict()`` enumerates them.
    _state_names: tuple[str, ...]

    def _empty_model_aliases(
        self, *, aliases: set[str] | None = None
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _registrations(self) -> Sequence[Any]:
        """Each registered tensor with its binding: ``binding.name``,
        ``binding.object_id`` and the ``tensor`` whose geometry it has."""

        raise NotImplementedError

    def write_model_entries(self, values: Mapping[str, torch.Tensor]) -> None:
        """Write the named state entries into the pool, leaving the rest as
        they are: how a value set between invocations reaches state the plan
        owns.

        A planned model's registered tensors are the plan's handles, placed
        and moved by the runtime, so writing one in place lands wherever the
        handle points at that moment. The bytes go where the state lives
        instead: each entry's alias is read out of the spill pool, the entry
        replaced in that copy, and the alias written back, once the runtime
        is idle, so an invocation still reading the old value has finished
        with it. Only the aliases holding a named entry are touched.
        """

        unknown = sorted(set(values) - set(self._state_names))
        if unknown:
            raise KeyError(
                f"no persistent model state entries named {unknown}; a buffer "
                "registered with persistent=False is not state a call can set"
            )
        targets = {
            item.binding.name: item
            for item in self._registrations()
            if item.binding.name in values
        }
        missing = sorted(set(values) - set(targets))
        if missing:
            raise RuntimeError(f"unsupported model state entries: {missing}")
        aliases = {
            name: self.bridge.objects.alias_for_object(item.binding.object_id)
            for name, item in targets.items()
        }
        owners = self._read_model_aliases(aliases=set(aliases.values()))
        for name, item in targets.items():
            source = values[name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"model state entry {name!r} must be a tensor")
            destination = self._cpu_view(owners[aliases[name]], item.tensor)
            if (
                tuple(source.shape) != tuple(destination.shape)
                or source.dtype != destination.dtype
            ):
                raise RuntimeError(
                    f"model state entry {name!r} has incompatible geometry"
                )
            destination.copy_(source.detach().to(device="cpu"))
        for alias_id, owner in owners.items():
            write_spill_tensor(self.bridge.objects, alias_id, owner)

    def publish_replacement_views(self, replacement: ReplacementStorageViews) -> None:
        """Keep the stable frontend representative rebound by the runtime boundary."""

        self.object_store[replacement.alias_id] = replacement.tensors[0]

    def _read_model_aliases(
        self,
        *,
        aliases: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        self.bridge.wait_runtime_idle()
        owners = self._empty_model_aliases(aliases=aliases)
        for alias_id, owner in owners.items():
            read_spill_tensor(self.bridge.objects, alias_id, owner)
        return owners

    @staticmethod
    def _cpu_view(owner: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            owner.untyped_storage(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )


__all__ = ["MaterializedState", "ReplacementStorageViews"]
