"""Persistent frontend views participating in a storage replacement."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.pytorch.representations import map_tensor, storage_view
from shadowspill.pytorch.spill import read_spill_tensor, spill_view, write_spill_tensor
from shadowspill.runtime.plan import RuntimeBridge

if TYPE_CHECKING:
    from shadowspill.ir import ShadowSpillProgram


def object_ids_by_alias(program: ShadowSpillProgram) -> dict[str, tuple[str, ...]]:
    """Index each object's alias once, preserving program order."""
    grouped: dict[str, list[str]] = {
        group.alias_group_id: [] for group in program.alias_groups
    }
    for item in program.objects:
        grouped[item.alias_group_id].append(item.object_id)
    return {alias: tuple(objects) for alias, objects in grouped.items()}


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
    model: torch.nn.Module
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

    def write_model_entries(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        cast_names: frozenset[str] = frozenset(),
    ) -> None:
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
        aliases_by_name: dict[str, set[str]] = {}
        entries_by_name: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for item in self._registrations():
            alias_id = self.bridge.objects.alias_for_object(item.binding.object_id)
            aliases_by_name.setdefault(item.binding.name, set()).add(alias_id)
            entries_by_name.setdefault(item.binding.name, []).append(
                (alias_id, item.tensor)
            )
        missing = sorted(set(values) - set(aliases_by_name))
        if missing:
            raise RuntimeError(f"unsupported model state entries: {missing}")
        templates = self.model.state_dict(keep_vars=True)
        for name, source in values.items():
            destination = templates[name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"model state entry {name!r} must be a tensor")
            if tuple(source.shape) != tuple(destination.shape) or (
                source.dtype != destination.dtype and name not in cast_names
            ):
                raise RuntimeError(
                    f"model state entry {name!r} has incompatible geometry"
                )
        logical = dict(self.model.named_parameters(remove_duplicate=False))
        logical.update(self.model.named_buffers(remove_duplicate=False))
        self.bridge.wait_runtime_idle()
        # A logical parameter may have several physical components. Stage only
        # those roots; later aliases read back any preceding partial updates.
        for name, source in values.items():
            owners = {}
            for alias_id in aliases_by_name[name]:
                owner = spill_view(self.bridge.objects, alias_id)
                if owner is None:
                    owner = self._read_model_aliases(aliases={alias_id})[alias_id]
                owners[alias_id] = owner
            # Resolve only this entry's components. Re-enumerating the complete
            # model for every weight makes streamed restore quadratic.
            mapped = {
                id(tensor): self._cpu_view(owners[alias_id], tensor)
                for alias_id, tensor in entries_by_name[name]
            }
            destination = map_tensor(
                logical[name], lambda value: mapped[id(value)]
            ).detach()
            with torch.no_grad():
                destination.copy_(source.detach().to(device="cpu"))
            for alias_id, owner in owners.items():
                write_spill_tensor(self.bridge.objects, alias_id, owner)
            del destination, mapped, owners, owner

    def _state_from_owners(
        self, owners: Mapping[str, torch.Tensor], *, names: set[str] | None = None
    ) -> OrderedDict[str, torch.Tensor]:
        """Rebuild logical state around a set of physical CPU storage owners."""
        mapped = {}
        requested = set(self._state_names) if names is None else names
        for item in self._registrations():
            alias_id = self.bridge.objects.alias_for_object(item.binding.object_id)
            if alias_id in owners and item.binding.name in requested:
                mapped[id(item.tensor)] = self._cpu_view(owners[alias_id], item.tensor)
        tensors: dict[str, torch.Tensor] = dict(
            self.model.named_parameters(remove_duplicate=False)
        )
        tensors.update(self.model.named_buffers(remove_duplicate=False))
        memo: dict[int, Any] = {}
        return OrderedDict(
            (
                name,
                map_tensor(
                    tensors[name], lambda value: mapped[id(value)], memo=memo
                ).detach(),
            )
            for name in self._state_names
            if name in requested
        )

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
        return storage_view(
            owner, tensor.dtype, tensor.shape, tensor.stride(), tensor.storage_offset()
        )


__all__ = ["MaterializedState", "ReplacementStorageViews"]
