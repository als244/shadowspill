"""Canonical owners for persistent state across sequential operation Programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shadowspill.artifacts.program import Program
from shadowspill.artifacts.state import StateBinding, StateId, StateTensor
from shadowspill.artifacts.state_inventory import program_state

type StateLocation = tuple[str, str]


@dataclass(frozen=True)
class SharedState:
    """Borrowing maps to an earlier owning Program, never to another borrower."""

    aliases: Mapping[str, Mapping[str, StateLocation]]
    host: Mapping[str, Mapping[str, StateLocation]]


def shared_state(
    programs: Mapping[str, Program],
    *,
    logical_rank: int = 0,
) -> SharedState:
    """Resolve one inventory without allocating any tensor or changing a Plan.

    The first occurrence of a state owns its allocation. An evaluation-only
    primary Program need not contain gradients or moments: their first training
    Program can own them, and later Programs borrow those same destinations.
    Input buffers, outputs and scratch remain private to each Program.
    """
    if not programs:
        raise ValueError("shared state requires at least one Program")
    if any(not isinstance(name, str) for name in programs):
        raise TypeError("Program names must be strings")
    tensor_by_id: dict[StateId, StateTensor] = {}
    binding_by_id: dict[StateId, StateBinding] = {}
    locations: dict[StateId, StateLocation] = {}
    model_ids: set[StateId] | None = None
    optimizer_types: set[str] = set()
    aliases: dict[str, Mapping[str, StateLocation]] = {}
    hosts: dict[str, Mapping[str, StateLocation]] = {}
    for name, program in programs.items():
        inventory = program_state(program, logical_rank)
        registered = {
            t.identity
            for t in inventory.tensors
            if t.identity.role in ("parameter", "buffer")
        }
        if model_ids is None:
            model_ids = registered
        elif registered != model_ids:
            raise ValueError("shared Programs must describe the same model inventory")
        if optimizer_type := program.bindings.get("optimizer_type"):
            optimizer_types.add(optimizer_type)
        if len(optimizer_types) > 1:
            raise ValueError("shared Programs require the same optimizer family")
        current = {b.identity: b for b in inventory.bindings}
        physical: dict[str, StateLocation] = {}
        host: dict[str, StateLocation] = {}
        reverse: dict[StateLocation, str] = {}
        # Match known state first: a new tied name may share a known destination.
        for tensor in inventory.tensors:
            identity = tensor.identity
            if identity not in tensor_by_id:
                continue
            if tensor != tensor_by_id[identity]:
                raise ValueError(
                    f"shared state layout or ownership differs: {identity}"
                )
            old, new = binding_by_id[identity], current[identity]
            if old.storage != new.storage or (old.host_binding is None) != (
                new.host_binding is None
            ):
                raise ValueError(f"shared state physical storage differs: {identity}")
            location = locations[identity]
            if new.host_binding is not None:
                if host.setdefault(new.host_binding, location) != location:
                    raise ValueError("shared host state aliasing differs")
            else:
                assert new.alias_group_id is not None
                alias = new.alias_group_id
                if (
                    physical.setdefault(alias, location) != location
                    or reverse.setdefault(location, alias) != alias
                ):
                    raise ValueError(f"shared state aliasing differs: {identity}")
        for tensor in inventory.tensors:
            identity = tensor.identity
            if identity in tensor_by_id:
                continue
            binding = current[identity]
            if binding.host_binding is not None:
                location = host.get(binding.host_binding, (name, binding.host_binding))
            else:
                assert binding.alias_group_id is not None
                location = physical.get(
                    binding.alias_group_id, (name, binding.alias_group_id)
                )
            tensor_by_id[identity] = tensor
            binding_by_id[identity] = binding
            locations[identity] = location
        aliases[name] = MappingProxyType(physical)
        hosts[name] = MappingProxyType(host)
    return SharedState(MappingProxyType(aliases), MappingProxyType(hosts))
