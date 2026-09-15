"""Who holds a range, in PyTorch's terms: the frontend's `LiveBindings`.

`occupants` maps a pool range to the tensors whose storage lies inside it;
`retainers` names where each of those tensors is referenced from, which is what
a caller needs to release one; `dematerialize` detaches them so the runtime can
reclaim the bytes. All three compare addresses and return descriptions: none
keeps a reference, which would extend the lifetime of exactly what is being
investigated.

Only PyTorch can walk its own live objects, and only the runtime knows which
allocation owns an address, so `occupants` takes the lookup as an argument
rather than reaching for a runtime.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable, Sequence
from types import FrameType, ModuleType

import torch

from shadowspill.runtime.occupancy import PoolAllocation


def occupants(
    allocations: Sequence[PoolAllocation],
    locate: Callable[[int], int | None],
) -> dict[int, tuple[object, ...]]:
    """The frontend objects whose storage lies inside each given range.

    Answers "what is this range, in the framework's terms". Every live
    accelerator tensor is mapped back to the allocation that owns its address,
    and matched against the allocations asked about.

    A range with no match is held by something the framework does not own -- a
    library's retained state, say -- and that is itself the answer: there is no
    reference for a caller to drop.

    Addresses are compared rather than references kept. A runtime holding a
    reference to a frontend object would either keep it alive, which is wrong,
    or hold it weakly and be unable to act on it, so it holds neither.
    """

    wanted = {item.allocation_id for item in allocations}
    found: dict[int, list[object]] = {key: [] for key in wanted}
    # Walking every object touches deprecated framework attributes whose
    # getters warn; the warning belongs to the object being looked at, not
    # to this query.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = [
            item for item in gc.get_objects() if isinstance(item, torch.Tensor)
        ]
    for candidate in candidates:
        try:
            if candidate.device.type in ("cpu", "meta"):
                continue
            if candidate.is_meta or type(candidate).__name__ == "FakeTensor":
                # Export leaves these behind. They report a device and have no
                # storage, so asking for a data pointer warns and tells us
                # nothing: one cannot occupy a range in a pool.
                continue
            address = candidate.untyped_storage().data_ptr()
        except Exception:
            continue
        if not address:
            continue
        allocation_id = locate(address)
        if allocation_id is not None and allocation_id in found:
            found[allocation_id].append(candidate)
    return {key: tuple(value) for key, value in found.items()}


def _references_in(container: object) -> tuple[tuple[str, int], ...]:
    """Where one container refers to other objects, as (where, identity) pairs.

    Only the shapes a reference is actually held in: a mapping's values, a
    sequence's items, a set's members, and the slots of an object carrying no
    `__dict__`. An attribute on an ordinary object is not one of them -- it is
    held in that object's `__dict__`, which is what the collector reports.
    """

    if isinstance(container, dict):
        return tuple(
            (
                f".{key}"
                if isinstance(key, str) and key.isidentifier()
                else f"[{key!r}]",
                id(value),
            )
            for key, value in tuple(container.items())
        )
    if isinstance(container, (list, tuple)):
        return tuple((f"[{index}]", id(value)) for index, value in enumerate(container))
    if isinstance(container, (set, frozenset)):
        return tuple(("{...}", id(value)) for value in container)
    slots = getattr(type(container), "__slots__", ())
    named = (slots,) if isinstance(slots, str) else slots
    found: list[tuple[str, int]] = []
    for name in named:
        try:
            found.append((f".{name}", id(getattr(container, name))))
        except AttributeError:
            continue
    return tuple(found)


def _named(candidate: object) -> str | None:
    """What holds this, as a module or a type, if anything names it."""

    for holder in gc.get_referrers(candidate):
        if isinstance(holder, FrameType):
            continue
        if isinstance(holder, ModuleType):
            return holder.__name__
        mapping = getattr(holder, "__dict__", None)
        if isinstance(mapping, dict) and any(
            value is candidate for value in tuple(mapping.values())
        ):
            return type(holder).__name__
        if isinstance(holder, dict):
            for owner in gc.get_referrers(holder):
                if isinstance(owner, ModuleType):
                    return f"{owner.__name__}(globals)"
                if getattr(owner, "__dict__", None) is holder:
                    return type(owner).__name__
                # A decorator's cache lives in a closure cell, which is
                # reached by neither a module nor an instance dict. The
                # function that closed over it is the name worth having.
                if type(owner).__name__ == "cell":
                    for closed in gc.get_referrers(owner):
                        name = getattr(closed, "__qualname__", None)
                        if name is not None:
                            module = getattr(closed, "__module__", "?")
                            return f"{module}.{name}"
    return None


def retainers(
    held: Sequence[object], *, ignore: Sequence[object] = ()
) -> dict[int, tuple[str, ...]]:
    """What keeps each given object alive, named where the reference is held.

    `occupants` answers which frontend object occupies a range; this answers
    why that object is still reachable, which is what a caller needs to release
    it. A reference can only be dropped where it is held, so what is named is
    the container -- an attribute on a class, an entry in a module's globals --
    and not the object again. An object no container names is held by a
    library, and that is itself the answer: Python has nothing to drop.

    Keyed by `id`, so a caller reads answers back with the objects it passed
    in. Stack frames are excluded, this query's own among them; every other
    referrer is reported, including the caller's own container.

    Descriptions, never references, for the reason `occupants` keeps none:
    retaining what this walks would extend the lifetime of exactly the objects
    under investigation.
    """

    if not held:
        return {}
    found: dict[int, list[str]] = {id(item): [] for item in held}
    # This query's own containers refer to the objects it is asking about: the
    # sequence passed in, and whatever the caller built it from. Reporting
    # those would describe the question rather than the answer -- a dict keyed
    # by allocation id is `occupants`' own result, not a holder worth naming.
    mine = {id(held), *(id(item) for item in ignore)}
    referrers = [
        item
        for item in gc.get_referrers(*held)
        if not isinstance(item, FrameType) and id(item) not in mine
    ]
    # An attribute arrives as the owner's `__dict__`, which names neither the
    # owner nor the attribute. One further hop resolves it, batched, because
    # each hop walks the whole heap.
    owners: dict[int, str] = {}
    mappings = [item for item in referrers if isinstance(item, dict)]
    if mappings:
        for owner in gc.get_referrers(*mappings):
            if isinstance(owner, (FrameType, dict)):
                continue
            mapping = getattr(owner, "__dict__", None)
            if mapping is None:
                continue
            owners[id(mapping)] = (
                owner.__name__
                if isinstance(owner, ModuleType)
                else type(owner).__name__
            )
    # A list or tuple carries no name of its own, so it is named by whatever
    # holds it: one more hop turns `list[2]` into `Owner.attribute[2]`. Named
    # top down, because a container's label depends on its holder's: naming
    # the container first and the dict afterwards cannot improve on
    # `dict[49362]`, which says nothing about whose dict it is.
    anonymous = [item for item in referrers if isinstance(item, (list, tuple, set))]
    for container in anonymous:
        for holder in gc.get_referrers(container):
            if isinstance(holder, FrameType) or not isinstance(holder, dict):
                continue
            keys = [key for key, value in tuple(holder.items()) if value is container]
            if not keys:
                continue
            whose = owners.get(id(holder)) or _named(holder) or "dict"
            owners[id(container)] = f"{whose}[{keys[0]}]"
            break

    for referrer in referrers:
        label = owners.get(id(referrer), type(referrer).__name__)
        try:
            references = _references_in(referrer)
        except Exception:  # a diagnostic must not fail on what it inspects
            continue
        for where, target in references:
            if target in found:
                found[target].append(f"{label}{where}")
    return {key: tuple(dict.fromkeys(value)) for key, value in found.items()}


def dematerialize(bindings: Sequence[object]) -> int:
    """Detach these tensors from their leases; return how many were detached.

    A tensor whose storage is already empty holds no lease and is skipped, so
    the count is what the runtime may now reclaim.
    """

    storages = [
        item
        for item in bindings
        if isinstance(item, torch.Tensor) and item.untyped_storage().data_ptr() != 0
    ]
    if not storages:
        return 0
    torch.ops.shadowspill._dematerialize_storages(storages)
    detached = len(storages)
    storages.clear()
    return detached


__all__ = ["dematerialize", "occupants", "retainers"]
