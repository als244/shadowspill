"""One plan's objects: plan-local aliases bound to shared runtime objects.

The registry owns registrations and borrows existing objects for shared plans.
Zero-byte aliases have no physical registration. Spill data access belongs to
execution.SpillStorage; this module only locates its current leases.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from shadowspill.errors import PlanningError
from shadowspill.ir import ShadowSpillProgram
from shadowspill.runtime import Runtime
from shadowspill.runtime.abi import (
    ObjectBinding,
    ObjectLocationSnapshot,
)
from shadowspill.runtime.failures import RuntimeExecutionError
from shadowspill.runtime.objects import (
    register_object,
    reserve_runtime_object_ids,
)
from shadowspill.runtime.occupancy import PoolAllocation

from .common import (
    plan_local_id,
    require_status,
)


class PlanObjects:
    """The alias-to-runtime-object registry of one plan, and what it may do."""

    def __init__(
        self,
        runtime: Runtime,
        runtime_library: Any,
        program: ShadowSpillProgram,
        plan_handle: int,
        *,
        spill_pool_id: int,
    ) -> None:
        self.runtime = runtime
        self.library = runtime._installed.library
        self.runtime_library = runtime_library
        self.plan_handle = plan_handle
        self.spill_pool_id = spill_pool_id
        self._alias_by_object: dict[str, str] = {
            item.object_id: item.alias_group_id for item in program.objects
        }
        self._size_by_alias: dict[str, int] = {
            item.alias_group_id: item.size_bytes for item in program.alias_groups
        }
        self._spill_size_by_alias = {
            item.alias_group_id: item.spill_size_bytes for item in program.alias_groups
        }
        # An alias group's role, so a range the runtime reports as bound to an
        # object can say what the object is for. The runtime deliberately does
        # not know this; the program does.
        self._role_by_alias: dict[str, str] = {
            item.alias_group_id: item.role.value
            if hasattr(item.role, "value")
            else str(item.role)
            for item in program.objects
        }
        self._registered: set[str] = set()
        self._borrowed: set[str] = set()
        self._runtime_object_ids: dict[str, int] = {}

    def runtime_object_id(self, alias_id: str) -> int:
        """The shared runtime identity bound to one plan-local alias."""

        try:
            return self._runtime_object_ids[alias_id]
        except KeyError as exc:
            raise RuntimeExecutionError(
                f"plan object {alias_id!r} is not bound to a runtime object"
            ) from exc

    def _record_runtime_object(self, alias_id: str, runtime_object_id: int) -> None:
        existing = self._runtime_object_ids.get(alias_id)
        if existing is not None and existing != runtime_object_id:
            raise RuntimeExecutionError(
                f"plan object {alias_id!r} changed runtime identity: "
                f"{existing} -> {runtime_object_id}"
            )
        self._runtime_object_ids[alias_id] = runtime_object_id

    def _allocate_runtime_object_id(self, alias_id: str) -> int:
        existing = self._runtime_object_ids.get(alias_id)
        if existing is not None:
            return existing
        runtime_object_id = reserve_runtime_object_ids(self.runtime, 1)[0]
        self._record_runtime_object(alias_id, runtime_object_id)
        return runtime_object_id

    def _bind_plan_object(self, alias_id: str) -> None:
        if not self.requires_storage(alias_id):
            return
        handle = ctypes.c_size_t()
        require_status(
            self.library,
            self.runtime_library.shadowspill_object_handle_acquire(
                self.runtime._runtime_handle,
                self.runtime_object_id(alias_id),
                ctypes.byref(handle),
            ),
            f"acquire runtime object {alias_id}",
        )
        if handle.value == 0:
            raise RuntimeExecutionError(
                f"runtime returned an empty object handle for {alias_id!r}"
            )
        try:
            require_status(
                self.library,
                self.runtime_library.shadowspill_plan_bind_object(
                    self.plan_handle,
                    plan_local_id(alias_id, "alias_"),
                    handle.value,
                    0,  # Shared plans execute in causal order.
                ),
                f"bind plan object {alias_id}",
            )
        finally:
            require_status(
                self.library,
                self.runtime_library.shadowspill_object_handle_release(handle.value),
                f"release runtime object handle {alias_id}",
            )

    def bind(self, alias_ids: Iterable[str]) -> None:
        for alias_id in dict.fromkeys(alias_ids):
            self._bind_plan_object(alias_id)

    def role_of(self, item: PoolAllocation) -> str:
        """What a held range is for, resolving a bound object against the program.

        The runtime reports a bound object by id because it does not know what
        an object is for. This bridge holds the program, so a planned range can
        say parameter or activation rather than merely planned.
        """

        if item.object_id is None:
            return str(item.role)
        alias = next(
            (
                alias
                for alias, bound in self._runtime_object_ids.items()
                if bound == item.object_id
            ),
            None,
        )
        role = None if alias is None else self._role_by_alias.get(alias)
        return role if role is not None else str(item.role)

    def alias_for_object(self, object_id: str) -> str:
        try:
            return self._alias_by_object[object_id]
        except KeyError as exc:
            raise PlanningError(f"unknown program object {object_id!r}") from exc

    def reserve_spill(
        self, alias_id: str, *, retain_spill_copy: bool, transfer_address: int = 0
    ) -> None:
        """Reserve a lease that the Executor fills through bounded spill access."""
        self._register(
            alias_id,
            initially_resident=True,
            retain_spill_copy=retain_spill_copy,
            transfer_address=transfer_address,
        )

    def register_placeholder(self, alias_id: str, *, transfer_address: int = 0) -> None:
        """Register an object before its first production, without an initial value."""
        if alias_id not in self._registered:
            self._register(
                alias_id,
                initially_resident=False,
                retain_spill_copy=False,
                transfer_address=transfer_address,
            )

    def _register(
        self,
        alias_id: str,
        *,
        initially_resident: bool,
        retain_spill_copy: bool,
        transfer_address: int,
    ) -> None:
        size = self._size(alias_id)
        if self._spill_size(alias_id) != size and transfer_address == 0:
            raise PlanningError("sharded host storage requires a transfer descriptor")
        if size:
            require_status(
                self.library,
                register_object(
                    self.runtime,
                    self._allocate_runtime_object_id(alias_id),
                    size,
                    pool_id=self.spill_pool_id,
                    retain_spill_copy=retain_spill_copy,
                    initially_resident=initially_resident,
                    transfer_address=transfer_address,
                ),
                f"register plan object {alias_id}",
            )
        self._registered.add(alias_id)
        self._bind_plan_object(alias_id)

    def unregister(self, alias_ids: Iterable[str]) -> None:
        for alias_id in dict.fromkeys(alias_ids):
            if alias_id not in self._registered:
                continue
            if alias_id not in self._borrowed and self.requires_storage(alias_id):
                require_status(
                    self.library,
                    self.runtime_library.shadowspill_unregister_object(
                        self.runtime._runtime_handle,
                        self.runtime_object_id(alias_id),
                    ),
                    "unregister object",
                )
            self._borrowed.discard(alias_id)
            self._registered.remove(alias_id)
            self._runtime_object_ids.pop(alias_id, None)

    def registered_aliases(self) -> frozenset[str]:
        return frozenset(self._registered)

    def registered_runtime_objects(self) -> Mapping[str, int]:
        """Return plan aliases and their stable runtime object identities."""

        return {
            alias_id: runtime_object_id
            for alias_id, runtime_object_id in self._runtime_object_ids.items()
            if alias_id in self._registered
        }

    def borrow(self, objects: Mapping[str, int]) -> None:
        """Bind shared-plan objects without taking over their registrations."""
        for alias_id, runtime_object_id in objects.items():
            self._size(alias_id)
            self._record_runtime_object(alias_id, runtime_object_id)
            self._registered.add(alias_id)
            self._borrowed.add(alias_id)
            self._bind_plan_object(alias_id)

    def requires_storage(self, alias_id: str) -> bool:
        """Return whether an alias bundle owns any physical payload bytes."""

        return self._size(alias_id) != 0

    def zero_binding(self, alias_id: str) -> ObjectBinding:
        if self.requires_storage(alias_id):
            raise AssertionError("zero binding requested for a materialized alias")
        return ObjectBinding(
            plan_local_id(alias_id, "alias_"),
            0,
            0,
            0,
            None,
        )

    def expand_bindings(
        self,
        aliases: Sequence[str],
        materialized_aliases: Sequence[str],
        materialized_bindings: ctypes.Array[ObjectBinding] | Sequence[ObjectBinding],
    ) -> tuple[ObjectBinding, ...]:
        if len(materialized_aliases) != len(materialized_bindings):
            raise RuntimeExecutionError("runtime binding count differs")
        iterator = iter(materialized_bindings)
        result: list[ObjectBinding] = []
        for alias_id in aliases:
            result.append(
                next(iterator)
                if self.requires_storage(alias_id)
                else self.zero_binding(alias_id)
            )
        try:
            next(iterator)
        except StopIteration:
            return tuple(result)
        raise RuntimeExecutionError("runtime returned excess object bindings")

    def spill_location(self, alias_id: str) -> tuple[int, int] | None:
        """Where one alias's current bytes are in the spill pool, and how many.

        Return None for non-addressable pools or a stale spill copy. The
        Executor uses bounded range transfers for non-addressable pools.
        """

        size = self._spill_size(alias_id)
        pool = next(
            (
                item
                for item in self.runtime.pools.values()
                if item.pool_id == self.spill_pool_id
            ),
            None,
        )
        if size == 0 or pool is None or not pool.addressable:
            return None
        snapshot = ObjectLocationSnapshot()
        status = int(
            self.runtime_library.shadowspill_object_location_snapshot(
                self.runtime._runtime_handle,
                self.runtime_object_id(alias_id),
                self.spill_pool_id,
                ctypes.byref(snapshot),
            )
        )
        if status != 0 or not snapshot.has_lease or not snapshot.current:
            return None
        if snapshot.size_bytes != size:
            raise RuntimeExecutionError("physical spill capacity differs from the Plan")
        return (int(snapshot.pointer), size) if snapshot.pointer else None

    def _spill_size(self, alias_id: str) -> int:
        self._size(alias_id)  # Same diagnostic for an unknown alias.
        return self._spill_size_by_alias[alias_id]

    def _size(self, alias_id: str) -> int:
        try:
            return self._size_by_alias[alias_id]
        except KeyError as exc:
            raise PlanningError(f"unknown alias group {alias_id!r}") from exc


__all__ = ["PlanObjects"]
