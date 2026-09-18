"""One plan's objects: plan-local aliases bound to shared runtime objects.

A program names its objects by plan-local alias; the runtime names them by a
process-wide identity. `PlanObjects` is the registry that holds the binding for
one plan, and every operation on it: registering a host payload, adopting a
persistent or shared object, the placeholder a task publishes into, reading
and writing the spill copy, and unregistering. A zero-byte alias owns no
physical payload and never reaches the runtime; the registry answers for it.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from shadowspill.errors import PlanningError
from shadowspill.ir import ShadowSpillProgram
from shadowspill.runtime import (
    ObjectConsistency,
    ObjectRef,
    Runtime,
)
from shadowspill.runtime.abi import ObjectBinding, ObjectSnapshot
from shadowspill.runtime.failures import RuntimeExecutionError
from shadowspill.runtime.objects import (
    acquire_object_reference,
    register_object,
    release_object_generation,
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
        # An alias group's role, so a range the runtime reports as bound to an
        # object can say what the object is for. The runtime deliberately does
        # not know this; the program does.
        self._role_by_alias: dict[str, str] = {
            item.alias_group_id: item.role.value
            if hasattr(item.role, "value")
            else str(item.role)
            for item in program.objects
        }
        self._zero_generations: dict[str, int] = {}
        self._registered: set[str] = set()
        self._borrowed: set[str] = set()
        self._binding_consistency: dict[str, int] = {}
        self._runtime_object_ids: dict[str, int] = {}

    def runtime_object_id(self, alias_id: str) -> int:
        """The shared runtime identity bound to one plan-local alias."""

        try:
            return self._runtime_object_ids[alias_id]
        except KeyError as exc:
            raise RuntimeExecutionError(
                f"plan object {alias_id!r} is not bound to a runtime object"
            ) from exc

    def acquire_object_reference(self, alias_id: str) -> ObjectRef:
        """Retain one registered alias independently of this plan's lifetime."""

        if alias_id not in self._registered:
            raise RuntimeExecutionError(
                f"cannot retain unregistered plan object {alias_id!r}"
            )
        if not self.requires_storage(alias_id):
            raise RuntimeExecutionError(
                "zero-byte shared objects are not supported by the public reference API"
            )
        return acquire_object_reference(
            self.runtime,
            object_id=self.runtime_object_id(alias_id),
            size_bytes=self._size(alias_id),
        )

    def release_object_generation(
        self,
        alias_id: str,
        *,
        expected_generation: int,
    ) -> None:
        """Release a closed public slot's residency, preserving identity."""

        if alias_id not in self._registered:
            raise RuntimeExecutionError(
                f"cannot release unregistered plan object {alias_id!r}"
            )
        if not self.requires_storage(alias_id):
            return
        release_object_generation(
            self.runtime,
            object_id=self.runtime_object_id(alias_id),
            expected_generation=expected_generation,
        )

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

    def _bind_plan_object(
        self,
        alias_id: str,
        *,
        consistency: int | None = None,
    ) -> None:
        if not self.requires_storage(alias_id):
            return
        resolved_consistency = self._binding_consistency.get(alias_id, 0)
        if consistency is not None:
            existing = self._binding_consistency.get(alias_id)
            if existing is not None and existing != consistency:
                raise RuntimeExecutionError(
                    f"plan object {alias_id!r} changed consistency policy"
                )
            resolved_consistency = consistency
            self._binding_consistency[alias_id] = consistency
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
                    resolved_consistency,
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
        alias = self._alias_by_runtime_object().get(int(item.object_id))
        role = None if alias is None else self._role_by_alias.get(alias)
        return role if role is not None else str(item.role)

    def _alias_by_runtime_object(self) -> dict[int, str]:
        """The inverse of the alias-to-runtime-object map, built on demand."""

        return {value: key for key, value in self._runtime_object_ids.items()}

    def alias_for_object(self, object_id: str) -> str:
        try:
            return self._alias_by_object[object_id]
        except KeyError as exc:
            raise PlanningError(f"unknown program object {object_id!r}") from exc

    def register_spill_bytes(
        self, alias_id: str, *, address: int, size: int, retain_spill_copy: bool
    ) -> None:
        """Register one object from host bytes the caller already holds.

        The caller says which bytes; whether they are host bytes is the
        frontend's to check, because only a framework knows where its own
        memory lives.
        """

        expected = self._size(alias_id)
        if size != expected:
            raise PlanningError(
                f"host payload for {alias_id!r} has {size} bytes; "
                f"the plan requires {expected}"
            )
        if expected == 0:
            self._registered.add(alias_id)
            self._zero_generations.setdefault(alias_id, 0)
            return
        runtime_object_id = self._allocate_runtime_object_id(alias_id)
        require_status(
            self.library,
            register_object(
                self.runtime,
                runtime_object_id,
                expected,
                pool_id=self.spill_pool_id,
                retain_spill_copy=bool(retain_spill_copy),
                initially_resident=True,
                source_address=address,
            ),
            "register host object",
        )
        self._registered.add(alias_id)
        self._bind_plan_object(alias_id)

    def adopt_persistent_object(
        self,
        alias_id: str,
        *,
        current_object_id: int,
        pool_id: int,
        size_bytes: int,
        pool_pointer: int,
    ) -> int:
        """Adopt one preloaded spill lease without allocating or copying it."""

        expected = self._size(alias_id)
        if size_bytes != expected:
            raise PlanningError(
                f"persistent payload for {alias_id!r} has {size_bytes} bytes; "
                f"the plan requires {expected}"
            )
        if pool_id != self.spill_pool_id:
            raise PlanningError(
                f"persistent object for {alias_id!r} resides in pool {pool_id}; "
                f"the plan selected spill pool {self.spill_pool_id}"
            )
        require_status(
            self.library,
            self.library.shadowspill_pytorch_validate_object_binding(
                pool_id, current_object_id, pool_pointer, size_bytes
            ),
            f"validate persistent object {alias_id}",
        )
        self._record_runtime_object(alias_id, current_object_id)
        self._registered.add(alias_id)
        self._bind_plan_object(alias_id)
        return current_object_id

    def adopt_shared_object(
        self,
        alias_id: str,
        reference: ObjectRef,
        *,
        consistency: ObjectConsistency,
    ) -> None:
        """Bind a plan alias to an externally owned runtime object."""

        reference._require_open()
        if not reference._belongs_to(self.runtime):
            raise PlanningError("shared input belongs to another Runtime")
        expected = self._size(alias_id)
        if reference.size_bytes != expected:
            raise PlanningError(
                f"shared input for {alias_id!r} has "
                f"{reference.size_bytes} bytes; the plan requires {expected}"
            )
        runtime_object_id = reference.object_id
        self._record_runtime_object(alias_id, runtime_object_id)
        self._registered.add(alias_id)
        self._borrowed.add(alias_id)
        consistency_code = 0 if consistency is ObjectConsistency.CAUSAL else 1
        self._bind_plan_object(
            alias_id,
            consistency=consistency_code,
        )

    def register_placeholder(self, alias_id: str) -> None:
        """Register a logical alias bundle before its first production."""

        if alias_id in self._registered:
            return
        if not self.requires_storage(alias_id):
            self._registered.add(alias_id)
            self._zero_generations.setdefault(alias_id, 0)
            return
        runtime_object_id = self._allocate_runtime_object_id(alias_id)
        require_status(
            self.library,
            register_object(
                self.runtime,
                runtime_object_id,
                self._size(alias_id),
                pool_id=self.spill_pool_id,
                retain_spill_copy=False,
                initially_resident=False,
            ),
            "register placeholder object",
        )
        self._registered.add(alias_id)
        self._bind_plan_object(alias_id)

    def write_spill_bytes(self, alias_id: str, *, address: int, size: int) -> None:
        if alias_id not in self._registered:
            raise RuntimeExecutionError(f"object {alias_id!r} is not registered")
        expected = self._size(alias_id)
        if size != expected:
            raise RuntimeExecutionError(
                f"runtime storage for {alias_id!r} has {size} bytes; "
                f"the plan requires {expected}"
            )
        if expected == 0:
            return
        require_status(
            self.library,
            self.runtime_library.shadowspill_write_object(
                self.runtime._runtime_handle,
                self.runtime_object_id(alias_id),
                self.spill_pool_id,
                address,
                expected,
            ),
            "write host object",
        )

    def read_spill_bytes(self, alias_id: str, *, address: int, size: int) -> None:
        expected = self._size(alias_id)
        if size != expected:
            raise RuntimeExecutionError(
                f"writeback storage for {alias_id!r} has {size} bytes; "
                f"the plan requires {expected}"
            )
        if expected == 0:
            return
        require_status(
            self.library,
            self.runtime_library.shadowspill_read_object(
                self.runtime._runtime_handle,
                self.runtime_object_id(alias_id),
                self.spill_pool_id,
                address,
                expected,
            ),
            "read host object",
        )

    def unregister(self, alias_ids: Iterable[str]) -> None:
        for alias_id in dict.fromkeys(alias_ids):
            if alias_id not in self._registered:
                continue
            if alias_id in self._borrowed:
                self._borrowed.remove(alias_id)
                self._registered.remove(alias_id)
                self._runtime_object_ids.pop(alias_id, None)
                self._binding_consistency.pop(alias_id, None)
                continue
            if not self.requires_storage(alias_id):
                self._registered.remove(alias_id)
                self._zero_generations.pop(alias_id, None)
                continue
            require_status(
                self.library,
                self.runtime_library.shadowspill_unregister_object(
                    self.runtime._runtime_handle,
                    self.runtime_object_id(alias_id),
                ),
                "unregister object",
            )
            self._registered.remove(alias_id)
            self._runtime_object_ids.pop(alias_id, None)
            self._binding_consistency.pop(alias_id, None)

    def current_generation(self, alias_id: str) -> int:
        """Snapshot one public object's authoritative runtime generation."""

        if alias_id not in self._registered or not self.requires_storage(alias_id):
            raise RuntimeExecutionError(
                f"cannot snapshot unregistered plan object {alias_id!r}"
            )
        snapshot = ObjectSnapshot()
        require_status(
            self.library,
            self.runtime_library.shadowspill_object_snapshot(
                self.runtime._runtime_handle,
                self.runtime_object_id(alias_id),
                ctypes.byref(snapshot),
            ),
            f"snapshot object {alias_id}",
        )
        return int(snapshot.generation)

    def alias_for_runtime_object(self, object_id: int) -> str | None:
        """The plan-local alias bound to one runtime object, for failure reports."""

        for alias_id, bound in self._runtime_object_ids.items():
            if bound == object_id:
                return alias_id
        return None

    def registered_aliases(self) -> frozenset[str]:
        return frozenset(self._registered)

    def registered_runtime_objects(self) -> Mapping[str, int]:
        """Return plan aliases and their stable runtime object identities."""

        return {
            alias_id: runtime_object_id
            for alias_id, runtime_object_id in self._runtime_object_ids.items()
            if alias_id in self._registered
        }

    def adopt_registered(self, objects: Mapping[str, int]) -> None:
        """Adopt objects registered by a compatible provisional bridge."""

        for alias_id, runtime_object_id in objects.items():
            self._size(alias_id)
            self._registered.add(alias_id)
            self._record_runtime_object(alias_id, runtime_object_id)
            self._bind_plan_object(alias_id)
            if not self.requires_storage(alias_id):
                self._zero_generations.setdefault(alias_id, 0)

    def requires_storage(self, alias_id: str) -> bool:
        """Return whether an alias bundle owns any physical payload bytes."""

        return self._size(alias_id) != 0

    def zero_binding(self, alias_id: str) -> ObjectBinding:
        if self.requires_storage(alias_id):
            raise AssertionError("zero binding requested for a materialized alias")
        return ObjectBinding(
            plan_local_id(alias_id, "alias_"),
            self._zero_generations.setdefault(alias_id, 0),
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

    def _size(self, alias_id: str) -> int:
        try:
            return self._size_by_alias[alias_id]
        except KeyError as exc:
            raise PlanningError(f"unknown alias group {alias_id!r}") from exc

    def release_zero_generation(self, alias_id: str) -> None:
        """Forget a zero-byte alias's generation once it leaves with the caller."""

        self._zero_generations.pop(alias_id, None)


__all__ = ["PlanObjects"]
