"""What the neutral side needs a framework frontend to do for it.

ShadowSpill's runtime, planning, profiling and diagnostics are framework-neutral
and reach a framework only through `RuntimeFrontend`. PyTorch is one
implementation of it, in :mod:`shadowspill.pytorch.frontend`; nothing here
imports it, or any framework, so reading this module is the whole answer to what
a second frontend has to provide.

Twelve methods, in three groups: the device the frontend submits to, the process
allocator it installs, and the framework objects standing on a runtime lease.
Every one is cold path -- device selection when a runtime opens, the allocator
install at bootstrap, a synchronize while a failure is being reported, a sweep
for live objects when a plan closes. The per-task boundary is not here and does
not come here: it stays inside the frontend, one call into its adapter.

Values crossing these calls are already neutral -- ordinals, byte counts,
allocation ids, paths. The one exception is deliberate: `occupants` hands back
framework objects, because all the neutral side does with them is count them and
give them straight back to `dematerialize`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from shadowspill.runtime.occupancy import PoolAllocation


@runtime_checkable
class RuntimeFrontend(Protocol):
    """The one framework object a runtime is opened with, and keeps for its life.

    Every framework call a runtime makes goes through this and through nothing
    else, so `runtime.frontend` is the whole of the runtime's dependence on a
    framework.
    """

    # The device this frontend submits work to. A runtime is opened against one
    # ordinal and refuses any other, so the only judgement here is what a
    # caller's device argument means.

    def current_device_ordinal(self) -> int:
        """The ordinal the framework would submit work to on this thread."""

    def device_ordinal(self, value: object) -> int:
        """The ordinal ``value`` names: an ordinal, a device, or a device string.

        Raise :class:`TypeError` when it does not name a device at all, and
        :class:`~shadowspill.runtime.RuntimeConfigurationError` when it names a
        device this frontend cannot execute on.
        """

    def select_device(self, ordinal: int) -> None:
        """Make ``ordinal`` the device this thread submits work to."""

    def synchronize(self, ordinal: int) -> None:
        """Wait for everything the framework has submitted to ``ordinal``.

        Called while a failure is being reported, so it must be safe after a
        fault and must raise rather than hang if the device is lost.
        """

    # The process allocator, installed once and never uninstalled. Frameworks
    # take a plain function for allocation, with no user pointer, so the runtime
    # cannot install itself and the steps below happen in this order: refuse an
    # unusable build, report missing operations, prepare, bootstrap the runtime,
    # activate, warm the provider.

    def refuse_unusable_build(self) -> None:
        """Raise now if this build cannot host the runtime's allocator.

        A framework that has already initialized its device provider is one such
        refusal: the runtime's pool has to be the first thing on the device, or
        admission cannot account for what is there.
        """

    def missing_operations(self) -> Sequence[str]:
        """The operations this frontend needs its library to register, and did
        not find.

        A library that registered none of them is the usual symptom of a stale
        build, and is worth one clear message rather than an attribute error at
        the first task boundary.
        """

    def prepare_allocator(self, library_path: Path, record_stream_pointer: int) -> None:
        """Build the allocator over ``library_path``, with that callback.

        ``record_stream_pointer`` is the adapter's record-stream entry point,
        resolved by the runtime; a framework that tracks stream use through its
        allocator needs it, and one that does not may ignore it. Preparing is
        separate from activating because the runtime has to bootstrap, and be
        checked, before anything allocates through it.
        """

    def activate_allocator(self) -> None:
        """Make the prepared allocator the one this process allocates through.

        Irreversible for the process lifetime.
        """

    def initialize_provider_workspaces(self, device_ordinal: int) -> None:
        """Force the device provider's retained workspaces into the pool now.

        A provider that creates its workspace lazily creates it in the middle of
        a plan, splitting a slab admission had already certified. Doing it here,
        while the pool is empty, makes the cost explicit and its placement
        deterministic. A frontend with nothing to warm may do nothing.
        """

    # The framework objects standing on a runtime lease. A lease is bytes in a
    # pool; a binding is the object standing on those bytes. The runtime
    # reclaims the lease and only the frontend can reach the binding, so when a
    # plan closes the runtime asks who is still there and asks them to step off.
    # Addresses are compared rather than references kept: holding a reference
    # would extend the lifetime of exactly what is being investigated.

    def occupants(
        self,
        allocations: Sequence[PoolAllocation],
        locate: Callable[[int], int | None],
    ) -> dict[int, tuple[object, ...]]:
        """The framework objects whose storage lies inside each allocation.

        The frontend walks its own live objects and asks ``locate`` which
        allocation owns an address: only the framework can enumerate its
        objects, and only the runtime knows whose bytes an address is. Every
        allocation asked about appears in the result, mapped to an empty tuple
        when nothing holds it.
        """

    def retainers(
        self, held: Sequence[object], *, ignore: Sequence[object] = ()
    ) -> dict[int, tuple[str, ...]]:
        """Where each of ``held`` is referenced from, named so it can be freed.

        `occupants` answers which object occupies a range; this answers why that
        object is still reachable. Keyed by ``id``, so a caller reads answers
        back with the objects it passed in. Descriptions, never references.
        """

    def dematerialize(self, bindings: Sequence[object]) -> int:
        """Detach ``bindings`` from their leases; return how many were detached.

        After this returns, none of them reads or writes pool memory and the
        runtime is free to reclaim the bytes.
        """


__all__ = ["RuntimeFrontend"]
