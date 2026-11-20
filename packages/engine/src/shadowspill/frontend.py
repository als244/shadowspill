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
    """The framework object a runtime is opened with and keeps for its life.

    A framework frontend implements these methods; the PyTorch frontend is
    ShadowSpill's own. `~shadowspill.runtime.Runtime` takes one as its
    ``frontend`` and makes every framework call through it and through nothing
    else. Every method is off the per-task path: device selection when a
    runtime opens, the allocator install, a synchronize while a failure is
    reported, a search for live objects when a Plan closes.
    """

    # The device this frontend submits work to. A runtime is opened against one
    # ordinal and refuses any other, so the only judgement here is what a
    # caller's device argument means.

    def current_device_ordinal(self) -> int:
        """Return the ordinal the framework submits work to on this thread.

        Returns
        -------
        int
            The device ordinal.
        """

    def device_ordinal(self, value: object) -> int:
        """Return the device ordinal ``value`` names.

        Parameters
        ----------
        value : object
            An ordinal, a framework device or a device string.

        Returns
        -------
        int
            The device ordinal.

        Raises
        ------
        TypeError
            If ``value`` does not name a device.
        RuntimeConfigurationError
            If it names a device this frontend cannot execute on.
        """

    def select_device(self, ordinal: int) -> None:
        """Make ``ordinal`` the device this thread submits work to.

        Parameters
        ----------
        ordinal : int
            The device ordinal.
        """

    def synchronize(self, ordinal: int) -> None:
        """Wait for everything the framework has submitted to ``ordinal``.

        The runtime calls this while it reports a failure, so it must be safe
        after a fault and must raise rather than hang if the device is lost.

        Parameters
        ----------
        ordinal : int
            The device ordinal.
        """

    # The process allocator, installed once and never uninstalled. Frameworks
    # take a plain function for allocation, with no user pointer, so the runtime
    # cannot install itself and the steps below happen in this order: refuse an
    # unusable build, report missing operations, prepare, bootstrap the runtime,
    # activate, warm the provider.

    def refuse_unusable_build(self) -> None:
        """Raise if this build cannot host the runtime's allocator.

        A framework that has already initialized its device provider is one
        such build: the runtime's pool has to be the first thing on the device,
        or admission cannot account for what is there.
        """

    def missing_operations(self) -> Sequence[str]:
        """Return the operations this frontend needs and its library lacks.

        A library that registered none of them is the usual symptom of a stale
        build; the runtime names them in one error before installing anything.

        Returns
        -------
        sequence of str
            Names of the missing operations; empty when every one is
            registered.
        """

    def prepare_allocator(self, library_path: Path, record_stream_pointer: int) -> None:
        """Build the allocator over the runtime's adapter library.

        Preparing is separate from activating because the runtime has to
        bootstrap, and be checked, before anything allocates through it.

        Parameters
        ----------
        library_path : pathlib.Path
            The adapter library whose allocation functions the allocator
            calls.
        record_stream_pointer : int
            Address of the adapter's record-stream function, resolved by the
            runtime. A framework that tracks stream use through its allocator
            needs it; one that does not may ignore it.
        """

    def activate_allocator(self) -> None:
        """Make the prepared allocator the one this process allocates through.

        Irreversible for the life of the process.
        """

    def initialize_persistent_workspaces(self, device_ordinal: int) -> None:
        """Make the libraries create their retained workspaces in the pool now.

        A library that creates its workspace lazily does so mid-Plan, inside
        space that admission has already certified. Creating it while the pool
        is empty makes the cost explicit and the placement deterministic. A
        frontend with nothing to create may do nothing.

        Parameters
        ----------
        device_ordinal : int
            The runtime's device ordinal.
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
        """Find the framework objects whose storage lies inside each allocation.

        Only the framework can enumerate its objects and only the runtime knows
        whose bytes an address is, so the frontend walks its live objects and
        asks ``locate`` about each one's address.

        Parameters
        ----------
        allocations : sequence of PoolAllocation
            The pool allocations to explain.
        locate : callable
            Takes an address and returns the id of the allocation that owns
            it, or None when no pool allocation does.

        Returns
        -------
        dict of int to tuple
            For each allocation id asked about, the objects whose storage lies
            in it; an empty tuple when nothing does.
        """

    def retainers(
        self, held: Sequence[object], *, ignore: Sequence[object] = ()
    ) -> dict[int, tuple[str, ...]]:
        """Name where each of ``held`` is referenced from, so it can be freed.

        `occupants` answers which object occupies a range; this answers why
        that object is still reachable. It returns descriptions, never
        references.

        Parameters
        ----------
        held : sequence of object
            The objects to explain, usually from `occupants`.
        ignore : sequence of object, default ()
            The caller's own containers of those objects, which are not
            reported as referrers.

        Returns
        -------
        dict of int to tuple of str
            For each object's ``id``, descriptions of what references it.
        """

    def dematerialize(self, bindings: Sequence[object]) -> int:
        """Detach ``bindings`` from their leases.

        Afterwards none of them reads or writes pool memory, and the runtime
        may reclaim the bytes.

        Parameters
        ----------
        bindings : sequence of object
            Framework objects, as `occupants` returns them.

        Returns
        -------
        int
            How many were detached.
        """


__all__ = ["RuntimeFrontend"]
