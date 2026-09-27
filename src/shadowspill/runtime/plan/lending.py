"""Slabs lent to planning between calls.

Planning profiles and compiles on real values in the execution pool, and it
runs between calls, when no plan is using the slab it reserved. Left
reserved, those slabs would leave a plan made after another only the bytes
they do not cover -- too few, once a training step has planned itself into
the whole pool. So a planning call borrows them: every slab an admitted plan
reserved is lent back to its pool when planning begins, and taken back where
it was before a layout is admitted and before any call begins, which the
runtime refuses while anything allocated in the meantime still lies in it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shadowspill.runtime.failures import RuntimeExecutionError

from ..abi import runtime_library
from ..configuration import RuntimeConfigurationError

if TYPE_CHECKING:
    from ..core import Runtime


def lend_reserved_slabs(runtime: Runtime) -> None:
    """Lend every slab an admitted plan reserved to the planning call beginning."""

    installed = runtime._installed
    owners = sorted(
        handle
        for handle, reserved in installed.admitted_layout_bytes.items()
        if reserved > 0
    )
    if not owners:
        return
    library = runtime_library()
    # Lending is refused while anything a call placed is live or retiring.
    library.shadowspill_runtime_wait_idle(runtime._runtime_handle)
    for handle in owners:
        status = int(library.shadowspill_plan_lend_fixed_layout(handle))
        if status != 0:
            plan_id = int(library.shadowspill_plan_id(handle))
            raise RuntimeConfigurationError(
                f"lending the slab plan {plan_id} reserved to planning failed "
                f"(status {status}): plan only between calls"
            )
        installed.lent_slabs.add(handle)


def take_back_lent_slabs(runtime: Runtime) -> tuple[int, ...]:
    """Take back, where each was, the slabs lent to planning; return the ids of
    the plans whose slab something allocated since still lies in."""

    installed = runtime._installed
    if not installed.lent_slabs:
        return ()
    library = runtime_library()
    # What planning freed retires first, or its bytes would still be taken.
    library.shadowspill_runtime_wait_idle(runtime._runtime_handle)
    kept: list[int] = []
    for handle in sorted(installed.lent_slabs):
        if int(library.shadowspill_plan_reclaim_fixed_layout(handle)) == 0:
            installed.lent_slabs.discard(handle)
        else:
            kept.append(int(library.shadowspill_plan_id(handle)))
    return tuple(kept)


def require_lent_slabs_back(runtime: Runtime, operation: str) -> None:
    """Take back every lent slab before ``operation``, or refuse it."""

    kept = take_back_lent_slabs(runtime)
    if kept:
        raise RuntimeExecutionError(
            f"cannot {operation}: allocations made while planning borrowed the "
            f"slabs of plans {list(kept)} still lie in them; release them first"
        )
