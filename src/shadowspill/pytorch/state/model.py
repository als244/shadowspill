"""Public model-state import and export operations."""

from __future__ import annotations

import os
from typing import cast

import torch
import torch.nn as nn

from shadowspill.pytorch.runtime_adapter.runtime import Runtime

from .model_copy import copy_model_with_runtime_storages
from .storage import (
    NamedTensor,
    export_tensors,
    import_state_from_file,
    import_tensors,
    own_persistent_state,
    persistent_state,
    read_state,
    register_tensor_storages,
    release_persistent_tensors,
    unregister_tensor_storages,
)


def import_model_state[ModelT: nn.Module](
    model: ModelT,
    *,
    runtime: Runtime,
    pool: str,
    release_source: bool = True,
) -> ModelT:
    """Return a model copy whose registered state resides in ``pool``.

    The returned module has distinct Python module and tensor identities while
    preserving topology, ties, views, values, and metadata. Its registered
    tensors point directly into runtime-owned pool leases.

    The default ``release_source=True`` means neither the returned model nor
    ShadowSpill retains the input model. Assign the return value back to the
    same variable to let Python release the source when no other references
    exist. Set ``release_source=False`` only when the original model must
    remain available independently::

        model = import_model_state(
            model, runtime=runtime, pool="spill", release_source=True
        )
    """

    if persistent_state(runtime, model) is not None:
        raise RuntimeError("model state is already owned by this Runtime")
    storages = register_tensor_storages(
        _model_tensors(model),
        runtime=runtime,
        pool=pool,
    )
    try:
        imported, imported_storages = copy_model_with_runtime_storages(model, storages)
        own_persistent_state(
            imported,
            runtime=runtime,
            pool=pool,
            storages=imported_storages,
            source_owner=None if release_source else model,
        )
    except BaseException:
        unregister_tensor_storages(storages, runtime=runtime)
        raise
    return cast(ModelT, imported)


def import_model_state_from_file(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    runtime: Runtime,
    pool: str,
) -> None:
    """Fill the model's state in ``pool`` from a checkpoint on disk.

    The values go from the file into pool memory without a whole copy of the
    checkpoint appearing in ordinary host memory first. Unlike
    :func:`import_model_state` this rebinds the model that was passed rather
    than returning a copy, so the caller keeps using the object it has.
    """

    if persistent_state(runtime, model) is not None:
        raise RuntimeError("model state is already owned by this Runtime")
    import_state_from_file(
        model, _model_tensors(model), path, runtime=runtime, pool=pool
    )


def require_model_state_for_plan(
    model: nn.Module,
    *,
    runtime: Runtime,
    pool: str,
) -> None:
    """Require model state to have been explicitly imported into ``pool``.

    For entry points that return no callable, so nothing would own or release
    state imported on the caller's behalf. Planning calls that do return one
    use :func:`adopt_model_state_for_plan` instead.
    """

    existing = persistent_state(runtime, model)
    if existing is None:
        raise RuntimeError(
            "model state is not owned by this Runtime; call "
            "import_model_state(model, runtime=runtime, pool=spill, ...) "
            "before planning"
        )
    if existing.pool != pool:
        raise RuntimeError(
            f"model state is in pool {existing.pool!r}, not requested {pool!r}"
        )


def adopt_model_state_for_plan(
    model: nn.Module,
    *,
    runtime: Runtime,
    pool: str,
    owning_plan: int,
) -> bool:
    """Give one plan the model state it needs, and say whether it owns it.

    State the caller imported is adopted as it stands and outlives the plan.
    State the caller did not import is imported here, in place, and belongs
    to the plan: closing the plan releases it, so the caller reads what it
    wants before that. Returns whether the plan now owns the state.

    Importing in place rebinds the module that was passed rather than
    returning a copy, so a planning call never hands back a different object.
    """

    existing = persistent_state(runtime, model)
    if existing is not None:
        if existing.pool != pool:
            raise RuntimeError(
                f"model state is in pool {existing.pool!r}, not requested {pool!r}"
            )
        return False
    import_tensors(
        model,
        _model_tensors(model),
        runtime=runtime,
        pool=pool,
        release_source=True,
        owning_plan=owning_plan,
        _allow_in_progress_plan=True,
    )
    return True


def export_model_state[ModelT: nn.Module](
    model: ModelT,
    *,
    runtime: Runtime,
    release_runtime: bool = False,
) -> ModelT:
    """Copy authoritative model bytes into ordinary CPU allocations.

    The existing registered tensor identities are rebound to the new ordinary
    CPU storages. ``release_runtime=False`` retains the persistent runtime
    objects for later reuse; ``release_runtime=True`` releases them after the
    copy.
    """

    export_tensors(model, runtime=runtime, release_runtime=release_runtime)
    return model


def read_model_state(
    model: nn.Module,
    *,
    runtime: Runtime,
    copy: bool = True,
) -> dict[str, torch.Tensor]:
    """Return the model's current values without rebinding its tensors.

    Answerable while a plan holds the model, which ``export_model_state`` is
    not. ``copy`` has the meaning it has in :func:`read_state`: copied values
    are yours and keep what they held, uncopied ones view the pool and are
    read-only and only current until the plan runs again.
    """

    return read_state(model, _model_tensors(model), runtime=runtime, copy=copy)


def release_model_state(
    model: nn.Module,
    *,
    runtime: Runtime,
) -> None:
    """Release imported model state without materializing a CPU copy.

    Unlike ``export_model_state()``, no ordinary CPU allocation is created
    and the module's registered tensors are not rebound: they become invalid
    the moment their pool leases are released, so the module must be
    discarded afterward. This is the teardown operation for callers that no
    longer need the state, such as throughput qualification on hosts that
    cannot hold an additional anonymous model copy beside the pinned spill
    arena. Export remains the correct operation when the model is used
    again. A model that is not owned by ``runtime`` is left unchanged.
    """

    runtime._require_state_operation_allowed()
    release_persistent_tensors(model, runtime=runtime)


def _model_tensors(model: nn.Module) -> tuple[NamedTensor, ...]:
    return (
        *(
            NamedTensor(name, value)
            for name, value in model.named_parameters(remove_duplicate=False)
        ),
        *(
            NamedTensor(name, value)
            for name, value in model.named_buffers(remove_duplicate=False)
        ),
    )


__all__ = [
    "adopt_model_state_for_plan",
    "export_model_state",
    "import_model_state",
    "import_model_state_from_file",
    "read_model_state",
    "release_model_state",
    "require_model_state_for_plan",
]
