"""Public model-state import and export operations."""

from __future__ import annotations

import os
from typing import cast

import torch
import torch.nn as nn

from shadowspill.pytorch.runtime_adapter.runtime import MemoryPool, Runtime

from .model_copy import copy_model_with_runtime_storages
from .storage import (
    NamedTensor,
    export_tensors,
    import_state_from_file,
    import_tensors,
    own_persistent_state,
    persistent_state,
    pool_backed_tensor,
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
    on_meta = _meta_names(model)
    if on_meta:
        materialized = [
            item.name for item in _model_tensors(model) if item.name not in on_meta
        ]
        if materialized:
            raise RuntimeError(
                "model mixes meta and materialized state, so it is unclear "
                "which values are real: "
                f"{', '.join(sorted(materialized)[:4])}"
                f"{' and more' if len(materialized) > 4 else ''}. "
                "Construct the whole model under torch.device('meta')."
            )
        return _materialize_meta_model(model, runtime=runtime, pool=pool)
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


def _meta_names(model: nn.Module) -> frozenset[str]:
    """The names of this model's tensors that carry no storage."""

    return frozenset(
        item.name for item in _model_tensors(model) if item.tensor.is_meta
    )


def _modules_without_reset(model: nn.Module) -> tuple[str, ...]:
    """Modules that own state directly and cannot initialize it.

    Stock PyTorch modules implement ``reset_parameters``; a custom module that
    holds its own parameters has to as well, or its values would be whatever
    the pool memory happened to contain -- a wrong answer rather than a
    failure.
    """

    offenders: list[str] = []
    for name, module in model.named_modules():
        owns = any(item is not None for item in module._parameters.values()) or any(
            item is not None for item in module._buffers.values()
        )
        if owns and not hasattr(module, "reset_parameters"):
            offenders.append(name or type(module).__name__)
    return tuple(offenders)


def _materialize_meta_model[ModelT: nn.Module](
    model: ModelT,
    *,
    runtime: Runtime,
    pool: str,
) -> ModelT:
    """Give a meta model pool storage, then let it initialize itself.

    The model is rebound in place rather than copied: a meta model holds no
    values, so there is nothing to copy and nothing to release. Every tensor
    is allocated in the pool and then written by ``reset_parameters``, so no
    host memory proportional to the model is ever allocated.
    """

    offenders = _modules_without_reset(model)
    if offenders:
        raise RuntimeError(
            "these modules own state but do not implement reset_parameters, so "
            "materializing them would leave uninitialized values: "
            f"{', '.join(offenders[:4])}"
            f"{' and more' if len(offenders) > 4 else ''}"
        )
    selected = _require_pool(runtime, pool)
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is None or not parameter.is_meta:
                continue
            view, _ = pool_backed_tensor(
                runtime,
                selected,
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
            )
            module._parameters[name] = nn.Parameter(
                view, requires_grad=parameter.requires_grad
            )
        for name, buffer in list(module._buffers.items()):
            if buffer is None or not buffer.is_meta:
                continue
            view, _ = pool_backed_tensor(
                runtime, selected, shape=tuple(buffer.shape), dtype=buffer.dtype
            )
            module._buffers[name] = view
    import_tensors(
        model,
        _model_tensors(model),
        runtime=runtime,
        pool=pool,
        release_source=True,
    )
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    return model


def _require_pool(runtime: Runtime, pool: str) -> MemoryPool:
    try:
        return runtime.pools[pool]
    except KeyError as exc:
        raise RuntimeError(f"no pool named {pool!r} on this Runtime") from exc


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
