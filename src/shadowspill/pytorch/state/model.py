"""Public relocation operations for registered model state."""

from __future__ import annotations

from typing import cast

import torch.nn as nn

from shadowspill.pytorch.runtime_adapter.runtime import Runtime

from .model_copy import copy_model_with_spill_storages
from .storage import (
    NamedTensor,
    externalize_tensors,
    own_persistent_state,
    persistent_state,
    register_tensor_storages,
    unregister_tensor_storages,
)


def relocate_model_state[ModelT: nn.Module](
    model: ModelT,
    *,
    runtime: Runtime,
    pool: str,
    release_source: bool = False,
) -> ModelT:
    """Return a model copy whose registered state resides in ``pool``.

    The returned module has distinct Python module and tensor identities while
    preserving topology, ties, views, values, and metadata. Its registered
    tensors point directly into runtime-owned spill leases.

    ``release_source=False`` leaves ``model`` unchanged and retains it as the
    source owner. With ``release_source=True``, neither the returned model nor
    ShadowSpill retains the input model. Assign the return value back to the
    same variable to let Python release the source when no other references
    exist::

        model = relocate_model_state(
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
        relocated, relocated_storages = copy_model_with_spill_storages(model, storages)
        own_persistent_state(
            relocated,
            runtime=runtime,
            pool=pool,
            storages=relocated_storages,
            source_owner=None if release_source else model,
        )
    except BaseException:
        unregister_tensor_storages(storages, runtime=runtime)
        raise
    return cast(ModelT, relocated)


def require_model_state_for_plan(
    model: nn.Module,
    *,
    runtime: Runtime,
    pool: str,
) -> None:
    """Require model state to have been explicitly relocated into ``pool``."""

    existing = persistent_state(runtime, model)
    if existing is None:
        raise RuntimeError(
            "model state is not owned by this Runtime; call "
            "relocate_model_state(model, runtime=runtime, pool=spill, ...) "
            "before planning"
        )
    if existing.pool != pool:
        raise RuntimeError(
            f"model state is in pool {existing.pool!r}, not requested {pool!r}"
        )


def externalize_model_state[ModelT: nn.Module](
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

    externalize_tensors(model, runtime=runtime, release_runtime=release_runtime)
    return model


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
    "externalize_model_state",
    "relocate_model_state",
    "require_model_state_for_plan",
]
