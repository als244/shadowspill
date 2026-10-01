"""Bind explicit distributed state at public planning/import boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import Any

from torch import nn

from shadowspill.planner import StepDataOrdering
from shadowspill.pytorch.runtime import Runtime

from . import Distributed, current


def prepared_model_import[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Synchronize the imported state without writing into the source model."""
    return prepared(function, synchronize_result=True)


def prepared[**P, R](
    function: Callable[P, R], *, synchronize_result: bool = False
) -> Callable[P, R]:
    """Keep process groups alive on the runtime, never on temporary fake models."""

    @wraps(function)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        model = args[0] if args else kwargs.get("model")
        if not isinstance(model, nn.Module):
            return function(*args, **kwargs)
        runtime = kwargs.get("runtime")
        specification = kwargs.get("distributed")
        if specification is not None and not isinstance(specification, Distributed):
            raise TypeError("distributed must be a Distributed specification or None")
        active = current()
        if active is not None:
            if specification is not None and specification is not active.specification:
                raise ValueError(
                    "nested preparation supplied a different distributed binding"
                )
            return function(*args, **kwargs)
        if runtime is None:
            return function(*args, **kwargs)
        if not isinstance(runtime, Runtime):
            if specification is not None:
                raise TypeError("distributed preparation requires a PyTorch Runtime")
            return function(*args, **kwargs)
        bound = runtime._distributed_for(model, specification)
        if bound is None:
            return function(*args, **kwargs)
        with bound.activate():
            sharded = kwargs.get("shard_optimizer", bound.shard_optimizer)
            if not isinstance(sharded, bool):
                raise TypeError("shard_optimizer must be a bool")
            bound.shard_optimizer = sharded
            bound.control.agree("prepare/entry", function.__name__)
            bound.control.agree("prepare/shard_optimizer", bound.shard_optimizer)
            if (
                not synchronize_result
                and not bound.initialized
                and not any(p.is_meta for p in model.parameters())
            ):
                bound.synchronize_initial(model)
                bound.initialized = True
            result = function(*args, **kwargs)
            if isinstance(result, nn.Module):
                runtime._distributed_models[result] = bound
                if synchronize_result or not bound.initialized:
                    bound.synchronize_initial(result)
                    if bound.specification.sync_initial_state:
                        from shadowspill.pytorch.state.storage import (
                            refresh_persistent_state,
                        )

                        # Addressable state already changed in its pool. A remote
                        # pool needs the synchronized host view written back.
                        refresh_persistent_state(runtime, result)
                    bound.initialized = True
            bound.control.exchange("prepare/complete", function.__name__)
            return result

    return call


def collective_identity(local: dict[str, Any]) -> dict[str, Any]:
    """Bind an archive key to the full participant context and this rank."""
    bound = current()
    if bound is None:
        return local
    state = {
        "rank": bound.control.rank,
        "parameters": [value.record() for value in bound.parameters],
        "groups": bound.groups.aliases,
        "shard_optimizer": bound.shard_optimizer,
    }
    peers = bound.control.exchange(
        "archive/identity", {"local": local, "distributed": state}
    )
    return {**local, "distributed": {"rank": bound.control.rank, "participants": peers}}


def collective_misses(
    orderings: Sequence[StepDataOrdering], found: Mapping[StepDataOrdering, Any]
) -> tuple[StepDataOrdering, ...]:
    """Every rank builds an ordering if any participant lacks its archive."""
    bound = current()
    local = tuple(item for item in orderings if item not in found)
    if bound is None:
        return local
    bound.control.agree("archive/orderings", [item.label for item in orderings])
    peers = bound.control.exchange("archive/missing", [item.label for item in local])
    needed = {label for labels in peers for label in labels}
    return tuple(item for item in orderings if item.label in needed)
