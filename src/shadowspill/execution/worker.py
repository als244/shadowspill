"""Assemble admitted paths; startup failures reach all participating workers."""

from collections.abc import Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.planning.shared import shared_memory
from shadowspill.pytorch.resource_streams import ResourceStreams
from shadowspill.resources.description import resource_groups
from shadowspill.resources.inventory import resource_inventory
from shadowspill.resources.lifecycle import ResourceBindings
from shadowspill.runtime import Runtime

from .distributed.checks import coordinated
from .distributed.worker import DistributedWorker
from .paths import ExecutionPath, bind_path
from .resources import plan_inventory, seal_paths, worker_headroom, worker_streams
from .setup import open_path, open_worker

if TYPE_CHECKING:
    from .observations.collector import ParameterNorms


def create_paths(
    plan: Plan,
    device: int,
    external_headroom_bytes: int | None,
    plans: Mapping[str, Plan] | None = None,
    rank_map: Mapping[int, int] | None = None,
    parameter_norms: bool = False,
) -> tuple[
    Runtime,
    tuple[ExecutionPath, ...],
    dict[str, ExecutionPath],
    ResourceStreams,
    DistributedWorker | None,
    "ParameterNorms | None",
    ResourceBindings,
]:
    plans = dict(plans or {})
    inventory = plan_inventory(plan, plans)
    candidates = tuple(inventory.values())
    distributed = (
        DistributedWorker(inventory, rank_map)
        if plan.program.parallelism is not None
        and (
            plan.program.parallelism.world_size > 1
            or any(resource_groups(p.program.resources) for p in candidates)
        )
        else None
    )
    runtime, paths, named, observations = None, [], {}, None
    providers = ResourceBindings()
    try:

        def validate_observations() -> None:
            if type(parameter_norms) is not bool:
                raise TypeError("parameter_norms must be a bool")
            if parameter_norms and plan.program.bindings["kind"] != "step":
                raise ValueError(
                    "parameter norms currently require an optimizer step Plan"
                )

        coordinated(distributed, validate_observations)
        if distributed is not None and any(
            value != parameter_norms
            for value in distributed.groups.gather(parameter_norms)
        ):
            raise ValueError("workers selected different parameter norm options")
        joint = coordinated(
            distributed, lambda: shared_memory(plan, plans) if plans else None
        )
        headroom = coordinated(
            distributed, lambda: worker_headroom(candidates, external_headroom_bytes)
        )
        failure = None
        try:
            runtime = open_worker(plan, device, headroom)
        except Exception as error:
            if distributed is None:
                raise
            failure = str(error)
        if distributed is not None:
            distributed.groups.check(failure)
            distributed.prepare(device)
        assert runtime is not None
        descriptions, sources = coordinated(
            distributed, lambda: resource_inventory(p.program for p in candidates)
        )
        coordinated(
            distributed,
            lambda: providers.open(
                descriptions,
                sources,
                device=torch.device("cuda", device),
                groups=None if distributed is None else distributed.groups,
            ),
        )
        streams = coordinated(distributed, lambda: worker_streams(candidates, device))

        def bind() -> None:
            nonlocal observations
            memory, bridge = open_path(runtime, plan, device)
            other = plan._accumulation_plan
            extra = (
                set()
                if other is None
                else {r.alias_group_id for r in other.schedule.initial_residency}
            )
            if joint is not None:
                extra.update(
                    alias
                    for mapping in joint.aliases.values()
                    for alias in mapping.values()
                )
            paths.append(
                bind_path(
                    runtime,
                    plan,
                    memory,
                    bridge,
                    extra_initial=tuple(extra),
                    shared_slice_bytes=None
                    if joint is None
                    else joint.fixed_slice_bytes,
                    distributed=distributed,
                )
            )
            owner = memory.plan_handle
            if other is not None:
                memory, bridge = open_path(runtime, other, device, slab_owner=owner)
                paths.append(
                    bind_path(
                        runtime,
                        other,
                        memory,
                        bridge,
                        shared=paths[0].storage,
                        distributed=distributed,
                    )
                )
            for name, additional in plans.items():
                assert joint is not None
                memory, bridge = open_path(
                    runtime, additional, device, slab_owner=owner
                )
                named[name] = bind_path(
                    runtime,
                    additional,
                    memory,
                    bridge,
                    shared=paths[0].storage,
                    shared_aliases=joint.aliases[name],
                    distributed=distributed,
                )
            if parameter_norms:
                from .observations.collector import ParameterNorms

                observations = ParameterNorms(paths[0], distributed)
            seal_paths(candidates)

        coordinated(distributed, bind)
        return (
            runtime,
            tuple(paths),
            named,
            streams,
            distributed,
            observations,
            providers,
        )
    except BaseException:
        with ExitStack() as cleanup:
            if runtime is not None:
                cleanup.callback(runtime.close)
            if distributed is not None:
                cleanup.callback(distributed.close)
            cleanup.callback(providers.close)
            for path in [*paths, *named.values()]:
                assert runtime is not None
                cleanup.callback(path.close, runtime)
            if observations is not None:
                cleanup.callback(observations.close)
        raise
