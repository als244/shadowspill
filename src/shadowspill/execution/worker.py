"""Assemble admitted paths; startup failures reach all participating workers."""

from collections.abc import Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.memory import SpillPool
from shadowspill.planning.shared import shared_memory
from shadowspill.pytorch.numerics import validate_numerical_settings
from shadowspill.pytorch.resource_streams import ResourceStreams
from shadowspill.resources.description import resource_groups
from shadowspill.resources.inventory import resource_inventory
from shadowspill.runtime import Runtime

from .distributed.checks import coordinated
from .distributed.sequence import check_collective_order
from .distributed.worker import DistributedWorker
from .paths import ExecutionPath, bind_path, plan_numerical_contracts
from .providers import WorkerProviders
from .resources import (
    plan_inventory,
    seal_paths,
    worker_external_baseline,
    worker_headroom,
    worker_streams,
)
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
    spill: SpillPool | None = None,
    reject_overbudget: bool = False,
) -> tuple[
    Runtime,
    tuple[ExecutionPath, ...],
    dict[str, ExecutionPath],
    ResourceStreams,
    DistributedWorker | None,
    "ParameterNorms | None",
    WorkerProviders,
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
    providers = WorkerProviders(device)
    try:

        def validate_configuration() -> None:
            if type(parameter_norms) is not bool:
                raise TypeError("parameter_norms must be a bool")
            if parameter_norms and plan.program.bindings["kind"] != "step":
                raise ValueError("parameter norms require an optimizer step Plan")
            for candidate in candidates:
                validate_numerical_settings(plan_numerical_contracts(candidate))

        coordinated(distributed, validate_configuration)
        if distributed is not None and any(
            value != parameter_norms
            for value in distributed.groups.gather(parameter_norms)
        ):
            raise ValueError("workers selected different parameter norm options")
        if distributed is not None:
            from shadowspill.parallelism.agreement import agreement_digest

            agreements = distributed.groups.gather(agreement_digest(plan.program))
            if any(value != agreements[0] for value in agreements):
                raise ValueError(
                    "ranks run Programs that disagree on what they exchange: the "
                    "parallelism, the tensors or their shards and geometry differ"
                )
            check_collective_order(distributed.groups, inventory)
        joint = coordinated(
            distributed,
            lambda: (
                shared_memory(plan, plans)
                if plans or plan._accumulation_plan is not None
                else None
            ),
        )
        headroom = coordinated(
            distributed, lambda: worker_headroom(candidates, external_headroom_bytes)
        )
        external_baseline = coordinated(
            distributed, lambda: worker_external_baseline(candidates)
        )
        failure = None
        try:
            runtime = open_worker(
                plan,
                device,
                headroom,
                spill,
                reject_overbudget,
                external_baseline,
                logical_rank=0
                if distributed is None
                else distributed.groups.logical_rank,
            )
        except Exception as error:
            if distributed is None:
                raise
            failure = str(error)
        if distributed is not None:
            distributed.groups.check(failure)
            distributed.prepare(device)
        assert runtime is not None
        # Allocator selection must precede PyTorch CUDA initialization. The
        # cold pool already reserves the profiles' mature external footprint,
        # including modules. Load them now, before fixed-layout Plan admission
        # and the final physical-budget seal, never inside a task.
        coordinated(distributed, lambda: providers.prepare(candidates))
        descriptions, sources = coordinated(
            distributed, lambda: resource_inventory(p.program for p in candidates)
        )
        coordinated(
            distributed,
            lambda: providers.resources.open(
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
                    preparations=providers.preparations,
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
                        preparations=providers.preparations,
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
                    preparations=providers.preparations,
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
    except BaseException as error:
        try:
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
        except BaseException as cleanup_error:
            # As after a failed run: once the runtime has closed, the setup
            # error is what is raised, and a runtime that latched it refuses
            # the plans' own steps. A cleanup that could not close the runtime
            # raises; ExitStack drops the setup error from its chain, so it is
            # named as the cause here.
            if runtime is None or not runtime._closed:
                raise cleanup_error from error
        raise
