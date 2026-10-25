"""Assemble admitted paths; startup failures reach all participating workers."""

from collections.abc import Mapping

from shadowspill.artifacts.plan import Plan
from shadowspill.planning.shared import shared_memory
from shadowspill.pytorch.resource_streams import ResourceStreams
from shadowspill.runtime import Runtime

from .distributed.checks import coordinated
from .distributed.worker import DistributedWorker
from .paths import ExecutionPath, bind_path
from .resources import plan_inventory, seal_paths, worker_headroom, worker_streams
from .setup import open_path, open_worker


def create_paths(
    plan: Plan,
    device: int,
    external_headroom_bytes: int | None,
    plans: Mapping[str, Plan] | None = None,
    rank_map: Mapping[int, int] | None = None,
) -> tuple[
    Runtime,
    tuple[ExecutionPath, ...],
    dict[str, ExecutionPath],
    ResourceStreams,
    DistributedWorker | None,
]:
    plans = dict(plans or {})
    inventory = plan_inventory(plan, plans)
    candidates = tuple(inventory.values())
    distributed = (
        DistributedWorker(inventory, rank_map)
        if plan.program.parallelism is not None
        and plan.program.parallelism.world_size > 1
        else None
    )
    runtime, paths, named = None, [], {}
    try:
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
        streams = coordinated(distributed, lambda: worker_streams(candidates, device))

        def bind() -> None:
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
            seal_paths(candidates)

        coordinated(distributed, bind)
        return runtime, tuple(paths), named, streams, distributed
    except BaseException:
        for path in reversed([*paths, *named.values()]):
            assert runtime is not None
            path.close(runtime)
        if distributed is not None:
            distributed.close()
        if runtime is not None:
            runtime.close()
        raise
