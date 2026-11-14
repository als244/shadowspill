"""Resolved definition jobs use the ordinary build/profile/calibrate/plan APIs."""

from functools import partial
from pathlib import Path

from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication.info import CommunicationInfo
from shadowspill.orchestration.launch.workers import workers

from .bundle import TrainingBundle
from .groups import capture_groups
from .job import BUNDLE_NAME, load_job
from .job_artifacts import admitted_inventory, attempt, profile_at, program_at, reused


def build(directory: str | Path) -> None:
    from .capture import build_template

    root, registry, inventory = load_job(directory)
    with workers(inventory.layouts.setup.context.parallelism.world_size) as group:
        for peers in capture_groups(inventory):
            key = peers[group.logical_rank]
            path = root / "programs" / f"{key}.ssp"

            def capture(*, key: str = key, path: Path = path) -> None:
                if (
                    reused(path, group, partial(program_at, root, inventory, key))
                    is None
                ):
                    build_template(
                        registry, inventory, key, rank_map=group.rank_map
                    ).save(path)

            attempt(root, key, "build", capture, group)
    admitted_inventory(root, inventory)


def profile(directory: str | Path) -> None:
    from shadowspill import TaskProfileCache

    from .profile import profile_template

    root, _, original = load_job(directory)
    inventory, _ = admitted_inventory(root, original)
    with workers(inventory.layouts.setup.context.parallelism.world_size) as group:
        cache = TaskProfileCache(root / "cache")
        for peers in capture_groups(inventory):
            key = peers[group.logical_rank]
            path = root / "profiled" / f"{key}.ssp"

            def measure(*, key: str = key, path: Path = path) -> None:
                if (
                    reused(path, group, partial(profile_at, root, inventory, key))
                    is None
                ):
                    program = program_at(root, inventory, key)
                    profile_template(
                        inventory, key, program, rank_map=group.rank_map, cache=cache
                    ).save(path)

            attempt(root, key, "profile", measure, group)
    admitted_inventory(root, original)


def communicate(directory: str | Path) -> None:
    from shadowspill import (
        OverlapProfileCache,
        apply_profiles,
        measure_communication,
        measure_overlap,
    )
    from shadowspill.orchestration.communication import communication_probes
    from shadowspill.task.profiling import ProfilingOptions

    root, _, original = load_job(directory)
    inventory, _ = admitted_inventory(root, original)
    with workers(inventory.layouts.setup.context.parallelism.world_size) as group:
        groups = capture_groups(inventory)
        keys = tuple(dict.fromkeys(peers[group.logical_rank] for peers in groups))
        program, collectives = communication_probes(
            program_at(root, inventory, key) for key in keys
        )
        communication = measure_communication(
            program=program,
            collectives=collectives,
            rank_map=group.rank_map,
            options=ProfilingOptions.from_dict(
                inventory.layouts.setup.config.preparation.profiling.to_dict()
            ),
        )
        cache = OverlapProfileCache(root / "cache")
        for peers in groups:
            key = peers[group.logical_rank]
            profiled = profile_at(root, inventory, key)
            overlap = measure_overlap(
                profiled, communication, rank_map=group.rank_map, cache=cache
            )
            apply_profiles(profiled.program, profiled.profiles.merge(overlap)).save(
                root / "calibrated" / f"{key}.ssp"
            )
            communication.save(root / "communication" / f"{key}.ssp")


def plan(directory: str | Path) -> None:
    from .family import plan_family

    root, registry, original = load_job(directory)
    inventory, rejected = admitted_inventory(root, original)
    measured = {
        key: ProfiledProgram.load(root / "calibrated" / f"{key}.ssp")
        for key in inventory.templates
    }
    communication = {
        key: CommunicationInfo.load(root / "communication" / f"{key}.ssp")
        for key in inventory.templates
    }
    family = plan_family(inventory, measured, communication, prior_rejections=rejected)
    bundle = TrainingBundle.from_prepared(family, measured)
    bundle.recheck(registry)
    bundle.save(root / BUNDLE_NAME)
