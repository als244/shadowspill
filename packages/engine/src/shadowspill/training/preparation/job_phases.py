"""Resolved definition jobs use the ordinary build/profile/calibrate/plan APIs."""

import json
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.program import Program
from shadowspill.communication.info import CommunicationInfo
from shadowspill.diagnostics.progress import event, logged_phase, phase
from shadowspill.orchestration.launch.workers import workers

from .allocation_coverage import allocation_coverage
from .bundle import TrainingBundle
from .compiled_reuse import reuse_compiled_tasks
from .groups import capture_groups
from .job import BUNDLE_NAME, load_job
from .job_artifacts import (
    admitted_inventory,
    attempt,
    calibration_at,
    profile_at,
    program_at,
    reused,
)
from .measurement_reuse import REUSE_NAME
from .progress import candidate_title, describe_candidate


@logged_phase("Build")
def build(directory: str | Path) -> None:
    import torch
    from torch._inductor import config as inductor

    from .capture import build_template

    root, registry, inventory = load_job(directory)
    with workers(inventory.layouts.setup.context.parallelism.world_size) as group:
        groups = capture_groups(inventory)
        event(
            f"Workers: {group.count} rank(s), candidates run sequentially per rank; "
            f"CPU threads: {torch.get_num_threads()}, "
            f"interop: {torch.get_num_interop_threads()}, "
            f"Inductor compiler pool: {inductor.compile_threads}; "
            f"CPU affinity: {len(os.sched_getaffinity(0))}"
        )
        for index, peers in enumerate(groups, 1):
            key = peers[group.logical_rank]
            path = root / "programs" / f"{key}.ssp"

            def capture(
                *, key: str = key, path: Path = path, canonical_key: str = peers[0]
            ) -> None:
                program = reused(path, group, partial(program_at, root, inventory, key))
                if program is None:
                    program = build_template(
                        registry, inventory, key, rank_map=group.rank_map
                    )
                    program.save(path)
                if group.count > 1:
                    group.gather(None)  # All rank artifacts must exist before reuse.
                    canonical = Program.load(root / "programs" / f"{canonical_key}.ssp")
                    shared = reuse_compiled_tasks(program, canonical)
                    if shared is not program:
                        replace(
                            shared,
                            bindings={
                                **shared.bindings,
                                "allocation_coverage": allocation_coverage(
                                    inventory.templates[key], shared
                                ),
                            },
                        ).save(path)

            template = inventory.templates[key]
            with phase(candidate_title(template, index, len(groups))) as result:
                describe_candidate(inventory, template)
                result.rejected = not attempt(root, key, "build", capture, group)
    admitted_inventory(root, inventory)


@logged_phase("Profile")
def profile(directory: str | Path) -> None:
    from shadowspill import TaskProfileCache

    from .profile import profile_template

    root, _, original = load_job(directory)
    inventory, _ = admitted_inventory(root, original)
    with workers(inventory.layouts.setup.context.parallelism.world_size) as group:
        cache = TaskProfileCache(root / "cache")
        groups = capture_groups(inventory)
        for index, peers in enumerate(groups, 1):
            key = peers[group.logical_rank]
            path = root / "profiled" / f"{key}.ssp"

            def measure(*, key: str = key, path: Path = path) -> None:
                if (
                    reused(path, group, partial(profile_at, root, inventory, key))
                    is None
                ):
                    with phase("Load captured program"):
                        program = program_at(root, inventory, key)
                    profiled = profile_template(
                        inventory, key, program, rank_map=group.rank_map, cache=cache
                    )
                    with phase("Save task measurements"):
                        profiled.save(path)

            template = inventory.templates[key]
            with phase(candidate_title(template, index, len(groups))) as result:
                describe_candidate(inventory, template)
                result.rejected = not attempt(root, key, "profile", measure, group)
    admitted_inventory(root, original)


@logged_phase("Communication measurement")
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
        receipt = root / REUSE_NAME
        reusable = receipt.is_file() and json.loads(receipt.read_bytes()).get(
            "calibration_reused", False
        )
        if all(group.gather(reusable)):
            event(f"Reuse communication calibration for {len(groups)} candidates")
            for peers in groups:
                key = peers[group.logical_rank]
                if (
                    reused(
                        root / "communication" / f"{key}.ssp",
                        group,
                        partial(calibration_at, root, inventory, key),
                    )
                    is None
                ):
                    raise ValueError("staged calibration is missing a required case")
            return
        with phase("Load programs and collect communication requests"):
            program, collectives = communication_probes(
                program_at(root, inventory, key) for key in keys
            )
            event(
                f"Workers: {group.count}; candidates: {len(groups)}; "
                f"distinct collective configurations: {len(collectives)}"
            )
        with phase("Measure host transfers and collectives"):
            communication = measure_communication(
                program=program,
                collectives=collectives,
                rank_map=group.rank_map,
                options=ProfilingOptions.from_dict(
                    inventory.layouts.setup.config.preparation.profiling.to_dict()
                ),
            )
        cache = OverlapProfileCache(root / "cache")
        for index, peers in enumerate(groups, 1):
            key = peers[group.logical_rank]
            template = inventory.templates[key]
            with phase(candidate_title(template, index, len(groups))):
                describe_candidate(inventory, template)
                with phase("Load task measurements"):
                    profiled = profile_at(root, inventory, key)
                with phase("Measure compute and collective overlap"):
                    overlap = measure_overlap(
                        profiled, communication, rank_map=group.rank_map, cache=cache
                    )
                with phase("Save calibrated measurements"):
                    apply_profiles(
                        profiled.program, profiled.profiles.merge(overlap)
                    ).save(root / "calibrated" / f"{key}.ssp")
                    communication.save(root / "communication" / f"{key}.ssp")


@logged_phase("Plan")
def plan(directory: str | Path) -> None:
    from .family import plan_family
    from .parallel_search import parallel_search

    root, _, original = load_job(directory)
    inventory, rejected = admitted_inventory(root, original)
    with phase("Load measured programs and communication"):
        measured = {
            key: ProfiledProgram.load(root / "calibrated" / f"{key}.ssp")
            for key in inventory.templates
        }
        communication = {
            key: CommunicationInfo.load(root / "communication" / f"{key}.ssp")
            for key in inventory.templates
        }
    with phase("Search layouts and orderings"):
        searched = parallel_search(
            root, inventory, sources=measured, communication=communication
        )
        family = plan_family(
            inventory,
            measured,
            communication,
            prior_rejections=rejected,
            search_results=searched,
        )
    with phase("Assemble prepared Plan family"):
        bundle = TrainingBundle.from_prepared(family, measured)
    with phase("Save prepared Plans"):
        bundle.save(root / BUNDLE_NAME)
        event(
            f"Archive size: {(root / BUNDLE_NAME).stat().st_size / (1 << 20):,.2f} MiB"
        )
