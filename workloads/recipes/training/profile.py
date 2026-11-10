"""Measure each exact candidate, with its declared resource participants."""

import sys

from benchmarking.quickstart.phases.distributed import workers
from shadowspill import Program, apply_profiles, measure_communication, measure_program
from workloads.recipes.phases import settings

from .candidates import attempt_candidate, available, program_paths
from .profile_reuse import completed


def profile_candidate(root, options, candidate, group, timing):
    for kind in ("training", "evaluation") if options["eval_every"] else ("training",):
        relative = f"{candidate['path']}/{kind}.ssp"
        print(f"profiling {relative}", flush=True)
        program = Program.load(root / "programs" / relative)
        profiled = completed(root / "profiles" / relative, program, options, group)
        if profiled is None:
            profiles = measure_program(
                program,
                rank_map=group.rank_map if group.count > 1 else None,
                external_headroom_bytes=options["external_headroom_mib"] << 20,
                **timing,
            )
            if group.rank == 0:
                profiles.save(root / "profiles" / relative)
                profiled = apply_profiles(program, profiles)
        elif group.rank == 0:
            print(f"reusing completed measurements for {relative}", flush=True)
        if group.rank == 0:
            profiled.save(root / "profiled" / relative)
    return True


def run(directory):
    root, options = settings(directory)
    timing = {key: options[key] for key in ("warmup_seconds", "measure_seconds")}
    with workers({}, {"world_size": options["preparation_ranks"]}) as group:
        for candidate in available(root, options):
            attempt_candidate(
                root,
                candidate,
                "profile",
                lambda candidate=candidate: profile_candidate(
                    root, options, candidate, group, timing
                ),
                group,
            )
        if options["ranks"] == 1:
            communication = measure_communication(**timing)
            for relative in program_paths(options, root=root):
                communication.save(root / "communication" / relative)


if __name__ == "__main__":
    run(sys.argv[1])
