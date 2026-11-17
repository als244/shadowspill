"""Direct phase preparation for the two-rank EP arithmetic qualification."""

import argparse
import json
from pathlib import Path

from shadowspill import (
    ProfiledProgram,
    Program,
    apply_profiles,
    build_step,
    measure_communication,
    measure_overlap,
    measure_program,
)
from shadowspill.orchestration.communication import communication_probes
from shadowspill.orchestration.launch.workers import workers
from shadowspill.orchestration.profiling import profiling_options


def read(path):
    return json.loads(Path(path).read_text())


def experiment(request, *, logical_rank=0):
    from workload_tests.cases.expert_parallel import experiment as setup

    return setup(
        device="cuda:0", logical_rank=logical_rank, **request["model_settings"]
    )


def run(root, phase):
    request = read(root / "request.json")
    filename = "candidate-0000.ssp"
    with workers(2, rank_map=request["rank_map"]) as group:
        if phase == "build":
            setup = experiment(request, logical_rank=group.logical_rank)
            model = setup["model_factory"]()
            candidate = next(iter(setup["candidates"].values()))
            program = build_step(
                model,
                optimizer=setup["optimizer"](model.parameters()),
                objective=setup["objective"],
                example_calls=[(args, {}) for args in candidate["args"]],
                target="cuda:0",
                rank_map=group.rank_map,
                **setup["build_options"],
            )
            if group.rank == 0:
                program.save(root / "programs" / filename)
            return
        program = Program.load(root / "programs" / filename)
        timing = profiling_options(request)
        if phase == "profile":
            profiles = measure_program(
                program,
                device="cuda:0",
                rank_map=group.rank_map,
                external_headroom_bytes=2 << 30,
                options=timing,
            )
            if group.rank == 0:
                apply_profiles(program, profiles).save(root / "profiled" / filename)
            return
        probe, collectives = communication_probes((program,))
        communication = measure_communication(
            program=probe,
            collectives=collectives,
            rank_map=group.rank_map,
            device="cuda:0",
            options=timing,
        )
        profiled = ProfiledProgram.load(root / "profiled" / filename)
        overlap = measure_overlap(
            profiled, communication, device="cuda:0", rank_map=group.rank_map
        )
        if group.rank == 0:
            communication.save(root / "communication.ssp")
            apply_profiles(program, profiled.profiles.merge(overlap)).save(
                root / "profiled" / filename
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("phase", choices=("build", "profile", "communicate"))
    args = parser.parse_args()
    run(args.root, args.phase)


if __name__ == "__main__":
    main()
