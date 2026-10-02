"""Replay the failed 128K evaluation's archived planner inputs without a GPU."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from shadowspill.ir import MemoryLocation, ResidencySpec, ShadowSpillProgram
from shadowspill.planner import SearchOptions
from shadowspill.pytorch.planning.admission.bindings import (
    TaskOutputBinding,
    build_admission_facts,
)
from shadowspill.simulator import DeviceSimulationConfig, SimulationConfig
from shadowspill.task.profiles import TaskMeasurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_dir", type=Path)
    args = parser.parse_args()
    base = args.rank_dir / "artifacts/v1"
    requests = []
    for path in (base / "planning/requests").rglob("request.json"):
        request = json.loads(path.read_text())
        digest = request["program_digest"]
        program = ShadowSpillProgram.from_json(
            (base / f"planning/programs/{digest[:2]}/{digest}/program.json").read_text()
        )
        if len(program.tasks) == 2 and program.tasks[-1].phase == "forward":
            requests.append((path.stat().st_mtime, request, program))
    _, request, program = max(requests, key=lambda item: item[0])
    task = program.tasks[-1]
    digest = next(
        p.compatibility_digest
        for p in program.profiles
        if p.profile_id == task.profile_id
    )
    measurement = TaskMeasurement.from_dict(
        json.loads(
            (
                base
                / f"build/profiling/measurements/{digest[:2]}/{digest}/measurement.json"
            ).read_text()
        )["measurement"]
    )
    objects = {obj.object_id: obj for obj in program.objects}
    bindings = tuple(
        TaskOutputBinding(i, objects[obj].alias_group_id)
        for i, obj in enumerate(task.outputs)
    )
    capacity = request["simulation"]["devices"][0]["capacity_bytes"]
    facts = build_admission_facts(
        program,
        execution_pool_bytes=capacity,
        object_capacity_bytes=capacity,
        output_bindings={task.task_id: bindings},
        allocation_traces_by_compatibility={digest: measurement.allocation_trace},
    )
    options = SearchOptions.from_dict(request["search_options"], "search_options")
    config = SimulationConfig(
        devices=tuple(
            DeviceSimulationConfig(**v) for v in request["simulation"]["devices"]
        ),
        spill_capacity_bytes=request["simulation"]["spill_capacity_bytes"],
    )
    kwargs = dict(
        initial_residency=tuple(
            ResidencySpec(v["alias_group_id"], MemoryLocation(v["location"]))
            for v in request["initial_residency"]
        ),
        final_residency=tuple(
            ResidencySpec(v["alias_group_id"], MemoryLocation(v["location"]))
            for v in request["final_residency"]
        ),
        config=config,
        generic=options.generic,
        workers=1,
    )
    print(
        "Capacity GiB",
        capacity / 2**30,
        "workspace GiB",
        measurement.workspace_charged_bytes / 2**30,
        flush=True,
    )
    results = []
    for label, placement in (
        ("physical", facts),
        ("logical", None),
        ("physical-80GiB", replace(facts, pool_capacity_bytes=80 << 30)),
    ):
        try:
            result = options.resolved_algorithm(program, placement=placement, **kwargs)
            row = dict(
                label=label, passed=True, seconds=result.simulation.makespan_ns / 1e9
            )
        except Exception as error:
            row = dict(
                label=label,
                passed=False,
                error=str(error),
                diagnostics=[asdict(d) for d in getattr(error, "diagnostics", ())],
            )
        results.append(row)
        print(json.dumps(row), flush=True)
    (args.rank_dir / "evaluation-admission-replay.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    (args.rank_dir / "evaluation-admission-facts.json").write_text(
        facts.to_json() + "\n"
    )


if __name__ == "__main__":
    main()
