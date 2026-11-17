"""Export already measured Programs on CPU; never construct a model or pool."""

import math
from typing import Any

from shadowspill import CommunicationInfo, MemoryBudgets, ProfiledProgram
from shadowspill.communication.memory import communication_overhead
from shadowspill.errors import PlanInfeasibleError
from shadowspill.step import StepDataOrdering

from .inputs import problem
from .io import write


def dump(args):
    sources = tuple(path.resolve() for path in args.inputs)
    if len({path.stem for path in sources}) != len(sources):
        raise ValueError("ProfiledProgram archives must have distinct filename stems")
    for name, value in (("physical", args.physical_gib), ("spill", args.spill_gib)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} budget must be positive and finite")
    programs = {path.stem: ProfiledProgram.load(path) for path in sources}
    communication = CommunicationInfo.load(args.communication)
    physical = int(args.physical_gib * (1 << 30))
    fixed = max(
        item.fixed_device_bytes + communication_overhead(item.program, communication)[0]
        for item in programs.values()
    )
    if physical <= fixed:
        raise ValueError("budget leaves no execution capacity after recorded overhead")
    budgets = MemoryBudgets(physical, int(args.spill_gib * (1 << 30)))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    failures = []
    for geometry, profiled in programs.items():
        templates = profiled.program.bindings.get("training_templates")
        count = None if templates is None else len(templates)
        for ordering in _orderings(count, args.orderings):
            record = {
                "geometry": geometry,
                "microbatches": count,
                "ordering": "default" if ordering is None else ordering.label,
                "profiled_program_digest": profiled.digest,
            }
            try:
                inputs, semantics = problem(
                    profiled, ordering, budgets, communication, fixed
                )
            except PlanInfeasibleError as error:
                failures.append({**record, "error": str(error), "kind": error.kind})
                continue
            name = f"program_{len(records):04d}"
            write(output / f"{name}.problem.json", inputs.to_dict())
            write(output / f"{name}.semantics.json", semantics)
            records.append(
                {
                    **record,
                    "name": name,
                    "program_digest": inputs.program.digest,
                    "problem": f"{name}.problem.json",
                    "semantics": f"{name}.semantics.json",
                }
            )
            print("DUMP", geometry, record["ordering"], flush=True)
    write(
        output / "dump.json",
        {
            "schema": "pressurefit_collection/v1",
            "sources": list(map(str, sources)),
            "communication": str(args.communication.resolve()),
            "programs": records,
            "infeasible_geometries": failures,
            "physical_budget_bytes": physical,
            "execution_budget_bytes": physical - fixed,
            "spill_budget_bytes": budgets.spill_bytes,
            "fixed_device_bytes_by_candidate": {
                name: item.fixed_device_bytes
                + communication_overhead(item.program, communication)[0]
                for name, item in programs.items()
            },
            "workload": args.label or ", ".join(programs),
        },
    )
    if not records:
        raise ValueError("no candidate has a supported ordering")


def _orderings(count, mode):
    if count is None:
        return [None]
    if mode == "depth-first":
        return [StepDataOrdering.depth_first(count)]
    return [
        StepDataOrdering(depth=depth, breadth=count // depth)
        for depth in range(1, count + 1)
        if count % depth == 0
    ]
