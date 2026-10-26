"""Export already measured Programs on CPU; never construct a model or pool."""

import math
from typing import Any

from benchmarking.quickstart.phases.common import read
from benchmarking.quickstart.phases.search import orderings
from shadowspill import CommunicationInfo, MemoryBudgets, ProfiledProgram
from shadowspill.communication.memory import communication_overhead
from shadowspill.errors import PlanInfeasibleError

from .inputs import problem
from .io import write


def dump(args):
    source = args.input.resolve()
    request, index = read(source / "request.json"), read(source / "profiled/index.json")
    programs = {
        item["name"]: ProfiledProgram.load(source / "profiled" / item["file"])
        for item in index["candidates"]
        if "failure" not in item
    }
    if not programs:
        raise ValueError("no successfully profiled candidates")
    communication = CommunicationInfo.load(
        request.get("communication_info") or source / "communication.ssp"
    )
    physical = max(request["search_budgets"])
    if args.physical_gib is not None:
        if not math.isfinite(args.physical_gib) or args.physical_gib <= 0:
            raise ValueError("physical budget must be positive and finite")
        physical = int(args.physical_gib * (1 << 30))
    fixed = max(
        item.fixed_device_bytes + communication_overhead(item.program, communication)[0]
        for item in programs.values()
    )
    if physical <= fixed:
        raise ValueError("budget leaves no execution capacity after recorded overhead")
    budgets = MemoryBudgets(physical, request["spill_bytes"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    failures = []
    for entry in index["candidates"]:
        if "failure" in entry:
            failures.append(entry)
            continue
        for ordering in orderings(
            entry.get("num_examples"), request.get("orderings", "factors")
        ):
            record = {
                "geometry": entry["name"],
                "microbatches": entry.get("num_examples"),
                "ordering": "default" if ordering is None else ordering.label,
                "profiled_program_digest": programs[entry["name"]].digest,
            }
            try:
                inputs, semantics = problem(
                    programs[entry["name"]], ordering, budgets, communication, fixed
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
            print("DUMP", entry["name"], record["ordering"], flush=True)
    write(
        output / "dump.json",
        {
            "schema": "pressurefit_collection/v1",
            "source": str(source),
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
            "workload": request.get("label", request["factory"]),
            "metadata": index.get("metadata", {}),
        },
    )
    if not records:
        raise ValueError("no candidate has a supported ordering")
