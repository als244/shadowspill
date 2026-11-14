"""Physical planning must consume saved measurement entirely on the CPU."""

import importlib.abc
import sys
from dataclasses import replace
from pathlib import Path


class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == name or fullname.startswith(name + ".")
            for name in (
                "torch",
                "triton",
                "mlops",
                "shadowspill.runtime",
                "shadowspill.pytorch",
                "shadowspill.build",
                "shadowspill.store",
            )
        ):
            raise AssertionError(f"CPU placement imported {fullname}")


sys.meta_path.insert(0, BlockGPU())

from shadowspill import (  # noqa: E402
    CommunicationInfo,
    MemoryBudgets,
    Plan,
    Profiles,
    Program,
    apply_profiles,
    plan_program,
    write_report,
)

directory = Path(sys.argv[1])
profiled = apply_profiles(
    Program.load(directory / "program.ssp"), Profiles.load(directory / "profiles.ssp")
)
plan = plan_program(
    profiled,
    memory_budgets=MemoryBudgets(profiled.fixed_device_bytes + (1 << 20), 1 << 20),
    communication_info=CommunicationInfo(
        fetch_bytes_per_second=1e10,
        evict_bytes_per_second=1e10,
    ),
)
assert plan.memory_layout is not None
assert not plan.simulation_only_reasons
plan.save(directory / "plan.ssp")
restored = Plan.load(directory / "plan.ssp")
assert restored.memory_layout.digest == plan.memory_layout.digest
assert restored._admission_facts.digest == plan._admission_facts.digest
write_report(restored, directory / "report")


def rejected(call, message):
    try:
        call()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"accepted invalid evidence: {message}")


key = next(key for key in profiled.task_profiles if key != "invocation_start")
record = profiled.task_profiles[key]
estimated = profiled.profiles
estimated.tasks[key] = replace(
    record, metadata={**record.metadata, "memory_kind": "estimated"}
)
simulation = plan_program(
    apply_profiles(profiled.program, estimated),
    memory_budgets=plan.memory_budgets,
    communication_info=plan.communication_info,
)
assert simulation.memory_layout is None
assert f"estimated memory: {key}" in simulation.simulation_only_reasons

corrupted = profiled.profiles
corrupted.tasks[key] = replace(
    record, metadata={**record.metadata, "compiled_layout": {}}
)
rejected(
    lambda: plan_program(
        apply_profiles(profiled.program, corrupted),
        memory_budgets=plan.memory_budgets,
        communication_info=plan.communication_info,
    ),
    "differs from its allocation evidence",
)
rejected(
    lambda: replace(
        plan, memory_layout=replace(plan.memory_layout, program_digest="wrong")
    ),
    "different plan inputs",
)
rejected(
    lambda: replace(plan, memory_budgets=MemoryBudgets(1, 1 << 20)),
    "exceeds the total GPU allowance",
)
