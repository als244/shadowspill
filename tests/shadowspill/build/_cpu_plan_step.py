"""Plan a saved distributed training build without importing GPU code."""

import importlib.abc
import sys
from pathlib import Path


class CPUOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split(".")[0] in {"torch", "mlops"} or name.startswith(
            ("shadowspill.runtime", "shadowspill.build", "shadowspill.execution")
        ):
            raise AssertionError("CPU planning imported " + name)


sys.meta_path.insert(0, CPUOnly())

from shadowspill import (  # noqa: E402
    CommunicationInfo,
    MemoryBudgets,
    Plan,
    ProfiledProgram,
    plan_program,
    write_report,
)

profiled = ProfiledProgram.load(sys.argv[1])
world = profiled.program.parallelism.world_size
plan = plan_program(
    profiled,
    memory_budgets=MemoryBudgets(profiled.fixed_device_bytes + (8 << 20), 8 << 20),
    communication_info=CommunicationInfo(
        20e9,
        20e9,
        topology={"node": tuple(range(world))},
        scale_up_bytes_per_second=100e9,
        scale_up_latency_seconds=2e-6,
    ),
)
destination = Path(sys.argv[2])
plan.save(destination / "plan.ssp")
loaded = Plan.load(destination / "plan.ssp")
assert loaded.simulation_only_reasons
assert loaded._accumulation_plan is None
assert loaded.summary["logical_ranks"] == world
assert any(task.phase == "weight_communication" for task in loaded.program.tasks)
write_report(loaded, destination / "report")
print("CPU-only full-step planning and archive roundtrip passed")
