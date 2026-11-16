"""Optional CPU planner process, separately installed from the NumPy client."""

import importlib.abc
import json
import sys
from dataclasses import replace
from pathlib import Path


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {
            "torch",
            "triton",
            "mlops",
            "shadowspill_workloads",
        }:
            raise AssertionError(
                "framework/catalog import in laptop planner: " + fullname
            )


sys.meta_path.insert(0, Block())

from shadowspill.training.preparation.local_planning import PlanningInputs  # noqa: E402
from shadowspill_client.protocol.training import ResolvedTraining  # noqa: E402

root = Path(sys.argv[1])
inputs = PlanningInputs.load(root / "measurements.zip")
config = ResolvedTraining.from_dict(json.loads((root / "config.json").read_text()))
profiles = {key: source.profiles for key, source in inputs.sources.items()}
for costs in profiles.values():
    for name, record in costs.tasks.items():
        if record.duration_seconds:
            costs.tasks[name] = replace(
                record,
                duration_seconds=record.duration_seconds * 2,
                resource_phases=None
                if record.resource_phases is None
                else tuple(
                    replace(phase, duration_ns=phase.duration_ns * 2)
                    for phase in record.resource_phases
                ),
                metadata={**record.metadata, "duration_kind": "estimated"},
            )
communication = {
    key: replace(
        value,
        fetch_bytes_per_second=value.fetch_bytes_per_second / 2,
        metadata={**value.metadata, "timing_kind": "estimated"},
    )
    for key, value in inputs.communication.items()
}
# Select explicit one-buffer alternatives; the service must retain a complete
# coverage witness and cannot recover a missing capability by replanning.
candidates = [
    key
    for key, members in inputs.candidates.items()
    if all(inputs.requests[name].use.count == 1 for name in members)
]
proposal = inputs.plan(
    config=config, profiles=profiles, communication=communication, candidates=candidates
)
reference = proposal.write(root / "local-plans.ssp")
(root / "proposal.json").write_text(
    json.dumps(
        {
            "reference": reference.to_dict(),
            "candidates": len(proposal.candidates),
            "plans": len(proposal.plans),
            "profiles": len(proposal.profiles),
            "framework_free": True,
        }
    )
)
