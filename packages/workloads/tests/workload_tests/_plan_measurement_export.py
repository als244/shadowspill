"""Load real exported captures and edit costs using the existing CPU planner."""

import hashlib
import importlib.abc
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {
            "torch",
            "triton",
            "shadowspill_workloads",
            "mlops",
        } or fullname.startswith(("shadowspill.pytorch", "shadowspill.runtime")):
            raise AssertionError("framework/model import in CPU planner: " + fullname)


sys.meta_path.insert(0, Block())

from shadowspill import (  # noqa: E402
    CommunicationInfo,
    MemoryBudgets,
    Plan,
    ProfiledProgram,
    apply_profiles,
    plan_program,
    write_report,
)
from shadowspill.planner import SearchOptions  # noqa: E402

root = Path(sys.argv[1])
with zipfile.ZipFile(root / "measurements.zip") as archive:
    manifest = json.loads(archive.read("manifest.json"))
    for name, expected in manifest["files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        body = archive.read(name)
        assert hashlib.sha256(body).hexdigest() == expected["sha256"]
        assert len(body) == expected["size_bytes"]
        target = root / "original" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
rows = []
for key, capture in manifest["captures"].items():
    paths = capture["files"]
    measured = ProfiledProgram.load(root / "original" / paths["calibrated"])
    communication = CommunicationInfo.load(root / "original" / paths["communication"])
    original_digest = measured.digest
    controls = dict(
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=communication,
        search_options=SearchOptions(workers=1),
    )
    planned = plan_program(measured, **controls)
    assert not planned.simulation_only_reasons
    profiles = measured.profiles
    for name, profile in profiles.tasks.items():
        if profile.duration_seconds is not None:
            profiles.tasks[name] = replace(
                profile,
                duration_seconds=profile.duration_seconds * 2,
                metadata={**profile.metadata, "timing_kind": "hypothetical"},
            )
    changed = plan_program(apply_profiles(measured.program, profiles), **controls)
    assert measured.digest == original_digest
    assert changed.timeline.makespan_ns >= planned.timeline.makespan_ns
    assert not changed.simulation_only_reasons
    assert changed.profiled_program.fixed_device_bytes == measured.fixed_device_bytes
    changed.save(root / "plans" / f"{key}.ssp")
    assert Plan.load(root / "plans" / f"{key}.ssp").to_dict() == changed.to_dict()
    rows.append(
        {
            "capture": key,
            "measured_ns": planned.timeline.makespan_ns,
            "hypothetical_ns": changed.timeline.makespan_ns,
            "fixed_device_bytes": measured.fixed_device_bytes,
        }
    )
write_report(changed, root / "report", plots=False)
assert (root / "report").is_dir()
(root / "planning.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps({"planned_captures": len(rows), "framework_free": True}))
