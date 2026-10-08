"""Verify equal planning requirements, then exchange CPU schedules only.

Device ordinals and compiled ABI digests remain rank-local. Everything affecting
storage, liveness or admission must match exactly. Timings use the slowest
profile and transfer calibration. No runtime tensor or entrypoint travels here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from shadowspill.ir import (
    MemorySchedule,
    ResidencySpec,
    ShadowSpillProgram,
    TaskAlternativeChoice,
)
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.planner.admission.refinement import (
    FixedLayoutSelection,
    placement_facts,
    resolve_fixed_layout_selection,
)
from shadowspill.planner.diagnostics import PlanningDiagnostics
from shadowspill.planner.plan_store import PlanLookup
from shadowspill.planner.result import ProgramPlanResult, ResolutionPlan
from shadowspill.planner.search import SearchOptions
from shadowspill.planner.serialization import (
    _resident_slice_from_value,
    _simulation_config_from_value,
    _simulation_config_to_dict,
    _simulation_result_from_value,
)
from shadowspill.simulator import SimulationConfig

from . import current


def _device(value: Any, identity: str) -> Any:
    """Translate explicit device fields, never arbitrary strings or substrings."""
    if isinstance(value, dict):
        return {
            key: identity if key == "device_id" else _device(item, identity)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_device(item, identity) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Symmetry:
    program: ShadowSpillProgram
    config: SimulationConfig
    facts: AdmissionFacts
    initial: tuple[ResidencySpec, ...]
    final: tuple[ResidencySpec, ...]
    scratch: int

    def requirements(self, extra: Any, incumbent: ProgramPlanResult | None) -> dict:
        program = self.program.to_dict()
        for device in program["devices"]:
            device.update(process_id="local", index=0)
        for profile in program["profiles"]:
            profile.pop("runtime_ns")
            profile.pop("compatibility_digest")
        machine = _simulation_config_to_dict(self.config)
        for device in machine["devices"]:
            for name in (
                "fetch_bandwidth_bytes_per_second",
                "evict_bandwidth_bytes_per_second",
                "fetch_latency_ns",
                "evict_latency_ns",
            ):
                device.pop(name)
        values = {
            "program": program,
            "allocation_contract": self.facts.to_dict(),
            "machine_capacity": machine,
            "residency": [
                [item.to_dict() for item in values]
                for values in (self.initial, self.final)
            ],
            "scratch_reserve": self.scratch,
            "request": extra,
            "incumbent": None
            if incumbent is None
            else {
                "schedule": incumbent.schedule.to_dict(),
                "choices": [item.to_dict() for item in incumbent.selections],
                "resident_slice": incumbent.resident_slice.to_dict(),
                "config": _simulation_config_to_dict(incumbent.simulation_config),
            },
        }
        return {key: _digest(_device(value, "local")) for key, value in values.items()}

    def subset(self, program: ShadowSpillProgram) -> Symmetry:
        tasks = {task.task_id for task in program.tasks}
        return replace(
            self,
            program=program,
            facts=replace(
                self.facts,
                tasks=tuple(task for task in self.facts.tasks if task.task_id in tasks),
            ),
        )

    def receive(self, payload: dict) -> FixedLayoutSelection:
        """Rebind a verified candidate and certify its schedule on this rank."""
        value = _device(payload, self.facts.device_id)
        facts = AdmissionFacts.from_dict(value["facts"])
        if (
            replace(facts, object_capacity_bytes=self.facts.object_capacity_bytes)
            != self.facts
        ):
            raise ValueError("shared plan changes the verified allocation contract")
        config = _simulation_config_from_value(value["config"], "shared.config")
        if config != self.config or value["scratch"] != self.scratch:
            raise ValueError("shared plan changes the verified machine contract")
        choices = tuple(
            TaskAlternativeChoice.from_value(item, "shared.choice")
            for item in value["choices"]
        )
        schedule = MemorySchedule.from_dict(value["schedule"])
        schedule.validate(self.program, choices)
        result = ProgramPlanResult(
            program=self.program,
            search_options=SearchOptions.from_dict(value["options"], "shared.options"),
            initial_residency=self.initial,
            final_residency=self.final,
            simulation_config=config,
            schedule=schedule,
            selections=choices,
            simulation=_simulation_result_from_value(
                value["simulation"], "shared.simulation"
            ),
            diagnostics=PlanningDiagnostics.from_value(
                value["diagnostics"], "shared.diagnostics"
            ),
            resident_slice=_resident_slice_from_value(
                value["resident_slice"], "shared.resident_slice"
            ),
            admission_facts=facts,
            placement_facts=placement_facts(facts, scratch_reserve_bytes=self.scratch),
            resolutions=tuple(
                ResolutionPlan(
                    row["selection_id"],
                    tuple(
                        TaskAlternativeChoice.from_value(
                            item, "shared.resolution.choice"
                        )
                        for item in row["choices"]
                    ),
                    row["candidate_id"],
                    MemorySchedule.from_dict(row["schedule"]),
                    _simulation_result_from_value(
                        row["simulation"], "shared.resolution.simulation"
                    ),
                    _resident_slice_from_value(
                        row["resident_slice"], "shared.resolution.resident_slice"
                    ),
                )
                for row in value["resolutions"]
            ),
        )
        # The normal admission path owns effective-capacity accounting, including
        # objects already shared with the runtime. Never import a foreign layout.
        selected = resolve_fixed_layout_selection(
            self.config,
            self.facts,
            lambda _: PlanLookup(result, False),
            scratch_reserve_bytes=self.scratch,
        )
        if selected.facts != facts:
            raise ValueError("shared plan changes the admitted allocation contract")
        for row in result.resolutions:
            row.schedule.validate(self.program, row.selections)
        if selected.admission.simulation.makespan_ns != result.simulation.makespan_ns:
            raise ValueError(
                "shared schedule replay differs from its verified prediction"
            )
        return selected


def verify(
    local: Symmetry, *, extra=None, incumbent=None
) -> tuple[Symmetry | None, dict]:
    bound = current()
    assert bound is not None
    control = bound.control
    enabled = bound.specification.symmetric_planning
    control.agree("symmetry/enabled", enabled)
    if not enabled:
        return None, {"mode": "independent", "reason": "disabled"}
    signature = local.requirements(extra, incumbent)
    values = control.exchange("symmetry/requirements", signature)
    differing = [
        key
        for key in signature
        if any(value[key] != values[0][key] for value in values)
    ]
    if differing or len(local.config.devices) != 1:
        reason = ", ".join(differing) or "multiple local devices"
        print(f"ShadowSpill: symmetric planning fallback ({reason}).", flush=True)
        return None, {"mode": "independent", "reason": reason, "requirements": values}
    timings = control.exchange(
        "symmetry/timings",
        {
            "tasks": [profile.runtime_ns for profile in local.program.profiles],
            "device": asdict(local.config.devices[0]),
        },
    )
    program = replace(
        local.program,
        profiles=tuple(
            replace(profile, runtime_ns=max(value["tasks"][index] for value in timings))
            for index, profile in enumerate(local.program.profiles)
        ),
    )
    cost = {}
    for key in (
        "fetch_bandwidth_bytes_per_second",
        "evict_bandwidth_bytes_per_second",
        "fetch_latency_ns",
        "evict_latency_ns",
    ):
        reduce = min if "bandwidth" in key else max
        cost[key] = reduce(value["device"][key] for value in timings)
    config = replace(local.config, devices=(replace(local.config.devices[0], **cost),))
    evidence = {
        "mode": "symmetric",
        "requirements": signature,
        "timing_policy": "max_task_time_min_bandwidth_max_latency",
        "local_programs": control.exchange("symmetry/programs", local.program.digest),
    }
    return replace(local, program=program, config=config), evidence
