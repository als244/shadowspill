"""Verify equal planning requirements, then exchange CPU schedules only.

Device ordinals and compiled ABI digests remain rank-local. Everything affecting
storage, liveness or admission must match exactly. Timings use the slowest
profile and transfer calibration. No runtime tensor or entrypoint travels here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from shadowspill.ir import (
    MemorySchedule,
    ResidencySpec,
    ShadowSpillProgram,
    TaskAlternativeChoice,
    TaskOverlapCost,
    TaskProfile,
    TaskResourcePhase,
    TransferPhase,
    TransferSpec,
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

    def requirements(
        self, extra: Any, incumbent: ProgramPlanResult | None
    ) -> dict[str, str]:
        program = cast(dict[str, Any], self.program.to_dict())
        for device in program["devices"]:
            device.update(process_id="local", index=0)
        for profile in program["profiles"]:
            profile.pop("runtime_ns")
            profile.pop("compatibility_digest")
            for phase in profile["resource_phases"]:
                phase.pop("duration_ns")
            for cost in profile["overlap_costs"]:
                cost.pop("task_rate")
                cost.pop("transfer_rate")
        for alias in program["alias_groups"]:
            if transfer := alias.get("transfer"):
                for direction in ("fetch", "evict"):
                    for phase in transfer[direction]:
                        phase.pop("duration_ns")
        machine = cast(dict[str, Any], _simulation_config_to_dict(self.config))
        for device in machine["devices"]:
            for name in (
                "fetch_solo_bandwidth_bytes_per_second",
                "fetch_concurrent_bandwidth_bytes_per_second",
                "evict_solo_bandwidth_bytes_per_second",
                "evict_concurrent_bandwidth_bytes_per_second",
                "fetch_latency_ns",
                "evict_latency_ns",
            ):
                device.pop(name)
        values = {
            "program": program,
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
        signatures = {
            key: _digest(_device(value, "local")) for key, value in values.items()
        }
        # AdmissionFacts already hashes every allocation field as compact
        # columns. Reuse that exact identity instead of expanding millions of
        # steps into temporary dictionaries just to hash them again.
        signatures["allocation_contract"] = replace(
            self.facts, device_id="local"
        ).digest
        return signatures

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

    def receive(self, payload: dict[str, Any]) -> FixedLayoutSelection:
        """Rebind a verified candidate and certify its schedule on this rank."""
        value = _device(payload, self.facts.device_id)
        supplied = value["facts"]
        if "reference" in supplied:
            capacity = supplied.get("object_capacity_bytes")
            if (
                set(supplied) != {"reference", "object_capacity_bytes"}
                or type(capacity) is not int
            ):
                raise ValueError("invalid shared allocation contract reference")
            facts = replace(self.facts, object_capacity_bytes=capacity)
            if replace(facts, device_id="local").digest != supplied["reference"]:
                raise ValueError("shared plan changes the verified allocation contract")
        else:
            facts = AdmissionFacts.from_dict(supplied)
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
    local: Symmetry, *, extra: Any = None, incumbent: ProgramPlanResult | None = None
) -> tuple[Symmetry | None, dict[str, Any]]:
    bound = current()
    assert bound is not None
    control = bound.control
    enabled = bound.specification.symmetric_planning
    control.agree("symmetry/enabled", enabled)
    if not enabled:
        if bound.transfer_sharding:
            raise ValueError("transfer sharding requires a common symmetric schedule")
        return None, {"mode": "independent", "reason": "disabled"}
    signature = local.requirements(extra, incumbent)
    values = control.exchange("symmetry/requirements", signature)
    differing = [
        key
        for key in signature
        if any(value[key] != values[0][key] for value in values)
    ]
    allocation_policy = "identical"
    if differing == ["allocation_contract"]:
        from ._allocation_symmetry import reconcile

        merged = reconcile(local.facts, control)
        if merged is not None:
            local = replace(local, facts=merged)
            signature = local.requirements(extra, incumbent)
            differing = []
            allocation_policy = "max_anonymous_allocation_bytes_same_workspace_peak"
    if differing or len(local.config.devices) != 1:
        reason = ", ".join(differing) or "multiple local devices"
        if bound.transfer_sharding:
            raise ValueError(
                f"transfer sharding cannot use independent schedules: {reason}"
            )
        print(f"ShadowSpill: symmetric planning fallback ({reason}).", flush=True)
        return None, {"mode": "independent", "reason": reason, "requirements": values}
    timings = control.exchange(
        "symmetry/timings",
        {
            "tasks": [
                {
                    "runtime_ns": profile.runtime_ns,
                    "resource_phases": [
                        item.to_dict() for item in profile.resource_phases
                    ],
                    "overlap_costs": [item.to_dict() for item in profile.overlap_costs],
                }
                for profile in local.program.profiles
            ],
            "transfers": [
                None
                if alias.transfer is None
                else {
                    direction: [
                        phase.duration_ns
                        for phase in getattr(alias.transfer, direction)
                    ]
                    for direction in ("fetch", "evict")
                }
                for alias in local.program.alias_groups
            ],
            "device": asdict(local.config.devices[0]),
        },
    )

    def transfer_cost(index: int, transfer: TransferSpec) -> TransferSpec:
        def phases(
            direction: str, original: tuple[TransferPhase, ...]
        ) -> tuple[TransferPhase, ...]:
            return tuple(
                replace(
                    phase,
                    duration_ns=max(
                        value["transfers"][index][direction][phase_index]
                        for value in timings
                    ),
                )
                for phase_index, phase in enumerate(original)
            )

        return replace(
            transfer,
            fetch=phases("fetch", transfer.fetch),
            evict=phases("evict", transfer.evict),
        )

    def task_cost(index: int, profile: TaskProfile) -> TaskProfile:
        rows = [value["tasks"][index] for value in timings]
        # Keep a coherent phase decomposition from the slowest complete task.
        # Taking the max of every segment could invent a duration no rank saw.
        slowest = max(rows, key=lambda row: row["runtime_ns"])
        costs = tuple(
            TaskOverlapCost(
                item.timing_class,
                item.bytes,
                min(row["overlap_costs"][j]["task_rate"] for row in rows),
                min(row["overlap_costs"][j]["transfer_rate"] for row in rows),
            )
            for j, item in enumerate(profile.overlap_costs)
        )
        return replace(
            profile,
            runtime_ns=slowest["runtime_ns"],
            resource_phases=tuple(
                TaskResourcePhase.from_value(item, "symmetry.resource_phases")
                for item in slowest["resource_phases"]
            ),
            overlap_costs=costs,
        )

    program = replace(
        local.program,
        profiles=tuple(
            task_cost(index, profile)
            for index, profile in enumerate(local.program.profiles)
        ),
        alias_groups=tuple(
            alias
            if alias.transfer is None
            else replace(
                alias,
                transfer=transfer_cost(index, alias.transfer),
            )
            for index, alias in enumerate(local.program.alias_groups)
        ),
    )
    cost = {}
    for key in (
        "fetch_solo_bandwidth_bytes_per_second",
        "fetch_concurrent_bandwidth_bytes_per_second",
        "evict_solo_bandwidth_bytes_per_second",
        "evict_concurrent_bandwidth_bytes_per_second",
        "fetch_latency_ns",
        "evict_latency_ns",
    ):
        reduce = min if "bandwidth" in key else max
        cost[key] = reduce(value["device"][key] for value in timings)
    config = replace(local.config, devices=(replace(local.config.devices[0], **cost),))
    evidence = {
        "mode": "symmetric",
        "requirements": signature,
        "timing_policy": (
            "slowest_task_coherent_phases_min_progress_"
            "max_transfer_time_min_bandwidth_max_latency"
        ),
        "allocation_policy": allocation_policy,
        "local_programs": control.exchange("symmetry/programs", local.program.digest),
    }
    return replace(local, program=program, config=config), evidence
