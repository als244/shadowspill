"""Planning-cache identity and the request recorded with each answer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cached_property

from shadowspill.ir import ResidencySpec, ShadowSpillProgram
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.planner.result import ProgramPlanResult
from shadowspill.planner.search import SearchOptions
from shadowspill.simulator import SimulationConfig

from .records import SCHEMA


@dataclass(frozen=True)
class PlanRequest:
    program: ShadowSpillProgram
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    config: SimulationConfig
    admission: AdmissionFacts | None
    placement: AdmissionFacts | None
    options: SearchOptions

    def fields(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "program_digest": self.program.digest,
            "initial_residency": [item.to_dict() for item in self.initial_residency],
            "final_residency": [item.to_dict() for item in self.final_residency],
            "simulation": {
                "devices": [asdict(device) for device in self.config.devices],
                "spill_capacity_bytes": self.config.spill_capacity_bytes,
            },
            "admission_digest": self.admission.digest
            if self.admission is not None
            else None,
            "search_options": self.options.to_dict(),
        }

    @cached_property
    def key(self) -> str:
        payload = {
            **self.fields(),
            "placement_digest": self.placement.digest
            if self.placement is not None
            else None,
            # Placement-search semantics are part of the cache identity.
            **({"placement_search_revision": 2} if self.placement is not None else {}),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def boundary(self) -> dict[str, object]:
        return {
            **self.fields(),
            "key_digest": self.key,
            "search": self.options.resolved_algorithm.name,
        }

    def describe(self) -> str:
        device = self.config.devices[0]
        return (
            f"program {self.program.digest[:12]}, capacity {device.capacity_bytes} B,"
            f" fetch {device.fetch_solo_bandwidth_bytes_per_second}/"
            f"{device.fetch_concurrent_bandwidth_bytes_per_second} B/s solo/concurrent"
            f" latency {device.fetch_latency_ns} ns (metadata only),"
            f" evict {device.evict_solo_bandwidth_bytes_per_second}/"
            f"{device.evict_concurrent_bandwidth_bytes_per_second} B/s solo/concurrent"
            f" latency {device.evict_latency_ns} ns (metadata only),"
            f" spill {self.config.spill_capacity_bytes} B"
        )


def incumbent_field(incumbent: ProgramPlanResult | None) -> dict[str, object]:
    """The plan to beat as a record field: which resolution, which schedule.

    Provenance for the request and the stored plan; never part of a key.
    """

    if incumbent is None:
        return {"incumbent": None}
    return {
        "incumbent": {
            "selections": [item.to_dict() for item in incumbent.selections],
            "schedule_digest": incumbent.schedule.digest,
        }
    }
