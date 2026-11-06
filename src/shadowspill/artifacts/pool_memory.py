"""Portable, capacity-bound observations of one worker's pools and lanes.

These are retained-memory observations, not a guarantee about transient startup
peaks. The latter remain explicitly unknown. No framework or engine library is
loaded when reading this evidence or using it for CPU planning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Self, cast

from ._values import digest, freeze, nonnegative_int, plain
from .archive import Artifact


@dataclass(frozen=True)
class PoolMemoryProfile(Artifact):
    """Exact pool configuration and per-logical-rank resource observations.

    Use :func:`shadowspill.measure_pool_memory` to measure a rank, then
    :meth:`combine` for independent devices. Multiple workers sharing a GPU
    require joint calibration and are currently refused. Process increments
    are conservative additions to compute-profile reservations: their overlap
    with mature provider state is unknown, not proven disjoint.
    """

    artifact_kind: ClassVar[str] = "pool_memory_profile"
    pool: Mapping[str, Any]
    ranks: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool", freeze(self.pool))
        object.__setattr__(self, "ranks", tuple(freeze(r) for r in self.ranks))
        if not self.ranks or self.pool.get("version") != 1:
            raise ValueError("pool memory requires versioned configuration and ranks")
        for name in ("kind", "capacity"):
            nonnegative_int(self.pool[name], f"pool.{name}")
        if not self.pool["capacity"] or not self.pool.get("kind_name"):
            raise ValueError("pool memory requires positive capacity and kind name")
        seen_ranks: set[int] = set()
        seen_devices: set[str] = set()
        for rank in self.ranks:
            logical = rank["logical_rank"]
            nonnegative_int(logical, "logical_rank")
            identity = rank["identity"]
            for name in ("gpu_uuid", "gpu_name", "driver", "host", "libraries"):
                if not identity.get(name):
                    raise ValueError(f"pool memory identity requires {name}")
            if logical in seen_ranks or identity["gpu_uuid"] in seen_devices:
                raise ValueError(
                    "pool memory requires distinct ranks and physical GPUs"
                )
            seen_ranks.add(logical)
            seen_devices.add(identity["gpu_uuid"])
            nonnegative_int(rank["execution_control_bytes"], "execution control")
            if not rank["execution_control_bytes"]:
                raise ValueError("pool memory requires an independent device control")
            if rank.get("startup_peak_bytes") is not None:
                raise ValueError("this method does not establish startup peak bounds")
            if rank.get("method") != "full_capacity_pool_and_routes_v1":
                raise ValueError("unsupported pool memory observation method")
            cycles = rank["cycles"]
            if len(cycles) < 2:
                raise ValueError("pool memory requires at least two lifecycle cycles")
            for cycle in cycles:
                for phase in ("before", "live", "warmed", "released"):
                    samples = cycle[phase]
                    if len(samples) < 2:
                        raise ValueError("pool memory requires repeated phase samples")
                    for sample in samples:
                        p = sample["physical"]
                        for name in (
                            "process_bytes",
                            "device_used_bytes",
                            "device_total_bytes",
                        ):
                            nonnegative_int(p[name], name)
                        if (
                            not 0
                            <= p["process_bytes"]
                            <= p["device_used_bytes"]
                            <= p["device_total_bytes"]
                            or not p["device_total_bytes"]
                        ):
                            raise ValueError("inconsistent pool physical observation")
                        if sample.get("other_processes") != ():
                            raise ValueError(
                                "pool memory measurement requires an idle GPU"
                            )
                        if sample.get("own_process_present") is not True:
                            raise ValueError(
                                "own process memory observation is unresolved"
                            )
                        if sample.get("own_process_memory_available") is not True:
                            raise ValueError(
                                "own process memory counter is unavailable"
                            )
                # Retained global state must return to its pre-acquire range.
                # Otherwise subtraction cannot isolate this pool's resources.
                before = self._outside(cycle["before"])
                released = self._outside(cycle["released"])
                if max(before) != min(before) or released != before:
                    raise ValueError(
                        "device-global baseline changed during pool measurement"
                    )
                live = (*cycle["live"], *cycle["warmed"])
                if min(self._outside(live)) < before[0]:
                    raise ValueError("device-global pool delta is unresolved")
            totals = {
                s["physical"]["device_total_bytes"]
                for c in cycles
                for phase in ("before", "live", "warmed", "released")
                for s in c[phase]
            }
            if len(totals) != 1:
                raise ValueError("device capacity changed during pool measurement")

    @staticmethod
    def _outside(samples: Any) -> tuple[int, ...]:
        return tuple(
            s["physical"]["device_used_bytes"] - s["physical"]["process_bytes"]
            for s in samples
        )

    @classmethod
    def combine(cls, *profiles: Self) -> Self:
        """Combine independently measured ranks; no extra GPU acquisition."""
        if not profiles or any(p.pool != profiles[0].pool for p in profiles):
            raise ValueError(
                "combined pool measurements require identical configurations"
            )
        return cls(profiles[0].pool, tuple(r for p in profiles for r in p.ranks))

    def rank(self, logical_rank: int) -> Mapping[str, Any]:
        for rank in self.ranks:
            if rank["logical_rank"] == logical_rank:
                return rank
        raise ValueError(f"pool memory has no logical rank {logical_rank}")

    @property
    def resource_identity(self) -> str:
        return digest(
            {
                "pool": self.pool,
                "ranks": sorted(
                    (
                        {"logical_rank": r["logical_rank"], "identity": r["identity"]}
                        for r in self.ranks
                    ),
                    key=lambda r: r["logical_rank"],
                ),
            }
        )

    @property
    def physical_limit_bytes(self) -> int:
        """Free plus own-process bytes with the full spill pool already live.

        This includes the global registration charge. Do not subtract it again.
        Use the least available observed allowance across all ranks and cycles.
        """
        return cast(
            int,
            min(
                s["physical"]["device_total_bytes"]
                - s["physical"]["device_used_bytes"]
                + s["physical"]["process_bytes"]
                for r in self.ranks
                for c in r["cycles"]
                for phase in ("live", "warmed")
                for s in c[phase]
            ),
        )

    @property
    def process_reservation_bytes(self) -> int:
        """Conservative non-control process increment, charged once per worker."""
        return cast(
            int,
            max(
                0,
                *(
                    max(
                        s["physical"]["process_bytes"]
                        for phase in ("live", "warmed")
                        for s in c[phase]
                    )
                    - min(s["physical"]["process_bytes"] for s in c["before"])
                    - r["execution_control_bytes"]
                    for r in self.ranks
                    for c in r["cycles"]
                ),
            ),
        )

    def global_reservation_bytes(self, logical_rank: int) -> int:
        """Observed outside-process increment to reserve before pool startup."""
        return max(
            max(self._outside((*c["live"], *c["warmed"])))
            - min(self._outside(c["before"]))
            for c in self.rank(logical_rank)["cycles"]
        )

    def validate_program(
        self,
        spill_bytes: int,
        world_size: int,
        *,
        gpu: str | None = None,
        spill_kind: str | None = None,
    ) -> None:
        if spill_bytes != self.pool["capacity"]:
            raise ValueError("pool memory capacity differs from the spill budget")
        if {r["logical_rank"] for r in self.ranks} != set(range(world_size)):
            raise ValueError(
                "pool memory must cover exactly the Program's logical ranks"
            )
        if gpu is not None and any(
            r["identity"]["gpu_name"] != gpu for r in self.ranks
        ):
            raise ValueError("pool memory device differs from the Program target")
        if spill_kind is not None and self.pool["kind_name"] != spill_kind:
            raise ValueError("pool memory backend differs from communication timing")

    def to_dict(self) -> dict[str, Any]:
        return {"pool": plain(self.pool), "ranks": plain(self.ranks)}

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Self:
        return cls(value["pool"], tuple(value["ranks"]))
