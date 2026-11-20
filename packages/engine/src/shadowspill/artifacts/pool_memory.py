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
    """The measured device memory one spill pool configuration costs each worker.

    `measure_pool_memory` returns one for one logical rank, `combine` joins
    ranks measured on distinct devices, and ``PoolMemoryProfile.load`` reads a
    saved one. Passed to `plan_program` as ``pool_memory``, it caps the device
    budget at the memory observed available with the pool live
    (`physical_limit_bytes`) and adds the pool's process memory to the fixed
    overhead (`process_reservation_bytes`), conservatively, since it may
    overlap provider state. The observations are of retained memory; startup
    peaks stay unknown. Workers sharing a device are refused.

    Attributes
    ----------
    pool : mapping
        The measured pool configuration: ``kind``, ``kind_name``,
        ``capacity`` in bytes and the kind's ``configuration``.
    ranks : tuple of mapping
        One record per logical rank: its ``logical_rank``, the ``identity``
        of the device, host and libraries measured on, and the raw samples of
        each acquisition ``cycles``.
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
        """Join profiles measured for different logical ranks into one.

        Nothing is measured again.

        Parameters
        ----------
        *profiles : PoolMemoryProfile
            Profiles of the same pool configuration, for distinct logical
            ranks on distinct devices.

        Returns
        -------
        PoolMemoryProfile
            A profile with every rank.

        Raises
        ------
        ValueError
            If no profile is given, the configurations differ, or a rank or
            device repeats.
        """
        if not profiles or any(p.pool != profiles[0].pool for p in profiles):
            raise ValueError(
                "combined pool measurements require identical configurations"
            )
        return cls(profiles[0].pool, tuple(r for p in profiles for r in p.ranks))

    def rank(self, logical_rank: int) -> Mapping[str, Any]:
        """Return one logical rank's record.

        Parameters
        ----------
        logical_rank : int
            The logical rank.

        Returns
        -------
        mapping
            The rank's record in `ranks`.

        Raises
        ------
        ValueError
            If the profile has no such rank.
        """
        for rank in self.ranks:
            if rank["logical_rank"] == logical_rank:
                return rank
        raise ValueError(f"pool memory has no logical rank {logical_rank}")

    @property
    def resource_identity(self) -> str:
        """SHA-256 hex digest of the pool configuration and each rank's identity."""
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
        """Least device memory available to a worker with the pool live, in bytes.

        Free plus the process's own device memory, the least over every rank,
        cycle and sample. It already accounts for the pool's device-wide cost.
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
        """Largest growth of the process's device memory from the pool, in bytes.

        Measured from before acquisition to the live and warmed samples, less
        the 4 MiB execution control, over every rank and cycle. Each worker
        charges it once as fixed overhead.
        """
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
        """Return the largest growth of device memory outside the process, in bytes.

        A worker of this rank reserves it before creating its pools.

        Parameters
        ----------
        logical_rank : int
            The logical rank.

        Returns
        -------
        int
            The largest growth, over the rank's cycles, of device memory used
            outside the process while the pool is live.

        Raises
        ------
        ValueError
            If the profile has no such rank.
        """
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
        """Check that the profile fits a Program's planning inputs.

        Parameters
        ----------
        spill_bytes : int
            The spill budget, in bytes; the measured capacity must equal it.
        world_size : int
            The Program's logical ranks; the profile must cover exactly them.
        gpu : str, optional
            The Program's device name; every rank must have been measured on
            one.
        spill_kind : str, optional
            The spill pool kind communication was measured with.

        Raises
        ------
        ValueError
            If any of these differ.
        """
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

    def validate_worker(self, logical_rank: int, identity: Mapping[str, Any]) -> None:
        """Check that a worker's device and libraries match its rank's measurement.

        The device UUID, host name and driver version are provenance only, so
        a Plan runs on another equivalent device or host. Startup still caps
        the execution pool by the memory available then.

        Parameters
        ----------
        logical_rank : int
            The worker's logical rank.
        identity : mapping
            The worker's identity, with ``gpu_name`` and ``libraries``.

        Raises
        ------
        ValueError
            If the profile has no such rank, or the device name or library
            hashes differ.
        """
        measured = self.rank(logical_rank)["identity"]
        mismatches = [
            name
            for name in ("gpu_name", "libraries")
            if measured[name] != identity.get(name)
        ]
        if mismatches:
            raise ValueError(
                "pool memory is incompatible with this worker: " + ", ".join(mismatches)
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as plain data.

        Returns
        -------
        dict
            ``pool`` and ``ranks``.
        """
        return {"pool": plain(self.pool), "ranks": plain(self.ranks)}

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Self:
        """Rebuild a profile from `to_dict` data.

        Parameters
        ----------
        value : dict
            The profile, in `to_dict` form.
        payloads : mapping of str to bytes, optional
            Ignored; a profile holds no binary payloads.

        Returns
        -------
        PoolMemoryProfile
            The profile.

        Raises
        ------
        ValueError
            If the observations are incomplete or inconsistent.
        """
        return cls(value["pool"], tuple(value["ranks"]))
