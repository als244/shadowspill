"""Reusable inputs for framework-independent planning."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from shadowspill.artifacts.budgets import MemoryBudgets as MemoryBudgets
from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import ResidencySpec, ShadowSpillProgram, shared_residency_footprint
from shadowspill.planning.memory_search.admission import AdmissionFacts
from shadowspill.planning.serialization import (
    _canonical_json,
    _digest,
    _integer,
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _simulation_config_from_value,
    _simulation_config_to_dict,
    _string,
)
from shadowspill.schema import artifact_schema
from shadowspill.simulator import SimulationConfig

_PLANNING_PROBLEM_SCHEMA = artifact_schema("plan_program")


@dataclass(frozen=True, slots=True)
class TransferBandwidths:
    """The copy rates a plan is priced at, replacing a problem's own.

    `memory_search.plan_program` takes them as ``transfer_bandwidths``.
    Every field is part of the record and its digest; planning prices copies
    at the four rates alone.

    Parameters
    ----------
    fetch_solo_bytes_per_second : int
        Fetch rate while no eviction moves bytes, in bytes per second.
        Positive.
    fetch_concurrent_bytes_per_second : int
        Fetch rate while an eviction moves bytes, in bytes per second.
        Positive.
    evict_solo_bytes_per_second : int
        Eviction rate while no fetch moves bytes, in bytes per second.
        Positive.
    evict_concurrent_bytes_per_second : int
        Eviction rate while a fetch moves bytes, in bytes per second.
        Positive.
    scale_numerator : int, default 1
        Numerator of a scale recorded with the calibration. Positive.
        Planning does not apply it.
    scale_denominator : int, default 1
        Denominator of that scale. Positive.
    calibration_digest : str, optional
        SHA-256 hex digest of the calibration the rates came from, recorded
        as provenance.
    provenance : str, optional
        A free-form note on where the rates came from.
    fetch_latency_ns : int, optional
        Fetch latency, in nanoseconds, at least zero, kept as metadata. None
        keeps the problem's own.
    evict_latency_ns : int, optional
        Eviction latency, in nanoseconds, at least zero, kept as metadata.
        None keeps the problem's own.

    Raises
    ------
    ValueError
        If a rate or scale is not positive, a latency is negative, or
        ``calibration_digest`` is not 64 characters long.
    """

    fetch_solo_bytes_per_second: int
    fetch_concurrent_bytes_per_second: int
    evict_solo_bytes_per_second: int
    evict_concurrent_bytes_per_second: int
    scale_numerator: int = 1
    scale_denominator: int = 1
    calibration_digest: str | None = None
    provenance: str | None = None
    fetch_latency_ns: int | None = None
    evict_latency_ns: int | None = None

    def __post_init__(self) -> None:
        if any(
            rate <= 0
            for rate in (
                self.fetch_solo_bytes_per_second,
                self.fetch_concurrent_bytes_per_second,
                self.evict_solo_bytes_per_second,
                self.evict_concurrent_bytes_per_second,
            )
        ):
            raise ValueError("transfer bandwidths must be positive")
        for latency in (self.fetch_latency_ns, self.evict_latency_ns):
            if latency is not None and latency < 0:
                raise ValueError("transfer latencies must be non-negative")
        if self.scale_numerator <= 0 or self.scale_denominator <= 0:
            raise ValueError("transfer bandwidth scale must be positive")
        if self.calibration_digest is not None and len(self.calibration_digest) != 64:
            raise ValueError("calibration_digest must be SHA-256 or None")

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of `TransferBandwidths.to_dict`."""
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Write the rates as a JSON-compatible mapping.

        Returns
        -------
        dict
            Every field by name.
        """
        return {
            "calibration_digest": self.calibration_digest,
            "evict_solo_bytes_per_second": self.evict_solo_bytes_per_second,
            "evict_concurrent_bytes_per_second": (
                self.evict_concurrent_bytes_per_second
            ),
            "evict_latency_ns": self.evict_latency_ns,
            "fetch_solo_bytes_per_second": self.fetch_solo_bytes_per_second,
            "fetch_concurrent_bytes_per_second": (
                self.fetch_concurrent_bytes_per_second
            ),
            "fetch_latency_ns": self.fetch_latency_ns,
            "provenance": self.provenance,
            "scale_denominator": self.scale_denominator,
            "scale_numerator": self.scale_numerator,
        }

    @classmethod
    def from_value(
        cls, value: object, path: str = "transfer_bandwidths"
    ) -> TransferBandwidths:
        """Read rates back from what `TransferBandwidths.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str, default "transfer_bandwidths"
            Where the value sits in its record, for error messages.

        Returns
        -------
        TransferBandwidths
            The rates.

        Raises
        ------
        ValueError
            If a field is missing, malformed or out of range.
        """
        data = _mapping(value, path)
        return cls(
            fetch_solo_bytes_per_second=_integer(
                data.get("fetch_solo_bytes_per_second"),
                f"{path}.fetch_solo_bytes_per_second",
            ),
            fetch_concurrent_bytes_per_second=(
                _integer(
                    data.get("fetch_concurrent_bytes_per_second"),
                    f"{path}.fetch_concurrent_bytes_per_second",
                )
            ),
            evict_solo_bytes_per_second=_integer(
                data.get("evict_solo_bytes_per_second"),
                f"{path}.evict_solo_bytes_per_second",
            ),
            evict_concurrent_bytes_per_second=(
                _integer(
                    data.get("evict_concurrent_bytes_per_second"),
                    f"{path}.evict_concurrent_bytes_per_second",
                )
            ),
            scale_numerator=_integer(
                data.get("scale_numerator"), f"{path}.scale_numerator"
            ),
            scale_denominator=_integer(
                data.get("scale_denominator"), f"{path}.scale_denominator"
            ),
            calibration_digest=_optional_string(
                data.get("calibration_digest"), f"{path}.calibration_digest"
            ),
            provenance=_optional_string(data.get("provenance"), f"{path}.provenance"),
            fetch_latency_ns=_optional_integer(
                data.get("fetch_latency_ns"), f"{path}.fetch_latency_ns"
            ),
            evict_latency_ns=_optional_integer(
                data.get("evict_latency_ns"), f"{path}.evict_latency_ns"
            ),
        )


@dataclass(frozen=True, slots=True)
class ShadowSpillPlanningProblem:
    """A program, the boundaries it runs between, and the machine it was built for.

    The question `memory_search.plan_program` answers, with no search
    policy in it. Its capacities must reconcile: the execution pool is the
    source budget less ``fixed_execution_bytes`` and the device's shared
    residency, and the object capacity is that pool less
    ``object_reserve_bytes``.

    Parameters
    ----------
    role : str
        ``"step"`` or ``"forward"``: whether the program is a whole step or
        a forward pass. Recorded with the problem; planning does not read
        it.
    program : ShadowSpillProgram
        The program, its task alternatives open.
    initial_residency : tuple of ResidencySpec
        Where each alias group is before the first task.
    final_residency : tuple of ResidencySpec
        Where each alias group must be after the last task.
    simulation_config : SimulationConfig
        The machine at the source budget: one device, whose capacity is the
        object capacity plus its shared residency, and the spill capacity.
    admission_facts : AdmissionFacts
        The physical facts at the source budget, for the same device; their
        pool and object capacities are the ones above.
    source_execution_budget_bytes : int
        The device budget the problem was built under, in bytes. Positive.
    maximum_execution_budget_bytes : int
        The largest device budget it may be planned under, in bytes: the
        pool its tasks were profiled in. At least the source budget.
    maximum_spill_budget_bytes : int
        The spill pool it was profiled against, in bytes, kept as a record;
        a larger spill budget may still be planned. At least the spill
        capacity.
    fixed_execution_bytes : int
        Device bytes outside the execution pool, such as the runtime's fixed
        costs, deducted from every budget. At least zero.
    object_reserve_bytes : int
        Pool bytes held back from object residency. At least zero.
    dynamic_scratch_reserve_bytes : int
        Pool bytes reserved for scratch outside the fixed layout: a layout
        is measured against the pool less this, and the certificate adds it
        to what a layout requires. At least zero.

    Raises
    ------
    ValueError
        If the role is unsupported, the configuration has other than one device
        or another device than the facts, a capacity is out of range or does
        not reconcile, or the facts do not describe the program.
    """

    role: str
    program: ShadowSpillProgram
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    simulation_config: SimulationConfig
    admission_facts: AdmissionFacts
    source_execution_budget_bytes: int
    maximum_execution_budget_bytes: int
    maximum_spill_budget_bytes: int
    fixed_execution_bytes: int
    object_reserve_bytes: int
    dynamic_scratch_reserve_bytes: int

    def __post_init__(self) -> None:
        if self.role not in {"step", "forward"}:
            raise ValueError(f"unsupported problem role {self.role!r}")
        if len(self.simulation_config.devices) != 1:
            raise ValueError("ShadowSpillPlanningProblem requires one device")
        device = self.simulation_config.devices[0]
        if device.device_id != self.admission_facts.device_id:
            raise ValueError("simulation and admission devices differ")
        if self.source_execution_budget_bytes <= 0:
            raise ValueError("source execution budget must be positive")
        if self.maximum_execution_budget_bytes < self.source_execution_budget_bytes:
            raise ValueError("maximum execution budget is below the source budget")
        source_spill = self.simulation_config.spill_capacity_bytes
        if self.maximum_spill_budget_bytes < source_spill:
            raise ValueError("maximum spill budget is below the source budget")
        if self.fixed_execution_bytes < 0 or self.object_reserve_bytes < 0:
            raise ValueError("capacity deductions must be non-negative")
        if self.dynamic_scratch_reserve_bytes < 0:
            raise ValueError("dynamic scratch reserve must be non-negative")
        shared = shared_residency_footprint(self.program)
        shared_execution = shared.for_device(device.device_id)
        expected_pool = (
            self.source_execution_budget_bytes
            - self.fixed_execution_bytes
            - shared_execution
        )
        if self.admission_facts.pool_capacity_bytes != expected_pool:
            raise ValueError(
                "source execution budget does not reconcile with admission pool"
            )
        expected_object = expected_pool - self.object_reserve_bytes
        if (
            device.capacity_bytes - shared_execution != expected_object
            or self.admission_facts.object_capacity_bytes != expected_object
        ):
            raise ValueError(
                "object reserve does not reconcile with simulation/admission capacity"
            )
        self.admission_facts.validate(self.program)

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of `ShadowSpillPlanningProblem.to_dict`."""
        return _digest(self.to_dict())

    @property
    def transfer_bandwidths(self) -> TransferBandwidths:
        """The device's own copy rates and latencies, as `TransferBandwidths`."""
        device = self.simulation_config.devices[0]
        return TransferBandwidths(
            device.fetch_solo_bandwidth_bytes_per_second,
            device.fetch_concurrent_bandwidth_bytes_per_second,
            device.evict_solo_bandwidth_bytes_per_second,
            device.evict_concurrent_bandwidth_bytes_per_second,
            fetch_latency_ns=device.fetch_latency_ns,
            evict_latency_ns=device.evict_latency_ns,
        )

    def machine_inputs(
        self,
        *,
        execution_budget_bytes: int | None = None,
        spill_budget_bytes: int | None = None,
        transfer_bandwidths: TransferBandwidths | None = None,
    ) -> tuple[SimulationConfig, AdmissionFacts]:
        """Derive the machine and physical facts for one budget and set of rates.

        The program is unchanged; the device capacity, the pool and object
        capacities and the copy rates follow the arguments.

        Parameters
        ----------
        execution_budget_bytes : int, optional
            Device budget, in bytes, positive and at most
            ``maximum_execution_budget_bytes``. None uses the source budget.
        spill_budget_bytes : int, optional
            Spill budget, in bytes, at least zero. None uses the spill
            capacity of ``simulation_config``.
        transfer_bandwidths : TransferBandwidths, optional
            Copy rates, replacing the device's; a latency left None keeps
            the device's. None keeps the device's rates.

        Returns
        -------
        tuple of (SimulationConfig, AdmissionFacts)
            The machine and the physical facts at that budget.

        Raises
        ------
        ValueError
            If a budget is out of range.
        PlanInfeasibleError
            If the budget leaves no positive pool or object capacity.
        """

        execution_budget = (
            self.source_execution_budget_bytes
            if execution_budget_bytes is None
            else execution_budget_bytes
        )
        spill_budget = (
            self.simulation_config.spill_capacity_bytes
            if spill_budget_bytes is None
            else spill_budget_bytes
        )
        if execution_budget <= 0 or spill_budget < 0:
            raise ValueError(
                "execution and spill budgets must be positive/non-negative"
            )
        if execution_budget > self.maximum_execution_budget_bytes:
            raise ValueError(
                "execution budget exceeds the runtime pool used for profiling: "
                f"requested={execution_budget}, "
                f"maximum={self.maximum_execution_budget_bytes}"
            )
        # No ceiling on the spill budget. Nothing measured depends on how
        # large the spill pool was: it holds what was evicted, and a task's
        # profile, a transfer's calibration and the layout are all unmoved by
        # its size. The budget reaches one place, the simulator's spill
        # capacity below, so asking this problem about a larger spill pool is
        # a question it can answer honestly. What a *run* needs is a different
        # matter, and the runtime checks it against the pool it actually has
        # when a plan is made through one. `maximum_spill_budget_bytes` stays
        # as the record of the pool this problem was profiled against.
        shared = shared_residency_footprint(self.program)
        shared_execution = shared.for_device(self.admission_facts.device_id)
        pool_capacity = execution_budget - self.fixed_execution_bytes - shared_execution
        object_capacity = pool_capacity - self.object_reserve_bytes
        if pool_capacity <= 0 or object_capacity <= 0:
            # A well-formed budget too small for what this program holds back
            # is an answer about the program, not a mistake by the caller: a
            # sweep records the point as infeasible and goes on to the next.
            raise PlanInfeasibleError(
                "execution budget leaves no positive pool/object capacity: "
                f"budget={execution_budget}, fixed={self.fixed_execution_bytes}, "
                f"shared={shared_execution}, "
                f"object_reserve={self.object_reserve_bytes}",
                kind="analytic_capacity",
                required_bytes=(
                    self.fixed_execution_bytes
                    + shared_execution
                    + self.object_reserve_bytes
                    + 1
                ),
                capacity_bytes=execution_budget,
            )
        selected_transfer = transfer_bandwidths or self.transfer_bandwidths
        source_device = self.simulation_config.devices[0]
        config = SimulationConfig(
            (
                replace(
                    source_device,
                    capacity_bytes=object_capacity + shared_execution,
                    fetch_solo_bandwidth_bytes_per_second=(
                        selected_transfer.fetch_solo_bytes_per_second
                    ),
                    fetch_concurrent_bandwidth_bytes_per_second=(
                        selected_transfer.fetch_concurrent_bytes_per_second
                    ),
                    evict_solo_bandwidth_bytes_per_second=(
                        selected_transfer.evict_solo_bytes_per_second
                    ),
                    evict_concurrent_bandwidth_bytes_per_second=(
                        selected_transfer.evict_concurrent_bytes_per_second
                    ),
                    fetch_latency_ns=(
                        source_device.fetch_latency_ns
                        if selected_transfer.fetch_latency_ns is None
                        else selected_transfer.fetch_latency_ns
                    ),
                    evict_latency_ns=(
                        source_device.evict_latency_ns
                        if selected_transfer.evict_latency_ns is None
                        else selected_transfer.evict_latency_ns
                    ),
                ),
            ),
            spill_budget,
        )
        facts = replace(
            self.admission_facts,
            pool_capacity_bytes=pool_capacity,
            object_capacity_bytes=object_capacity,
        )
        return config, facts

    def to_dict(self) -> dict[str, object]:
        """Write the problem as a JSON-compatible mapping.

        Returns
        -------
        dict
            The record's ``schema``, the role, the program with its digest,
            the residency boundaries, the capacities, the configuration and
            the physical facts.
        """
        return {
            "schema": _PLANNING_PROBLEM_SCHEMA,
            "role": self.role,
            "program": {
                "digest": self.program.digest,
                "value": self.program.to_dict(),
            },
            "residency": {
                "initial": [item.to_dict() for item in self.initial_residency],
                "final": [item.to_dict() for item in self.final_residency],
            },
            "capacity_contract": {
                "source_execution_budget_bytes": (self.source_execution_budget_bytes),
                "maximum_execution_budget_bytes": (self.maximum_execution_budget_bytes),
                "maximum_spill_budget_bytes": self.maximum_spill_budget_bytes,
                "fixed_execution_bytes": self.fixed_execution_bytes,
                "object_reserve_bytes": self.object_reserve_bytes,
                "dynamic_scratch_reserve_bytes": (self.dynamic_scratch_reserve_bytes),
            },
            "simulation_config": _simulation_config_to_dict(self.simulation_config),
            "admission_facts": self.admission_facts.to_dict(),
        }

    def to_json(self) -> str:
        """Write the problem as compact JSON with sorted keys.

        Returns
        -------
        str
            `ShadowSpillPlanningProblem.to_dict`, encoded.
        """
        return _canonical_json(self.to_dict())

    @classmethod
    def from_value(cls, value: object, path: str) -> ShadowSpillPlanningProblem:
        """Read a problem back from what `ShadowSpillPlanningProblem.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        ShadowSpillPlanningProblem
            The problem.

        Raises
        ------
        ValueError
            If the schema is another one, the program does not match its
            digest, or a field is missing, malformed or inconsistent.
        """
        data = _mapping(value, path)
        if data.get("schema") != _PLANNING_PROBLEM_SCHEMA:
            raise ValueError(f"{path}.schema: unsupported schema")
        program_value = _mapping(data.get("program"), f"{path}.program")
        program = ShadowSpillProgram.from_dict(program_value.get("value"))
        expected_digest = _string(program_value.get("digest"), f"{path}.program.digest")
        if program.digest != expected_digest:
            raise ValueError(f"{path}.program.digest: content digest mismatch")
        residency = _mapping(data.get("residency"), f"{path}.residency")
        initial = _list(residency.get("initial"), f"{path}.residency.initial")
        final = _list(residency.get("final"), f"{path}.residency.final")
        capacity = _mapping(data.get("capacity_contract"), f"{path}.capacity_contract")
        return cls(
            role=_string(data.get("role"), f"{path}.role"),
            program=program,
            initial_residency=tuple(
                ResidencySpec.from_value(item, f"{path}.residency.initial[{index}]")
                for index, item in enumerate(initial)
            ),
            final_residency=tuple(
                ResidencySpec.from_value(item, f"{path}.residency.final[{index}]")
                for index, item in enumerate(final)
            ),
            simulation_config=_simulation_config_from_value(
                data.get("simulation_config"), f"{path}.simulation_config"
            ),
            admission_facts=AdmissionFacts.from_dict(data.get("admission_facts")),
            source_execution_budget_bytes=_integer(
                capacity.get("source_execution_budget_bytes"),
                f"{path}.capacity_contract.source_execution_budget_bytes",
            ),
            maximum_execution_budget_bytes=_integer(
                capacity.get("maximum_execution_budget_bytes"),
                f"{path}.capacity_contract.maximum_execution_budget_bytes",
            ),
            maximum_spill_budget_bytes=_integer(
                capacity.get("maximum_spill_budget_bytes"),
                f"{path}.capacity_contract.maximum_spill_budget_bytes",
            ),
            fixed_execution_bytes=_integer(
                capacity.get("fixed_execution_bytes"),
                f"{path}.capacity_contract.fixed_execution_bytes",
            ),
            object_reserve_bytes=_integer(
                capacity.get("object_reserve_bytes"),
                f"{path}.capacity_contract.object_reserve_bytes",
            ),
            dynamic_scratch_reserve_bytes=_integer(
                capacity.get("dynamic_scratch_reserve_bytes"),
                f"{path}.capacity_contract.dynamic_scratch_reserve_bytes",
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> ShadowSpillPlanningProblem:
        """Read a problem back from what `ShadowSpillPlanningProblem.to_json` wrote.

        Parameters
        ----------
        payload : str
            The JSON text.

        Returns
        -------
        ShadowSpillPlanningProblem
            The problem.

        Raises
        ------
        ValueError
            If the text is not JSON or `ShadowSpillPlanningProblem.from_value`
            refuses its record.
        """
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("planning problem JSON is invalid") from error
        return cls.from_value(value, "plan_program")
