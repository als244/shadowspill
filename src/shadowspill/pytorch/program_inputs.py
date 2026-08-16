"""Reusable inputs for framework-independent PressureFit selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from shadowspill.ir import Program, ResidencySpec
from shadowspill.planner import AdmissionTopology, PressureFitOptions
from shadowspill.simulator import SimulationConfig

from .program_serialization import (
    _canonical_json,
    _digest,
    _integer,
    _list,
    _mapping,
    _optional_string,
    _options_from_value,
    _options_to_dict,
    _simulation_config_from_value,
    _simulation_config_to_dict,
    _string,
)

_PRESSUREFIT_PROGRAM_SCHEMA = "shadowspill.pressurefit_program/v1"


@dataclass(frozen=True, slots=True)
class TransferBandwidths:
    """Fetch/evict bandwidths consumed by planning and simulation."""

    fetch_bytes_per_second: int
    evict_bytes_per_second: int
    scale_numerator: int = 1
    scale_denominator: int = 1
    calibration_digest: str | None = None
    provenance: str | None = None

    def __post_init__(self) -> None:
        if self.fetch_bytes_per_second <= 0 or self.evict_bytes_per_second <= 0:
            raise ValueError("transfer bandwidths must be positive")
        if self.scale_numerator <= 0 or self.scale_denominator <= 0:
            raise ValueError("transfer bandwidth scale must be positive")
        if self.calibration_digest is not None and len(self.calibration_digest) != 64:
            raise ValueError("calibration_digest must be SHA-256 or None")

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_digest": self.calibration_digest,
            "evict_bytes_per_second": self.evict_bytes_per_second,
            "fetch_bytes_per_second": self.fetch_bytes_per_second,
            "provenance": self.provenance,
            "scale_denominator": self.scale_denominator,
            "scale_numerator": self.scale_numerator,
        }

    @classmethod
    def from_value(
        cls, value: object, path: str = "transfer_bandwidths"
    ) -> TransferBandwidths:
        data = _mapping(value, path)
        return cls(
            fetch_bytes_per_second=_integer(
                data.get("fetch_bytes_per_second"),
                f"{path}.fetch_bytes_per_second",
            ),
            evict_bytes_per_second=_integer(
                data.get("evict_bytes_per_second"),
                f"{path}.evict_bytes_per_second",
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
        )


@dataclass(frozen=True, slots=True)
class MemoryBudgets:
    """Physical execution and spill budgets for one annotated plan."""

    execution_bytes: int
    spill_bytes: int

    def __post_init__(self) -> None:
        if self.execution_bytes <= 0:
            raise ValueError("execution memory budget must be positive")
        if self.spill_bytes < 0:
            raise ValueError("spill memory budget must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "execution_bytes": self.execution_bytes,
            "spill_bytes": self.spill_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str = "memory_budgets") -> MemoryBudgets:
        data = _mapping(value, path)
        return cls(
            execution_bytes=_integer(
                data.get("execution_bytes"), f"{path}.execution_bytes"
            ),
            spill_bytes=_integer(data.get("spill_bytes"), f"{path}.spill_bytes"),
        )


@dataclass(frozen=True, slots=True)
class PressureFitProgram:
    """Self-contained pre-PressureFit boundary for one Program variant."""

    role: str
    program: Program
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    simulation_config: SimulationConfig
    admission_topology: AdmissionTopology
    source_execution_budget_bytes: int
    maximum_execution_budget_bytes: int
    maximum_spill_budget_bytes: int
    fixed_execution_bytes: int
    object_reserve_bytes: int
    dynamic_scratch_reserve_bytes: int
    options: PressureFitOptions = field(default_factory=PressureFitOptions)

    def __post_init__(self) -> None:
        if self.role not in {"initial", "recurrent", "forward"}:
            raise ValueError(f"unsupported Program role {self.role!r}")
        if len(self.simulation_config.devices) != 1:
            raise ValueError("PressureFitProgram currently requires one device")
        device = self.simulation_config.devices[0]
        if device.device_id != self.admission_topology.device_id:
            raise ValueError("simulation and admission devices differ")
        if self.source_execution_budget_bytes <= 0:
            raise ValueError("source execution budget must be positive")
        if self.maximum_execution_budget_bytes < self.source_execution_budget_bytes:
            raise ValueError("maximum execution budget is below the source budget")
        if self.maximum_spill_budget_bytes < self.simulation_config.host_capacity_bytes:
            raise ValueError("maximum spill budget is below the source budget")
        if self.fixed_execution_bytes < 0 or self.object_reserve_bytes < 0:
            raise ValueError("capacity deductions must be non-negative")
        if self.dynamic_scratch_reserve_bytes < 0:
            raise ValueError("dynamic scratch reserve must be non-negative")
        expected_pool = self.source_execution_budget_bytes - self.fixed_execution_bytes
        if self.admission_topology.pool_capacity_bytes != expected_pool:
            raise ValueError(
                "source execution budget does not reconcile with admission pool"
            )
        expected_object = expected_pool - self.object_reserve_bytes
        if device.capacity_bytes != expected_object or (
            self.admission_topology.object_capacity_bytes != expected_object
        ):
            raise ValueError(
                "object reserve does not reconcile with simulation/admission capacity"
            )
        self.admission_topology.validate(self.program)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def transfer_bandwidths(self) -> TransferBandwidths:
        device = self.simulation_config.devices[0]
        return TransferBandwidths(
            device.fetch_bandwidth_bytes_per_second,
            device.evict_bandwidth_bytes_per_second,
        )

    def pressurefit_inputs(
        self,
        *,
        execution_budget_bytes: int | None = None,
        spill_budget_bytes: int | None = None,
        transfer_bandwidths: TransferBandwidths | None = None,
    ) -> tuple[SimulationConfig, AdmissionTopology]:
        """Rebase budget-dependent inputs without changing the Program."""

        execution_budget = (
            self.source_execution_budget_bytes
            if execution_budget_bytes is None
            else execution_budget_bytes
        )
        spill_budget = (
            self.simulation_config.host_capacity_bytes
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
        if spill_budget > self.maximum_spill_budget_bytes:
            raise ValueError(
                "spill budget exceeds the runtime pool used for profiling: "
                f"requested={spill_budget}, maximum={self.maximum_spill_budget_bytes}"
            )
        pool_capacity = execution_budget - self.fixed_execution_bytes
        object_capacity = pool_capacity - self.object_reserve_bytes
        if pool_capacity <= 0 or object_capacity <= 0:
            raise ValueError(
                "execution budget leaves no positive pool/object capacity: "
                f"budget={execution_budget}, fixed={self.fixed_execution_bytes}, "
                f"object_reserve={self.object_reserve_bytes}"
            )
        selected_transfer = transfer_bandwidths or self.transfer_bandwidths
        source_device = self.simulation_config.devices[0]
        config = SimulationConfig(
            (
                replace(
                    source_device,
                    capacity_bytes=object_capacity,
                    fetch_bandwidth_bytes_per_second=(
                        selected_transfer.fetch_bytes_per_second
                    ),
                    evict_bandwidth_bytes_per_second=(
                        selected_transfer.evict_bytes_per_second
                    ),
                ),
            ),
            spill_budget,
        )
        topology = replace(
            self.admission_topology,
            pool_capacity_bytes=pool_capacity,
            object_capacity_bytes=object_capacity,
        )
        return config, topology

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _PRESSUREFIT_PROGRAM_SCHEMA,
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
            "admission_topology": self.admission_topology.to_dict(),
            "pressurefit_options": _options_to_dict(self.options),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_value(cls, value: object, path: str) -> PressureFitProgram:
        data = _mapping(value, path)
        if data.get("schema") != _PRESSUREFIT_PROGRAM_SCHEMA:
            raise ValueError(f"{path}.schema: unsupported schema")
        program_value = _mapping(data.get("program"), f"{path}.program")
        program = Program.from_dict(program_value.get("value"))
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
            admission_topology=AdmissionTopology.from_dict(
                data.get("admission_topology")
            ),
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
            options=_options_from_value(
                data.get("pressurefit_options"), f"{path}.pressurefit_options"
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> PressureFitProgram:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("PressureFit Program JSON is invalid") from error
        return cls.from_value(value, "pressurefit_program")
