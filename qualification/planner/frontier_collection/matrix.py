"""Stable point identities for one Program and planner-frontier grid."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from shadowspill.pytorch import PressureFitProgram, TransferBandwidths

from .config import BandwidthScale, FrontierGrid, TransferBandwidthBaseline


@dataclass(frozen=True, slots=True)
class FrontierAxes:
    """Budget and scale axes before Program-specific calibration is applied."""

    grid_name: str
    execution_budget_bytes: int
    spill_budget_bytes: int
    bandwidth_scale: BandwidthScale

    @property
    def label(self) -> str:
        execution = self.execution_budget_bytes >> 30
        spill = self.spill_budget_bytes >> 30
        return (
            f"execution-{execution}GiB_spill-{spill}GiB_"
            f"transfer-{self.bandwidth_scale.label}"
        )


@dataclass(frozen=True, slots=True)
class FrontierPointRequest:
    """One exact Program, budget, and calibrated-transfer planner input."""

    axes: FrontierAxes
    program_digest: str
    role: str
    transfer_bandwidths: TransferBandwidths

    @property
    def point_id(self) -> str:
        return self.axes.label

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "shadowspill.pressurefit_frontier_point_request/v1",
            "grid_name": self.axes.grid_name,
            "point_id": self.point_id,
            "program_digest": self.program_digest,
            "program_role": self.role,
            "memory_budgets": {
                "execution_bytes": self.axes.execution_budget_bytes,
                "spill_bytes": self.axes.spill_budget_bytes,
            },
            "transfer_bandwidths": self.transfer_bandwidths.to_dict(),
        }

    @classmethod
    def from_value(cls, value: object) -> FrontierPointRequest:
        if not isinstance(value, dict):
            raise ValueError("frontier point request must be an object")
        budgets = value.get("memory_budgets")
        if not isinstance(budgets, dict):
            raise ValueError("frontier point request has no memory_budgets")
        grid_name = value.get("grid_name")
        point_id = value.get("point_id")
        program_digest = value.get("program_digest")
        role = value.get("program_role")
        execution = budgets.get("execution_bytes")
        spill = budgets.get("spill_bytes")
        transfer = TransferBandwidths.from_value(value.get("transfer_bandwidths"))
        grid_name = _string(grid_name, "grid_name")
        point_id = _string(point_id, "point_id")
        program_digest = _string(program_digest, "program_digest")
        role = _string(role, "program_role")
        if isinstance(execution, bool) or not isinstance(execution, int):
            raise ValueError("frontier point execution budget is invalid")
        if isinstance(spill, bool) or not isinstance(spill, int):
            raise ValueError("frontier point spill budget is invalid")
        scale = BandwidthScale(
            transfer.scale_numerator,
            transfer.scale_denominator,
        )
        result = cls(
            FrontierAxes(grid_name, execution, spill, scale),
            program_digest,
            role,
            transfer,
        )
        if result.point_id != point_id:
            raise ValueError("frontier point label does not match its axes")
        return result


def expand_grid_axes(grids: tuple[FrontierGrid, ...]) -> tuple[FrontierAxes, ...]:
    """Expand and de-duplicate all configured Cartesian point grids."""

    result: list[FrontierAxes] = []
    identities: set[tuple[int, int, BandwidthScale]] = set()
    for grid in grids:
        for execution in grid.execution_budgets:
            for spill in grid.spill_budgets:
                for scale in grid.bandwidth_scales:
                    identity = (execution, spill, scale)
                    if identity in identities:
                        raise ValueError(
                            "frontier grids contain a duplicate point: "
                            f"execution={execution}, spill={spill}, scale={scale}"
                        )
                    identities.add(identity)
                    result.append(FrontierAxes(grid.name, execution, spill, scale))
    return tuple(result)


def expand_frontier_points(
    program: PressureFitProgram,
    grids: tuple[FrontierGrid, ...],
    *,
    transfer_baseline: TransferBandwidthBaseline,
) -> tuple[FrontierPointRequest, ...]:
    """Bind axes to one globally frozen concurrent transfer calibration."""

    return tuple(
        FrontierPointRequest(
            axes=axes,
            program_digest=program.digest,
            role=program.role,
            transfer_bandwidths=TransferBandwidths(
                fetch_bytes_per_second=_scale(
                    transfer_baseline.fetch_bytes_per_second,
                    axes.bandwidth_scale,
                ),
                evict_bytes_per_second=_scale(
                    transfer_baseline.evict_bytes_per_second,
                    axes.bandwidth_scale,
                ),
                scale_numerator=axes.bandwidth_scale.numerator,
                scale_denominator=axes.bandwidth_scale.denominator,
                calibration_digest=transfer_baseline.digest,
                provenance=transfer_baseline.provenance,
            ),
        )
        for axes in expand_grid_axes(grids)
    )


def _scale(value: int, scale: BandwidthScale) -> int:
    return max(1, value * scale.numerator // scale.denominator)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"frontier point request {name} is invalid")
    return value


__all__ = [
    "FrontierAxes",
    "FrontierPointRequest",
    "expand_frontier_points",
    "expand_grid_axes",
]
