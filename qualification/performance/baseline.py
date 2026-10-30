"""Throughput comparisons only within a measured hardware/configuration scope."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

from qualification.performance.cases import FullModelManifest, manifest_for


@dataclass(frozen=True, slots=True)
class RegressionComparison:
    device_name: str
    baseline_device_name: str
    floor: float | None
    mismatches: tuple[str, ...]

    @property
    def applicable(self) -> bool:
        return self.floor is not None and not self.mismatches

    @property
    def reason(self) -> str:
        if self.floor is None:
            return "no throughput baseline recorded for this cell"
        return "; ".join(self.mismatches) or "hardware and configuration match"

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "applicable": self.applicable, "reason": self.reason}


def regression_comparison(
    manifest: FullModelManifest,
    arguments: argparse.Namespace,
    *,
    device_name: str,
) -> RegressionComparison:
    """Check the scope of the RTX 5090 measurements in ``workloads.full_model``.

    The original measurements used 16/112 GiB pools, BF16 weights, gradients
    and moments, and no masters. Neither a new default nor a CLI override
    changes that provenance. An unrelated machine/configuration still runs
    every other gate and reports its measured throughput.
    """
    floor = manifest.regression_tokens_per_second
    baseline_device = "NVIDIA GeForce RTX 5090"
    mismatches: list[str] = []
    # The floors were measured with the pool in pinned host memory. A run
    # whose pool is on a peer crosses a link many times slower and is held to
    # its simulator's prediction instead: no floor applies to it.
    if getattr(arguments, "remote_spill", None) is not None:
        mismatches.append("a run spilling to a peer has no throughput floor")
    if device_name != baseline_device:
        mismatches.append(f"device {device_name!r} differs from {baseline_device!r}")
    expected = {
        "device_physical_capacity_bytes": 16 << 30,
        "spill_budget_bytes": 112 << 30,
        "external_headroom_bytes": 512 << 20,
        "model_dtype": "bfloat16",
        "master_dtype": "none",
        "grad_dtype": "bfloat16",
        "opt_state_dtype": "bfloat16",
    }
    actual = {**manifest.as_dict(), **manifest.dtypes.as_dict()}
    canonical = manifest_for(manifest.family, manifest.implementation).as_dict()
    for name in (
        "model_config",
        "sequence_length",
        "sequences_per_microbatch",
        "accumulation_count",
        "head_scratch_bytes",
    ):
        expected[name] = canonical[name]
    for name, value in expected.items():
        if actual[name] != value:
            mismatches.append(f"{name}={actual[name]!r}; baseline={value!r}")
    planning_budget = getattr(arguments, "planning_spill_budget_gib", None)
    if planning_budget is not None and planning_budget != 112:
        mismatches.append("planning spill budget differs from baseline (112 GiB)")
    return RegressionComparison(device_name, baseline_device, floor, tuple(mismatches))
