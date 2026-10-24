"""Optional GPU communication calibration, independent of execution pools."""

from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import median
from typing import Any

import torch

from shadowspill.profiling.timing import collect_timing_samples, condition_task
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.task.profiling import ProfilingOptions

from .copies import CopyProbe
from .fitting import fit_copy_cost
from .info import CommunicationInfo


def _measure(
    probe: CopyProbe, size: int, directions: tuple[int, ...], options: ProfilingOptions
) -> list[dict[str, Any]]:
    samples: list[tuple[int, ...]] = []

    def sample() -> int:
        values = probe.sample(size, directions)
        samples.append(values)
        return max(values)

    condition_task(sample, options=options)
    samples.clear()
    timing = collect_timing_samples(sample, options=options)
    return [
        {
            "direction": ("fetch", "evict")[direction],
            "bytes": size,
            "concurrent": len(directions) > 1,
            "duration_seconds": median(row[index] for row in samples) / 1e9,
            "samples": len(samples),
            "target_met": timing.window.target_met,
            "relative_mad": timing.relative_mad,
            "half_drift": timing.half_drift,
        }
        for index, direction in enumerate(directions)
    ]


def measure_communication(
    *,
    device: str | torch.device = "cuda",
    sizes_bytes: Sequence[int] = (1024, 1 << 20, 16 << 20, 64 << 20),
    warmup_seconds: float = 0.05,
    measure_seconds: float = 0.18,
) -> CommunicationInfo:
    """Measure local pinned-host copies using the ordinary PyTorch allocator.

    Fit solo startup/bandwidth from the size sweep; measure concurrent rates at
    its largest size. Peak buffers use twice that size on host and twice on GPU.
    Samples and fit errors are retained as metadata. Rates are local to this
    device/host placement, not aggregate multi-GPU host bandwidth. Collective
    calibration is independent and is not part of this initial single-rank probe.
    """
    sizes = tuple(sorted(set(sizes_bytes)))
    if len(sizes) < 2 or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("sizes_bytes needs at least two distinct positive integers")
    options = ProfilingOptions(
        conditioning_seconds=warmup_seconds,
        conditioning_wall_seconds=max(1.0, warmup_seconds * 4),
        measurement_seconds=measure_seconds,
        measurement_wall_seconds=max(1.0, measure_seconds * 4),
    )
    target = resolve_device(device)
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError(
            "communication measurement requires the native PyTorch allocator"
        )
    with torch.cuda.device(target), torch.no_grad():
        probe = CopyProbe(sizes[-1], target)
        observations = [
            row
            for direction in range(2)
            for size in sizes
            for row in _measure(probe, size, (direction,), options)
        ]
        observations.extend(_measure(probe, sizes[-1], (0, 1), options))
    values: dict[str, Any] = {}
    errors = {}
    for direction in ("fetch", "evict"):
        rows = [row for row in observations if row["direction"] == direction]
        rate, latency, error = fit_copy_cost(
            [
                (row["bytes"], row["duration_seconds"])
                for row in rows
                if not row["concurrent"]
            ]
        )
        concurrent = rows[-1]
        if concurrent["duration_seconds"] <= latency:
            raise ValueError("inconsistent copy startup fit; retry with larger sizes")
        # Noise must not make overlapping transfers faster than the solo lane.
        concurrent_rate = min(
            rate, concurrent["bytes"] / (concurrent["duration_seconds"] - latency)
        )
        values[f"{direction}_bytes_per_second"] = rate
        values[f"{direction}_latency_seconds"] = latency
        values[f"{direction}_concurrent_bytes_per_second"] = concurrent_rate
        errors[direction] = error
    return CommunicationInfo(
        **values,
        metadata={
            "source": "ordinary CUDA event measurements of pinned-host copies",
            "measured_at": datetime.now(UTC).isoformat(),
            "device": str(target),
            "gpu": torch.cuda.get_device_name(target),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "observations": observations,
            "fit": "startup + bytes / bandwidth; concurrent at largest size",
            "fit_rms_error_seconds": errors,
            "scope": "one GPU; caller's current host NUMA placement",
        },
    )
