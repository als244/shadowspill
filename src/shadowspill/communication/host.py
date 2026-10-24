"""Optional GPU communication calibration, independent of execution pools."""

from datetime import UTC, datetime
from statistics import median
from typing import Any

import torch

from shadowspill.profiling.timing import collect_timing_samples, condition_task
from shadowspill.task.profiling import ProfilingOptions

from .binding.groups import GroupBindings
from .copies import CopyProbe
from .fitting import fit_copy_cost
from .info import CommunicationInfo


def _measure(
    probe: CopyProbe,
    size: int,
    directions: tuple[int, ...],
    options: ProfilingOptions,
    bindings: GroupBindings | None,
) -> list[dict[str, Any]]:
    samples: list[tuple[int, ...]] = []

    def sample() -> int:
        values = probe.sample(size, directions)
        samples.append(values)
        return max(values)

    decide = None if bindings is None else bindings.continue_sampling
    condition_task(sample, options=options, continue_while=decide)
    samples.clear()
    timing = collect_timing_samples(sample, options=options, continue_while=decide)
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


def measure_host_copies(
    *,
    device: torch.device,
    sizes: tuple[int, ...],
    options: ProfilingOptions,
    bindings: GroupBindings | None = None,
) -> CommunicationInfo:
    """Fit local host-copy rates, with coordinated repetitions when distributed."""
    target = device
    with torch.cuda.device(target), torch.no_grad():
        probe, failure = None, None
        try:
            probe = CopyProbe(sizes[-1], target)
        except Exception as error:
            if bindings is None:
                raise
            failure = str(error)
        if bindings is not None:
            bindings.check(failure)
        assert probe is not None
        observations = [
            row
            for direction in range(2)
            for size in sizes
            for row in _measure(probe, size, (direction,), options, bindings)
        ]
        observations.extend(_measure(probe, sizes[-1], (0, 1), options, bindings))
    del probe  # Release copy buffers before allocating any collective buffers.
    values: dict[str, Any] = {}
    errors = {}
    for direction in ("fetch", "evict"):
        rows = [row for row in observations if row["direction"] == direction]
        rate, latency, fit_error = fit_copy_cost(
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
        errors[direction] = fit_error
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
