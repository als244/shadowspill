"""Measure a selected spill backend independently of task profiling."""

from contextlib import ExitStack
from datetime import UTC, datetime
from math import ceil
from typing import Any

import torch

from shadowspill.memory import SpillPool
from shadowspill.task.profiling import ProfilingOptions

from .binding.groups import GroupBindings
from .info import CommunicationInfo
from .spill_probe import SpillProbe


def measure_spill(
    spill: SpillPool,
    *,
    device: torch.device,
    sizes: tuple[int, ...],
    options: ProfilingOptions,
    bindings: GroupBindings | None,
) -> CommunicationInfo:
    with ExitStack() as cleanup:
        failure, probe = None, None
        try:
            probe = cleanup.enter_context(
                SpillProbe(spill, device.index or 0, sizes[-1])
            )
        except Exception as error:
            if bindings is None:
                raise
            failure = str(error)
        if bindings is not None:
            bindings.check(failure)
        assert probe is not None
        observations: list[dict[str, Any]] = []

        def measure(size: int, warmup: int, count: int) -> list[dict[str, Any]]:
            if bindings is not None:
                bindings.check(None)
            result, failure = None, None
            try:
                result = probe.measure(sizes[0], size, warmup, count)
            except Exception as error:
                if bindings is None:
                    raise
                failure = str(error)
            if bindings is not None:
                bindings.check(failure)
            assert result is not None
            return result

        for size in sizes[1:]:
            pilot = measure(size, 1, 1)
            duration = max(row["concurrent_ns"] for row in pilot) / 1e9
            if bindings is not None:
                duration = max(bindings.gather(duration))
            # Native calibration uses three timing windows per large-copy mode.
            warmup = min(4096, ceil(options.conditioning_seconds / max(duration, 1e-6)))
            count = max(
                1,
                min(
                    4096, ceil(options.measurement_seconds / (3 * max(duration, 1e-6)))
                ),
            )
            rows = measure(size, warmup, count)
            observations.extend(
                {
                    "direction": direction,
                    "bytes": size,
                    "samples_per_window": count,
                    "warmup_copies": warmup,
                    **row,
                }
                for direction, row in zip(("fetch", "evict"), rows, strict=True)
            )
    values = {}
    for direction, row in zip(("fetch", "evict"), rows, strict=True):
        values[f"{direction}_bytes_per_second"] = row["solo_bytes_per_second"]
        values[f"{direction}_concurrent_bytes_per_second"] = min(
            row["solo_bytes_per_second"], row["concurrent_bytes_per_second"]
        )
        values[f"{direction}_latency_seconds"] = row["latency_ns"] / 1e9
    return CommunicationInfo(
        **values,
        metadata={
            "source": "native spill lanes, including backend staging and completion",
            "spill_kind": spill.kind_name,
            "measured_at": datetime.now(UTC).isoformat(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "observations": observations,
            "fit": "small-copy completion latency + largest queued-copy bandwidth",
            "probe_sizes_bytes": sizes,
            "scope": "bounded scratch pool; caller's host/NUMA placement",
        },
    )
