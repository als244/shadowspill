"""Measure frontend and runtime ownership across model-state relocation."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import json
import weakref
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    externalize_model_state,
    relocate_model_state,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics


class Payload(nn.Module):
    """One storage root with a predictable requested byte count."""

    def __init__(self, size_bytes: int) -> None:
        super().__init__()
        elements = size_bytes // torch.tensor([], dtype=torch.float32).element_size()
        self.value = nn.Parameter(torch.randn(elements), requires_grad=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-mib", type=int, default=64)
    parser.add_argument(
        "--release-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="release the source owner after relocation (default: true)",
    )
    parser.add_argument("--library", type=Path)
    arguments = parser.parse_args()
    if arguments.state_mib <= 0:
        parser.error("--state-mib must be positive")

    state_bytes = arguments.state_mib << 20
    samples: dict[str, object] = {"initial": _sample(None)}
    source = Payload(state_bytes)
    source_reference = weakref.ref(source)
    source_pointer = int(source.value.untyped_storage().data_ptr())
    samples["source_created"] = _sample(None)

    runtime = Runtime(
        pools={
            "execution": device(
                physical_capacity=2 << 30,
                provider_headroom=512 << 20,
            ),
            "spill": pinned_host(capacity=max(512 << 20, state_bytes * 2)),
        },
        library_path=arguments.library,
    )
    samples["runtime_initialized"] = _sample(runtime)
    relocated = relocate_model_state(
        source,
        runtime=runtime,
        pool="spill",
        release_source=arguments.release_source,
    )
    spill_pointer = int(relocated.value.untyped_storage().data_ptr())
    samples["relocated"] = _sample(runtime)
    del source
    _release_python_memory()
    samples["caller_source_deleted"] = {
        **_sample(runtime),
        "source_alive": source_reference() is not None,
    }

    externalize_model_state(relocated, runtime=runtime, release_runtime=True)
    _release_python_memory()
    samples["externalized"] = {
        **_sample(runtime),
        "source_alive": source_reference() is not None,
    }
    external_pointer = int(relocated.value.untyped_storage().data_ptr())
    runtime.close()
    result = {
        "schema": "shadowspill.state_relocation_measurement/v1",
        "state_bytes": state_bytes,
        "release_source": arguments.release_source,
        "source_pointer": source_pointer,
        "spill_pointer": spill_pointer,
        "external_pointer": external_pointer,
        "spill_differs_from_source": spill_pointer != source_pointer,
        "external_differs_from_spill": external_pointer != spill_pointer,
        "samples": samples,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _sample(runtime: Runtime | None) -> dict[str, object]:
    result: dict[str, object] = {"process_rss_bytes": _resident_bytes()}
    if runtime is None:
        return result
    statistics = AdapterStatistics()
    status = int(
        runtime._installed.library.shadowspill_pytorch_allocator_statistics(
            ctypes.byref(statistics)
        )
    )
    if status != 0:
        raise RuntimeError(f"runtime statistics failed with status {status}")
    result.update(
        {
            "registered_objects": int(statistics.runtime.registered_objects),
            "spill_allocated_bytes": int(statistics.runtime.spill_allocated_bytes),
            "spill_peak_allocated_bytes": int(
                statistics.runtime.spill_peak_allocated_bytes
            ),
            "bytes_fetched": int(statistics.runtime.bytes_fetched),
            "bytes_evicted": int(statistics.runtime.bytes_evicted),
        }
    )
    return result


def _resident_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) << 10
    raise RuntimeError("/proc/self/status omitted VmRSS")


def _release_python_memory() -> None:
    gc.collect()
    with contextlib.suppress(AttributeError):
        ctypes.CDLL(None).malloc_trim(0)


if __name__ == "__main__":
    raise SystemExit(main())
