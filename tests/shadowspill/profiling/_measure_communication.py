"""Fresh ordinary-allocator worker with execution/build imports prohibited."""

import importlib.abc
import json
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == name or fullname.startswith(name + ".")
            for name in (
                "shadowspill.runtime",
                "shadowspill.execution",
                "shadowspill.store",
                "shadowspill.build",
                "shadowspill.pytorch.capture",
            )
        ):
            raise AssertionError(f"calibration must not import {fullname}")


sys.meta_path.insert(0, Block())

from dataclasses import replace  # noqa: E402

import torch  # noqa: E402

from shadowspill import CommunicationInfo, measure_communication  # noqa: E402
from shadowspill.pytorch.toolchain import build_setup  # noqa: E402
from tests.shadowspill.offline._example import program  # noqa: E402
from tests.spill_pool import spill_pool  # noqa: E402

baseline = torch.cuda.memory_allocated()
spill = spill_pool(64 << 20)
info = measure_communication(
    sizes_bytes=(1024, 1 << 20, 16 << 20),
    warmup_seconds=0.001,
    measure_seconds=0.005,
    spill=spill,
)
assert torch.cuda.memory.get_allocator_backend() == "native"
assert torch.cuda.memory_allocated() == baseline
for direction in ("fetch", "evict"):
    assert 1e6 < getattr(info, f"{direction}_bytes_per_second") < 1e12
    assert 0 <= getattr(info, f"{direction}_latency_seconds") < 0.01
    assert getattr(info, f"{direction}_concurrent_bytes_per_second") > 1e6
assert len(info.metadata["observations"]) == (8 if spill.addressable else 4)
assert info.metadata["spill_kind"] == spill.kind_name
info.save(sys.argv[1])
assert CommunicationInfo.load(sys.argv[1]).to_dict() == info.to_dict()
# The Program form never restores compiled tasks or creates runtime pools.
# Its frozen setup metadata must compare correctly with live lists/dicts.
source = replace(program(), setup=build_setup(torch.device("cuda", 0)))
derived = measure_communication(
    program=source,
    collectives=[
        {
            "operation": "all_reduce",
            "ranks": [0],
            "tensor_bytes": 64,
            "dtype": "torch.float32",
        }
    ],
    sizes_bytes=(1024, 1 << 20, 16 << 20),
    warmup_seconds=0.001,
    measure_seconds=0.005,
    spill=spill,
)
assert derived.measurements[0]["duration_seconds"] == 0
assert not torch.distributed.is_initialized()
assert torch.cuda.memory_allocated() == baseline
print(json.dumps(info.to_dict()))
