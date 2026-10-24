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

import torch  # noqa: E402

from shadowspill import CommunicationInfo, measure_communication  # noqa: E402

baseline = torch.cuda.memory_allocated()
info = measure_communication(
    sizes_bytes=(1024, 1 << 20, 16 << 20),
    warmup_seconds=0.001,
    measure_seconds=0.005,
)
assert torch.cuda.memory.get_allocator_backend() == "native"
assert torch.cuda.memory_allocated() == baseline
for direction in ("fetch", "evict"):
    assert 1e6 < getattr(info, f"{direction}_bytes_per_second") < 1e12
    assert 0 <= getattr(info, f"{direction}_latency_seconds") < 0.01
    assert getattr(info, f"{direction}_concurrent_bytes_per_second") > 1e6
assert len(info.metadata["observations"]) == 8
info.save(sys.argv[1])
assert CommunicationInfo.load(sys.argv[1]).to_dict() == info.to_dict()
print(json.dumps(info.to_dict()))
