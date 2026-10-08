"""Fresh-process CPU/meta workload discovery without EP dependencies."""
import importlib.abc
import sys

class BlockAcceleratorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"moonep", "quack", "transformer_engine"} or fullname.startswith("mlops.expert_parallel"):
            raise AssertionError(f"Unexpected optional dependency during local discovery: {fullname}")
        return None

sys.meta_path.insert(0, BlockAcceleratorImports())
import torch
from workloads.mlops import OLMoE, OLMoEConfig
from workloads.mlops.qwen3_moe import Qwen3MoE, Qwen3MoEConfig
from workloads.mlops.qwen35_moe import Qwen35MoE, Qwen35MoEConfig

with torch.device("meta"):
    for cls, config in [(OLMoE, OLMoEConfig.throughput()),
                        (Qwen3MoE, Qwen3MoEConfig()),
                        (Qwen35MoE, Qwen35MoEConfig())]:
        model = cls(config)
        assert all(p.is_meta for p in model.parameters())
        assert not list(model.expert_parameters())
        model.close()
        model.close()
        print(cls.__name__, sum(p.numel() for p in model.parameters()), flush=True)
print("PASS: discovery and local meta construction need no EP dependencies", flush=True)
