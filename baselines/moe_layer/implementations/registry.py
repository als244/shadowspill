"""Backend discovery without importing CUDA packages before device selection."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Backend:
    module: str
    environment: str
    parallelism: str
    label: str
    color: str
    marker: str
    reuse_model_across_tokens: bool = True


BACKENDS = {
    "sonicmoe": Backend("sonicmoe", "mlops", "dp", "DP (SonicMoE)", "#d55e00", "o"),
    "deepep": Backend(
        "deepep", "nvidia-training", "ep", "EP8 (DeepEP)", "#174ea6", "D"
    ),
    "hybridep": Backend(
        "hybridep", "nvidia-training", "ep", "EP8 (HybridEP)", "#168aad", "^"
    ),
    "ncclep": Backend(
        "ncclep", "nvidia-training", "ep", "EP8 (NcclEP)", "#728ec5", "s"
    ),
    "quackmoe": Backend(
        "quackmoe", "mlops", "ep", "EP8 (QuackMoE)", "#7c3aed", "v", False
    ),
}


def build(backend, **configuration):
    entry = BACKENDS[backend]
    module = import_module(f"{__package__}.{entry.module}")
    return module.build(**configuration)
