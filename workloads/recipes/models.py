"""Model sources and training methods, independent of worker import paths."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoRA:
    rank: int = 32
    alpha: float = 32

    def __post_init__(self):
        if (
            type(self.rank) is not int
            or self.rank <= 0
            or isinstance(self.alpha, bool)
            or not isinstance(self.alpha, int | float)
            or not math.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("LoRA rank and alpha must be positive")


@dataclass(frozen=True)
class Model:
    """A CPU-only source description; model tensors are created in workers.

    Architecture resolution belongs to the workload catalog, outside the generic
    runner. Saved configuration records the resolved adapter and model choices.
    """

    _adapter: str
    _settings: dict

    @classmethod
    def from_checkpoint(cls, checkpoint, *, lora=None):
        from workloads.models.catalog.sources import checkpoint_family

        directory = Path(checkpoint).expanduser().resolve()
        metadata = json.loads((directory / "config.json").read_text())
        family = checkpoint_family(metadata)
        adapters = {
            "qwen35moe": "workloads.recipes.text.qwen35_lora.adapter",
            "glm53_flash": "workloads.recipes.text.glm53_flash_lora.adapter",
        }
        if family not in adapters or not isinstance(lora, LoRA):
            raise ValueError(
                "the checkpoint training adapters currently support Qwen35 MoE "
                "and GLM53 Flash with an explicit LoRA configuration"
            )
        return cls(
            adapters[family],
            dict(
                checkpoint=str(directory),
                lora_rank=lora.rank,
                lora_alpha=lora.alpha,
            ),
        )

    @classmethod
    def from_config(cls, name):
        from workloads.models.catalog import entry

        if entry(name).family != "qwen3moe":
            raise ValueError("the full-training adapter currently supports Qwen3 MoE")
        return cls("workloads.recipes.text.qwen3.adapter", {"model": name})
