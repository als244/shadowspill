"""Convenience name for GLM's adapter to the central training runner."""

from workloads.recipes.training import run as train


def run(config, *, phase="all"):
    return train("workloads.recipes.text.glm53_flash_lora.adapter", config, phase=phase)
