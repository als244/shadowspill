"""Convenience name for GLM's adapter to the central training runner."""

from shadowspill.training.preparation import run as train


def run(config, *, phase="all"):
    return train(
        "shadowspill_workloads.recipes.text.glm53_flash_lora.adapter",
        config,
        phase=phase,
    )
