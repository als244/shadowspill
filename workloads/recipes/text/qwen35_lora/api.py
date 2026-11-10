"""Convenience name for Qwen35's adapter to the central training runner."""

from workloads.recipes.training import run as train


def run(config, *, phase="all"):
    return train("workloads.recipes.text.qwen35_lora.adapter", config, phase=phase)
