"""The single list of supplied families; metadata stays beside each model."""

from importlib import import_module

from .registration import Family

BUILTINS = (
    "llama3",
    "qwen3",
    "qwen35",
    "olmoe",
    "qwen3moe",
    "qwen35moe",
    "glm53",
    "glm53_flash",
    "regression",
)


def family(name: str) -> Family:
    if name not in BUILTINS:
        raise ValueError(f"unknown model family: {name!r}")
    result = import_module(f"shadowspill_workloads.models.{name}.registration").FAMILY
    if result.name != name:
        raise ValueError(f"family registration name differs from its package: {name}")
    return result


def families() -> tuple[Family, ...]:
    return tuple(family(name) for name in BUILTINS)


def checkpoint_families(*, training: bool = False) -> dict[str, Family]:
    result = {}
    for item in families():
        if training and item.training is None:
            continue
        for kind in item.checkpoint_types:
            if kind in result:
                raise ValueError(f"checkpoint model_type registered twice: {kind!r}")
            result[kind] = item
    return result
