"""Installed packed Qwen3 and Qwen3.5 MoE through the public extension API."""

from functools import partial

from shadowspill.training.definitions import ModelDefinition, PreparationDefinition
from shadowspill_workloads.inputs.text import preparation as inputs

from ..source import definition_source
from . import configuration


def register(registry):
    for settings in (configuration.Qwen3Settings, configuration.Qwen35Settings):
        family = settings.family
        preset = f"shadowspill/{family}_packed@1"
        source = definition_source(
            (
                "definitions/qwen_moe",
                "definitions/moe_allocations.py",
                "definitions/text_allocations.py",
                "inputs/text/preparation.py",
                "initialization",
                f"models/{family}",
                "models/shared",
                "lora",
            ),
            providers=(
                "mlops",
                "fla",
                "flash_attn_interface",
                "flash_attn_3",
                "flash_attn_config",
                "huggingface_hub",
                "httpx",
                "httpcore",
                "httpx2",
                "httpcore2",
            ),
        )
        registry.add_preparation(
            PreparationDefinition(
                preset,
                partial(inputs.defaults, preset=preset),
                inputs.buffers,
                inputs.representative,
            ),
            source=source,
        )
        registry.add_model(
            ModelDefinition(
                f"shadowspill/{family}@1",
                settings,
                configuration.create,
                inputs.schema,
                capabilities=("target_logprobs", "router_statistics"),
                input_adapter=inputs.adapter,
                preparation=preset,
                configure=configuration.configure,
                build=configuration.build,
                capture_context=configuration.capture_context,
                initialization=configuration.initialization,
                temporary_initialization=configuration.initialization,
            ),
            source=source,
        )
