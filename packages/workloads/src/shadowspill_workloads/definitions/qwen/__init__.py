"""Packed dense and MoE Qwen families through the public extension API."""

from functools import partial

from shadowspill.training.definitions import ModelDefinition, PreparationDefinition
from shadowspill_workloads.inputs.text import preparation as inputs

from .. import training_options
from ..moe_provider import PROVIDERS, gradient_destination_rules
from ..source import definition_source
from . import configuration


def register(registry):
    for settings in (
        configuration.Qwen3Settings,
        configuration.Qwen35Settings,
        configuration.Qwen3MoESettings,
        configuration.Qwen35MoESettings,
    ):
        family = settings.family
        preset = f"shadowspill/{family}_packed@1"
        source = definition_source(
            (
                "definitions/qwen",
                "definitions/training_options.py",
                *(
                    (
                        "definitions/expert_parallel.py",
                        "definitions/expert_allocations.py",
                        "definitions/moe_allocations.py",
                        "definitions/moe_provider.py",
                    )
                    if settings.moe
                    else ()
                ),
                *(
                    ("definitions/hybrid_allocations.py",)
                    if settings.hybrid or settings.moe
                    else ()
                ),
                "definitions/text_allocations.py",
                "inputs/text/preparation.py",
                "initialization",
                f"models/{family}",
                "models/shared",
                "lora",
            ),
            providers=(
                "mlops",
                *(
                    ("fla", "flash_attn_interface", "flash_attn_3", "flash_attn_config")
                    if settings.hybrid or settings.moe
                    else ()
                ),
                "huggingface_hub",
                "httpx",
                "httpcore",
                "httpx2",
                "httpcore2",
                *((*PROVIDERS, "moonep") if settings.moe else ()),
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
                capabilities=("target_logprobs",)
                + (("router_statistics",) if settings.moe else ()),
                input_adapter=inputs.adapter,
                preparation=preset,
                configure=partial(training_options.configure, label="Qwen"),
                build=configuration.build,
                capture_context=training_options.capture_context,
                gradient_destination_rules=gradient_destination_rules
                if settings.moe
                else None,
                initialization=configuration.initialization,
                temporary_initialization=configuration.initialization,
            ),
            source=source,
        )
