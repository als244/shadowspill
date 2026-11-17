"""Shared registration for packed decoder families using the extension API."""

from functools import partial

from shadowspill.training.definitions import ModelDefinition, PreparationDefinition
from shadowspill_workloads.inputs.text import preparation as inputs
from shadowspill_workloads.models.families import family as family_definition

from .. import training_options
from ..moe_provider import PROVIDERS, gradient_destination_rules
from ..source import definition_source
from . import configuration


def register(registry, settings, *, create=configuration.create):
    family = settings.family
    preset = f"shadowspill/{family}_packed@1"
    source = definition_source(
        (
            "definitions/decoder",
            "models/families.py",
            "models/registration.py",
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
            create,
            inputs.schema,
            capabilities=("target_logprobs",)
            + (("router_statistics",) if settings.moe else ()),
            input_adapter=inputs.adapter,
            preparation=preset,
            configure=partial(training_options.configure, label=family),
            build=configuration.build,
            capture_context=training_options.capture_context,
            gradient_destination_rules=gradient_destination_rules
            if settings.moe
            else None,
            initialization=configuration.initialization,
            temporary_initialization=configuration.initialization,
            parameter_dtypes=family_definition(family).parameter_dtypes,
        ),
        source=source,
    )
