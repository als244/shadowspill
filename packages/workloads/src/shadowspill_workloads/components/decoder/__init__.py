"""Shared registration for packed decoder families using the extension API."""

from functools import partial

from shadowspill.training.components import ModelComponent, PreparationComponent
from shadowspill_workloads.inputs.text import preparation as inputs
from shadowspill_workloads.models.families import family as family_definition

from .. import training_options
from ..moe_provider import PROVIDERS, gradient_destination_rules
from ..source import component_source
from . import configuration


def _create_training_decoder(factory, settings):
    model = factory(settings)
    if not callable(getattr(model, "token_features", None)) or not callable(
        getattr(getattr(model, "lm_head", None), "objective", None)
    ):
        raise TypeError(
            f"{settings.family} packed decoder training requires model.token_features "
            "and lm_head.objective to consume vocabulary blocks inside the loss. "
            "Use mlops.modules.LanguageModelHead or implement its bounded objective."
        )
    return model


def register(
    registry,
    settings,
    *,
    create=configuration.create,
    build=configuration.build,
    initialization=configuration.initialization,
    capture_context=training_options.capture_context,
    capabilities=None,
    sources=(),
    providers=(),
):
    family = settings.family
    preset = f"shadowspill/{family}_packed@1"
    source = component_source(
        (
            "components/decoder",
            "models/families.py",
            "models/registration.py",
            "components/training_options.py",
            *(
                (
                    "components/expert_parallel.py",
                    "components/expert_allocations.py",
                    "components/moe_allocations.py",
                    "components/moe_provider.py",
                )
                if settings.moe
                else ()
            ),
            *(
                ("components/hybrid_allocations.py",)
                if settings.hybrid or settings.moe
                else ()
            ),
            "components/text_allocations.py",
            "inputs/text/preparation.py",
            "initialization",
            f"models/{family}",
            "models/shared",
            "lora",
            *sources,
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
            *providers,
        ),
    )
    registry.add_preparation(
        PreparationComponent(
            preset,
            partial(inputs.defaults, preset=preset),
            inputs.buffers,
            inputs.representative,
        ),
        source=source,
    )
    registry.add_model(
        ModelComponent(
            f"shadowspill/{family}@1",
            settings,
            partial(_create_training_decoder, create),
            inputs.schema,
            capabilities=capabilities
            if capabilities is not None
            else ("target_logprobs",)
            + (("router_statistics",) if settings.moe else ()),
            input_adapter=inputs.adapter,
            preparation=preset,
            configure=partial(training_options.configure, label=family),
            build=build,
            capture_context=capture_context,
            gradient_destination_rules=gradient_destination_rules
            if settings.moe
            else None,
            initialization=initialization,
            temporary_initialization=initialization,
            parameter_dtypes=family_definition(family).parameter_dtypes,
        ),
        source=source,
    )
