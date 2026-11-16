"""Installed packed Llama using the same extension hooks as custom models."""

from functools import partial

from shadowspill.training.definitions import ModelDefinition, PreparationDefinition
from shadowspill_workloads.definitions import training_options
from shadowspill_workloads.definitions.source import definition_source
from shadowspill_workloads.inputs.text import preparation as inputs

from . import configuration

PRESET = "shadowspill/llama3_packed@1"


def register(registry):
    source = definition_source(
        (
            "models/families.py",
            "models/registration.py",
            "definitions/training_options.py",
            "definitions/text_allocations.py",
            "inputs/text/preparation.py",
            "initialization",
            "models/llama3",
            "models/shared",
            "lora",
        ),
        providers=(
            "mlops",
            "huggingface_hub",
            "httpx",
            "httpcore",
            "httpx2",
            "httpcore2",
        ),
    )
    registry.add_preparation(
        PreparationDefinition(
            PRESET,
            partial(inputs.defaults, preset=PRESET),
            inputs.buffers,
            inputs.representative,
        ),
        source=source,
    )
    registry.add_model(
        ModelDefinition(
            "shadowspill/llama3@1",
            configuration.Settings,
            configuration.create,
            inputs.schema,
            capabilities=("target_logprobs",),
            input_adapter=inputs.adapter,
            preparation=PRESET,
            configure=partial(training_options.configure, label="Llama"),
            build=configuration.build,
            capture_context=training_options.capture_context,
            initialization=configuration.initialization,
            temporary_initialization=configuration.initialization,
        ),
        source=source,
    )
