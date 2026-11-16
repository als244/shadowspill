"""Installed packed Llama using the same extension hooks as custom models."""

from functools import partial
from hashlib import sha256
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

from shadowspill.training.definitions import ModelDefinition, PreparationDefinition
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_workloads.inputs.text import preparation as inputs

from . import configuration

PRESET = "shadowspill/llama3_packed@1"


def register(registry):
    root = Path(__file__).resolve().parents[2]
    digest = sha256()
    # Include model, LoRA, initializer and provider semantics without importing
    # any of them in the CPU controller. Editable dependency changes are pinned.
    paths = [
        root / "recipes/text/initialization.py",
        root / "inputs/text/preparation.py",
        root / "definitions/text_allocations.py",
    ]
    for name in ("definitions/llama", "models/llama3", "models/shared", "lora"):
        paths.extend(sorted((root / name).rglob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    spec = find_spec("mlops")
    if spec is not None and spec.origin is not None:
        provider = Path(spec.origin).parent
        digest.update(version("mlops").encode())
        for path in sorted(provider.rglob("*.py")):
            digest.update(str(path.relative_to(provider)).encode())
            digest.update(path.read_bytes())
    source = DefinitionSource(
        "shadowspill-workloads", version("shadowspill-workloads"), digest.hexdigest()
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
            configure=configuration.configure,
            build=configuration.build,
            capture_context=configuration.capture_context,
            initialization=configuration.initialization,
        ),
        source=source,
    )
