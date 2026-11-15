"""Install the external fixture's source and metadata for fresh phase workers."""

import inspect
from pathlib import Path

from engine_tests.fixtures import (
    definition_operations,
    definition_provider,
    external_definition,
    stochastic_definition,
)


def install(root: Path, *, provider: bool = False, stochastic: bool = False) -> Path:
    if provider and stochastic:
        raise ValueError("select one external model fixture")
    package = root / "external_classifier_fixture"
    package.mkdir(parents=True)
    (package / "base.py").write_text(inspect.getsource(external_definition))
    imports = """from dataclasses import replace
from shadowspill.training.definitions import OptimizerDefinition, Registry
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.definitions import OutputSet, TensorOutput
from . import base as external_definition
"""
    definitions = "\n\n".join(
        inspect.getsource(getattr(definition_operations, name))
        for name in (
            "predictions",
            "objective_binding",
            "create_optimizer",
            "definitions",
        )
    )
    hook = """
def register(registry):
    model, objective, optimizer = definitions()
    registry.add_model(model.definition, source=model.source)
    registry.add_objective(objective.definition, source=objective.source)
    registry.add_optimizer(optimizer.definition, source=optimizer.source)
"""
    if provider:
        source = (
            inspect.getsource(definition_provider)
            .replace(
                "engine_tests.resources._provider",
                "external_classifier_fixture.provider",
            )
            .replace("ss_resource_test", "ss_definition_resource_test")
        )
        (package / "model_provider.py").write_text(source)
        source = (Path(__file__).parents[1] / "resources" / "_provider.py").read_text()
        (package / "provider.py").write_text(
            source.replace("ss_resource_test", "ss_definition_resource_test")
        )
        hook = hook.replace(
            "registry.add_model(model.definition, source=model.source)",
            "from . import model_provider\n"
            "    registry.add_model(replace(model.definition, "
            "config=model_provider.Config, create=model_provider.create, "
            "build=model_provider.describe), source=model.source)",
        )
    if stochastic:
        (package / "stochastic.py").write_text(inspect.getsource(stochastic_definition))
        hook = hook.replace(
            "registry.add_model(model.definition, source=model.source)",
            "from . import stochastic\n"
            "    registry.add_model(replace(model.definition, "
            "create=stochastic.create, initialization=stochastic.initialization), "
            "source=model.source)",
        ).replace(
            "registry.add_objective(objective.definition, source=objective.source)",
            "registry.add_objective(replace(objective.definition, "
            "outputs=stochastic.outputs(objective.definition.outputs), "
            "bind=stochastic.bind), source=objective.source)",
        )
    (package / "__init__.py").write_text(imports + "\n\n" + definitions + "\n" + hook)
    metadata = root / "external_classifier_fixture-1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: external-classifier-fixture\nVersion: 1\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[shadowspill.definitions]\nclassifier = external_classifier_fixture:register\n"
    )
    return root
