"""Install the external fixture's source and metadata for fresh phase workers."""

import inspect
from pathlib import Path

from engine_tests.fixtures import (
    component_operations,
    component_provider,
    external_component,
    stochastic_component,
)


def install(root: Path, *, provider: bool = False, stochastic: bool = False) -> Path:
    if provider and stochastic:
        raise ValueError("select one external model fixture")
    package = root / "external_classifier_fixture"
    package.mkdir(parents=True)
    (package / "base.py").write_text(inspect.getsource(external_component))
    imports = """from dataclasses import replace
from shadowspill.training.components import OptimizerComponent, Registry
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.components import OutputSet, TensorOutput
from . import base as external_component
"""
    components = "\n\n".join(
        inspect.getsource(getattr(component_operations, name))
        for name in (
            "predictions",
            "objective_binding",
            "create_optimizer",
            "components",
        )
    )
    hook = """
def register(registry):
    model, objective, optimizer = components()
    registry.add_model(model.component, source=model.source)
    registry.add_objective(objective.component, source=objective.source)
    registry.add_optimizer(optimizer.component, source=optimizer.source)
"""
    if provider:
        source = (
            inspect.getsource(component_provider)
            .replace(
                "engine_tests.resources._provider",
                "external_classifier_fixture.provider",
            )
            .replace("ss_resource_test", "ss_component_resource_test")
        )
        (package / "model_provider.py").write_text(source)
        source = (Path(__file__).parents[1] / "resources" / "_provider.py").read_text()
        (package / "provider.py").write_text(
            source.replace("ss_resource_test", "ss_component_resource_test")
        )
        hook = hook.replace(
            "registry.add_model(model.component, source=model.source)",
            "from . import model_provider\n"
            "    registry.add_model(replace(model.component, "
            "config=model_provider.Config, create=model_provider.create, "
            "build=model_provider.describe), source=model.source)",
        )
    if stochastic:
        (package / "stochastic.py").write_text(inspect.getsource(stochastic_component))
        hook = hook.replace(
            "registry.add_model(model.component, source=model.source)",
            "from . import stochastic\n"
            "    registry.add_model(replace(model.component, "
            "create=stochastic.create, initialization=stochastic.initialization), "
            "source=model.source)",
        ).replace(
            "registry.add_objective(objective.component, source=objective.source)",
            "registry.add_objective(replace(objective.component, "
            "outputs=stochastic.outputs(objective.component.outputs), "
            "bind=stochastic.bind), source=objective.source)",
        )
    (package / "__init__.py").write_text(imports + "\n\n" + components + "\n" + hook)
    metadata = root / "external_classifier_fixture-1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: external-classifier-fixture\nVersion: 1\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[shadowspill.components]\nclassifier = external_classifier_fixture:register\n"
    )
    return root
