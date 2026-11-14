"""Install the external fixture's source and metadata for fresh phase workers."""

import inspect
from pathlib import Path

from engine_tests.fixtures import definition_operations, external_definition


def install(root: Path) -> Path:
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
