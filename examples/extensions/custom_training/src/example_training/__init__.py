"""One installed registration hook; no model or framework loads on inspection."""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from shadowspill.training.components import (
    ModelComponent,
    ObjectiveComponent,
    OptimizerComponent,
)
from shadowspill_client.extensions import ComponentSource

from . import model, objective, optimizer


def source():
    # All implementation code is in this package. Torch is pinned by the engine
    # toolchain; add other implementation dependencies to this closure if used.
    root = Path(__file__).parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    package = "example-training-components"
    return ComponentSource(package, version(package), digest.hexdigest())


def register(registry):
    identity = source()
    registry.add_model(
        ModelComponent(
            "example/linear_regressor@1",
            model.Config,
            model.create,
            model.inputs,
            capabilities=("scalar_prediction",),
            initialization=model.initialization,
        ),
        source=identity,
    )
    registry.add_objective(
        ObjectiveComponent(
            "example/weighted_mse@1",
            objective.compute,
            inputs=objective.INPUTS,
            requires=("scalar_prediction",),
            participation=objective.PARTICIPATION,
            padding_safe=True,
            outputs=objective.OUTPUTS,
        ),
        source=identity,
    )
    registry.add_optimizer(
        OptimizerComponent(
            "example/sgd@1", optimizer.create, optimizer.CONTROLS, optimizer.BINDINGS
        ),
        source=identity,
    )
