"""Setup arguments stay portable and retain explicit versus omitted choices."""

from dataclasses import replace

import pytest

from client_tests.test_training_config import resolved
from shadowspill_client import Operation
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.setup import TrainingSetup


def setup(**options):
    config = resolved()
    return TrainingSetup(
        config.model, config.objective, config.batches, config.memory, **options
    )


def test_setup_owns_nested_choices_without_resolving_or_inventing_defaults():
    operations = {"forward": ["prediction", Operation("prediction", "training")]}
    parallelism = {"groups": {"custom": [[0]]}}
    value = setup(operations=operations, parallelism=parallelism)
    operations["forward"].clear()
    parallelism["groups"]["custom"].clear()
    assert len(value.operations["forward"]) == 2
    assert value.parallelism["groups"]["custom"] == ((0,),)
    assert value.prepare is value.execution is value.optimizer is None
    assert decode(encode(value)) == value
    assert TrainingSetup.from_dict(value.to_dict()) == value
    one = replace(value, batch=value.batch[0])
    assert one == value


@pytest.mark.parametrize(
    "changes",
    [
        {"batch": []},
        {"batch": "file.json"},
        {"model": None},
        {"operations": {"forward": "prediction"}},
        {"operations": {"forward": []}},
        {"parallelism": []},
        {"optimizer": lambda: None},
    ],
)
def test_setup_rejects_incomplete_or_executable_protocol_values(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(setup(), **changes)


def test_remote_setup_never_accepts_factory_or_filesystem_keywords():
    with pytest.raises(ValueError, match="unknown fields"):
        TrainingSetup.from_dict({**setup().to_dict(), "factory": "/client/model.py"})
