"""Portable settings preserve caller intent without importing a model stack."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from shadowspill_client import (
    BatchConfig,
    Datum,
    Execution,
    LoRA,
    Memory,
    Model,
    Objective,
    Operation,
    Optimizer,
    Weights,
)


def test_settings_own_nested_inputs_and_return_independent_json_values():
    config = {"layers": [2, 3], "attention": {"window": 128}}
    model = Model("external/regressor@1", config=config)
    config["layers"].append(9)
    config["attention"]["window"] = 512
    assert model.config["layers"] == (2, 3)
    assert model.config["attention"]["window"] == 128
    with pytest.raises(TypeError):
        model.config["attention"]["window"] = 1
    with pytest.raises(FrozenInstanceError):
        model.name = "other"
    exported = model.to_dict()
    exported["config"]["layers"].append(7)
    assert model.config["layers"] == (2, 3)


def test_resolved_weight_metadata_is_owned_and_roundtrips_with_the_model():
    from shadowspill_client.protocol.codec import decode, encode

    metadata = {"files": [{"name": "weights", "checksum": "abc"}]}
    weights = replace(Weights.pretrained("org/model"), metadata=metadata)
    metadata["files"][0]["checksum"] = "changed"
    model = Model("external/regressor@1", initialize=weights)
    restored = decode(encode(model))
    assert restored == model
    assert restored.initialize.metadata["files"][0]["checksum"] == "abc"
    with pytest.raises(TypeError):
        restored.initialize.metadata["files"][0]["checksum"] = "changed"


@pytest.mark.parametrize(
    "record",
    [
        Model(
            "external/regressor@1",
            config={"width": 8},
            initialize=Weights.random(seed=123),
            trainable=["projection.weight"],
        ),
        Model.from_pretrained(
            "org/model",
            revision="exact-revision",
            lora=LoRA(8),
            trainable=["*.lora_*", "head.weight"],
        ),
        Objective("clipped_policy", config={"reference_penalty": "k3"}),
        Optimizer("adamw", defaults={"lr": 0.001}, groups={"head": ["head.weight"]}),
        BatchConfig(limits={"max_examples": 64}, typical={"height": 224}),
        Memory(device_gib=1, spill_gib=0),
        Memory(device_gib=0, spill_gib=0),
        Execution(),
        Execution(reject_overbudget=True),
        Operation("logprobs", mode="training"),
        Weights.artifact("checkpoint_123"),
    ],
)
def test_json_roundtrip_preserves_typed_nested_configuration(record):
    encoded = json.loads(json.dumps(record.to_dict(), allow_nan=False))
    restored = type(record).from_dict(encoded)
    assert restored == record
    if isinstance(restored, Model):
        assert restored.initialize is None or isinstance(restored.initialize, Weights)
        assert restored.lora is None or isinstance(restored.lora, LoRA)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), lambda: 0])
def test_nonportable_configuration_is_refused(value):
    with pytest.raises((TypeError, ValueError)):
        Model("external/model@1", config={"nested": [value]})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Model(),
        lambda: LoRA(True),
        lambda: LoRA(0),
        lambda: LoRA(1, alpha=float("nan")),
        lambda: LoRA(1, dropout=1),
        lambda: Weights("random", seed=2, reference="other"),
        lambda: Weights("random", seed=2, metadata={"files": []}),
        lambda: Weights.random(seed=-1),
        lambda: Weights("artifact", reference="artifact", revision="v1"),
        lambda: BatchConfig(limits={}),
        lambda: BatchConfig(limits={"max_examples": True}),
        lambda: BatchConfig(limits={"max_examples": -1}),
        lambda: Memory(float("inf"), 1),
        lambda: Memory(10**1000, 1),
        lambda: Memory(-1, 1),
        lambda: Memory(1, -0.5),
        lambda: Execution(max_pending_requests=0),
        lambda: Execution(idle_timeout_seconds=0),
        lambda: Execution(reject_overbudget=1),
        lambda: Execution(reject_overbudget="false"),
        lambda: Operation("logprobs", mode="other"),
    ],
)
def test_invalid_settings_fail_before_preparation(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_unknown_record_fields_are_not_silently_ignored():
    with pytest.raises(ValueError, match=r"unknown fields.*nmae"):
        Model.from_dict({"name": "external/model@1", "nmae": "typo"})


def test_omitted_budget_enforcement_matches_the_executor_reporting_default():
    assert Execution().reject_overbudget is False
    assert Execution.from_dict({"seed": 7}).reject_overbudget is False


def test_capacity_names_are_not_reserved_for_text():
    batch = BatchConfig(limits={"max_examples": 16, "max_duration_seconds": 2.5})
    assert set(batch.limits) == {"max_examples", "max_duration_seconds"}
    assert Execution().idle_timeout_seconds is None


def test_datum_keeps_payloads_owned_by_caller_until_submission():
    pixels = [[1, 2], [3, 4]]
    data = {"pixels": pixels}
    datum = Datum(inputs=data, loss_inputs={"class_id": 7})
    data["extra"] = 3
    assert "extra" not in datum.inputs
    assert datum.inputs["pixels"] is pixels
    pixels[0][0] = 9
    assert datum.inputs["pixels"][0][0] == 9


def test_client_records_do_not_import_engine_or_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'shadowspill', 'torch', 'shadowspill_workloads',
                                  'mlops', 'transformers', 'triton'}:
            raise AssertionError('unexpected dependency: '+name)
sys.meta_path.insert(0, Block())
from shadowspill_client import Model, LoRA, BatchConfig, Objective
from shadowspill_client.protocol.codec import encode, decode
model = Model.from_pretrained('org/model', lora=LoRA(8), trainable=['head.weight'])
assert decode(encode(model)) == model
assert model.initialize.reference == 'org/model'
assert model.trainable == ('head.weight',)
assert BatchConfig(limits={'max_examples': 4})
assert Objective('external/classification@1')
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
