"""Native DP transfers must not be mistaken for captured coefficient consumers."""

from types import SimpleNamespace

import pytest

from engine_tests.fixtures.definition_operations import definitions
from shadowspill.build.lowering.catalog import CompiledTasks
from shadowspill.build.lowering.collectives import declare_collective
from shadowspill.training.preparation.capture_checks import validate_optimizer_program


def capture(operation):
    compiled, collectives = CompiledTasks({}, {}), {}
    key = declare_collective(
        {
            "operation": operation,
            "groups": [[0, 1]],
            "tensor_bytes": 8,
            "dtype": "torch.float32",
        },
        compiled,
        collectives,
    )
    assert "provenance" not in compiled.inputs[key]
    controls = ["optimizer_group.0.lr", "gradient_scale"]
    return SimpleNamespace(
        objects=[SimpleNamespace(object_id="weights", alias_group_id="weights")],
        tasks=[
            SimpleNamespace(task_id="transfer", profile_id=key),
            SimpleNamespace(task_id="optimizer", profile_id="compiled"),
        ],
        bindings={
            "collective_tasks": collectives,
            "task_inputs": {
                **compiled.inputs,
                "compiled": {
                    "provenance": [
                        {"consumer_targets": ["aten.mul.Tensor"]},
                        {"consumer_targets": ["aten.mul.Tensor"]},
                    ]
                },
            },
            "entrypoints": [
                {
                    "task_id": "transfer",
                    "input_slots": [{"leaf_index": 0, "object_id": "weights"}],
                },
                {
                    "task_id": "optimizer",
                    "host_inputs": [
                        {"leaf_index": index, "name": name}
                        for index, name in enumerate(controls)
                    ],
                },
            ],
            "gradient_scale_control": "gradient_scale",
            "optimizer_bindings": [
                {
                    "name": controls[0],
                    "role": "hyperparameter",
                    "mutable": False,
                    "shape": [],
                    "dtype": "torch.float32",
                    "object_id": None,
                }
            ],
        },
    )


@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter", "all_reduce"])
def test_native_collectives_keep_compiled_optimizer_control_checks(operation):
    _, _, optimizer = definitions()
    program = capture(operation)
    validate_optimizer_program(program, optimizer)
    for index, message in ((0, "not consumed"), (1, "final gradient scale")):
        provenance = program.bindings["task_inputs"]["compiled"]["provenance"]
        provenance[index]["consumer_targets"] = []
        with pytest.raises(ValueError, match=message):
            validate_optimizer_program(program, optimizer)
        provenance[index]["consumer_targets"] = ["aten.mul.Tensor"]
    del program.bindings["task_inputs"]["compiled"]["provenance"]
    with pytest.raises(KeyError, match="provenance"):
        validate_optimizer_program(program, optimizer)
