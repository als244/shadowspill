"""Optimizer-only Programs preserve captured tasks and original gradient storage."""

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.offline.test_state_inventory import training
from shadowspill import Program
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.ir import AliasGroupSpec, ObjectSpec
from shadowspill.planning.training.operations import optimizer_program


def step_program(*, reduction=False):
    source = training()
    gradient = "gradient"
    extra_aliases, extra_objects, reductions = (), (), []
    if reduction:
        gradient = "reduced"
        extra_aliases = (AliasGroupSpec("reduced", "rank0", 64),)
        extra_objects = (ObjectSpec("reduced", "reduced", 0, 64),)
        reductions = [
            {
                "name": "linear.weight",
                "source": "gradient",
                "stored": "reduced",
                "padding": None,
                "phases": [{"profile_key": "reduce", "operation": "reduce_scatter"}],
            }
        ]
    component = {
        "task_id": "update",
        "profile_key": "optimizer",
        "inputs": [
            {"leaf_index": i, "object_id": obj}
            for i, obj in enumerate(("master", "moment", gradient))
        ],
        "mutations": ["master", "moment"],
        "host_inputs": [],
        "replacements": [],
        "reductions": reductions,
    }
    return replace(
        source,
        alias_groups=(*source.alias_groups, *extra_aliases),
        objects=(*source.objects, *extra_objects),
        executables={"optimizer": b"already compiled", "unused": b"not needed"},
        task_definitions=tuple(
            TaskDefinition(key, key, 0)
            for key in ("invocation_start", "optimizer", "reduce", "unused")
        ),
        bindings={
            **source.bindings,
            "kind": "step",
            "optimizer_templates": [component],
            "training_templates": ["not inspected during derivation"],
            "gradients": source.bindings["persistent_gradients"],
            "task_inputs": {
                "optimizer": {
                    "tensor_positions": [0, 1, 2],
                    "tensor_geometry": [
                        {"shape": [4, 4], "stride": [4, 1], "dtype": "torch.float32"}
                    ]
                    * 3,
                },
                "reduce": {
                    "tensor_positions": [0],
                    "tensor_geometry": [
                        {"shape": [16], "stride": [1], "dtype": "torch.float32"}
                    ],
                },
            },
        },
    )


@pytest.mark.parametrize("reduction", [False, True])
def test_derive_reuses_update_and_reduction_dependencies_without_forward(
    reduction, tmp_path
):
    source = step_program(reduction=reduction)
    result = optimizer_program(source)
    assert result.bindings["kind"] == "optim_step"
    assert result.bindings["num_examples"] == 0
    assert "training_templates" not in result.bindings
    assert result.executables == {"optimizer": b"already compiled"}
    assert (
        result.bindings["optimizer_templates"] == source.bindings["optimizer_templates"]
    )
    phases = [t.phase for t in result.tasks]
    assert phases == (
        ["control", "gradient_communication", "optimizer"]
        if reduction
        else ["control", "optimizer"]
    )
    if reduction:
        assert result.tasks[1].task_id in result.tasks[2].dependencies
        assert "reduced" not in {r.alias_group_id for r in result.initial_residency}
        assert "reduced" not in {r.alias_group_id for r in result.final_residency}
    assert "gradient" in {r.alias_group_id for r in result.initial_residency}
    assert "gradient" in {r.alias_group_id for r in result.final_residency}
    assert any(t.identity.role == "gradient" for t in program_state(result).tensors)
    assert list(stage_jobs(result, None)) == [
        {"optimizer": result.bindings["optimizer_templates"][0]}
    ]
    result.save(tmp_path / "optimizer.ssp")
    assert Program.load(tmp_path / "optimizer.ssp") == result


def test_forward_or_missing_optimizer_cannot_be_mislabeled_as_an_update():
    source = step_program()
    for bindings in (
        {**source.bindings, "kind": "forward"},
        {**source.bindings, "optimizer_templates": []},
    ):
        with pytest.raises(ValueError, match="captured optimizer"):
            optimizer_program(replace(source, bindings=bindings))


def test_saved_derivation_imports_no_framework_or_sdk(tmp_path):
    step_program().save(tmp_path / "step.ssp")
    code = """
import sys
for module in ('torch', 'shadowspill_client', 'shadowspill_workloads', 'mlops'):
    sys.modules[module] = None
from shadowspill import Program, optimizer_program
step = Program.load('step.ssp')
result = optimizer_program(step)
result.save('optimizer.ssp')
assert Program.load('optimizer.ssp') == result
assert result.executables == {'optimizer': b'already compiled'}
"""
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_optimizer_call_rejects_unexpected_examples_before_accessing_storage():
    from shadowspill.execution.inputs import stage_inputs

    state = SimpleNamespace(
        plan=SimpleNamespace(program=SimpleNamespace(bindings={"num_examples": 0}))
    )
    stage_inputs(state, None, None, None)
    for args, kwargs, loader in (
        ([()], None, None),
        (None, [{}], None),
        (None, None, lambda _: ()),
    ):
        with pytest.raises(ValueError, match="accepts no examples"):
            stage_inputs(state, args, kwargs, loader)
