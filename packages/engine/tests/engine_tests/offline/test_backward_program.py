"""Backward-only and optimizer-only paths share one saved producer contract."""

import subprocess
import sys
from dataclasses import replace

import pytest

from engine_tests.offline.test_example_instances import captured
from shadowspill import (
    Program,
    backward_program,
    instantiate_examples,
    optimizer_program,
)
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.ir import MemoryLocation, Persistence
from shadowspill.planning.training.bind import bind_training


@pytest.mark.parametrize("reduction", [False, True])
def test_derivation_keeps_all_backward_forms_and_producer_gradients(
    reduction, tmp_path
):
    source = captured(reduction=reduction)
    result = backward_program(source)
    optimizer = optimizer_program(source)
    assert result.bindings["kind"] == "forward_backward"
    assert (
        result.bindings["persistent_gradients"]
        == optimizer.bindings["persistent_gradients"]
    )
    assert result.bindings["operation_derivation"] == {
        "kind": "forward_backward",
        "source_program": source.digest,
    }
    assert (
        result.bindings["training_templates"] == source.bindings["training_templates"]
    )
    assert result.bindings["input_signatures"] == source.bindings["input_signatures"]
    assert not result.bindings["optimizer_templates"]
    assert not any(
        t.phase in ("optimizer", "gradient_communication") for t in result.tasks
    )
    assert "optimizer_bindings" not in result.bindings
    assert "master" not in {obj.object_id for obj in result.objects}
    assert "moment" not in {obj.object_id for obj in result.objects}
    gradient = next(obj for obj in result.objects if obj.object_id == "gradient")
    assert gradient.persistence is Persistence.RUN
    assert any(
        r.alias_group_id == gradient.alias_group_id
        and r.location is MemoryLocation.SPILL
        for r in result.final_residency
    )
    assert {tensor.identity.role for tensor in program_state(result).tensors} == {
        "parameter",
        "gradient",
    }
    assert (
        program_state(result).shared_aliases(program_state(optimizer))["gradient"]
        == "gradient"
    )
    original_profiles = {
        d.profile_key
        for d in source.task_definitions
        if d.profile_key.startswith(("input", "backward"))
    }
    assert {d.profile_key for d in result.task_definitions} == original_profiles | {
        "invocation_start"
    }
    # Both first-write and additive forms remain available after pruning.
    fresh = instantiate_examples(result, [0])
    accumulated = bind_training(fresh, accumulate=True)
    assert {
        task.profile_id for task in accumulated.tasks if task.phase == "backward"
    } == {"backward0.True"}
    assert any(r.alias_group_id == "gradient" for r in accumulated.initial_residency)
    result.save(tmp_path / "backward.ssp")
    assert Program.load(tmp_path / "backward.ssp") == result


def test_nonstep_or_missing_templates_fail_without_preparation():
    source = captured()
    for bindings in (
        {**source.bindings, "kind": "forward_backward"},
        {**source.bindings, "training_templates": []},
        {**source.bindings, "optimizer_templates": []},
    ):
        with pytest.raises(ValueError, match="complete-step templates"):
            backward_program(replace(source, bindings=bindings))


def test_derivation_does_not_import_framework_or_client(tmp_path):
    captured().save(tmp_path / "source.ssp")
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_client', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill import (
    Program, backward_program, optimizer_program, instantiate_examples,
)
source = Program.load('source.ssp')
backward = backward_program(source)
optimizer = optimizer_program(source)
assert (backward.bindings['persistent_gradients']
        == optimizer.bindings['persistent_gradients'])
repeated = instantiate_examples(backward, [0, 0, 0])
repeated.save('backward.ssp')
assert repeated == Program.load('backward.ssp')
""",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
