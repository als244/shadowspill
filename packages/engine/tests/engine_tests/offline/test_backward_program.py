"""Backward-only and optimizer-only paths share one saved producer contract."""

import os
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
from shadowspill.artifacts._values import plain
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.artifacts.tensors import tensor_geometry
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
for name in ('torch', 'triton', 'mlops', 'shadowspill_client.service',
               'shadowspill_client.training', 'shadowspill_client.transport',
               'shadowspill_workloads'):
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


def test_saved_backward_identity_is_independent_of_python_hash_seed(tmp_path):
    source = captured()
    gradient = next(o for o in source.objects if o.object_id == "gradient")
    alias = next(
        a for a in source.alias_groups if a.alias_group_id == gradient.alias_group_id
    )
    bindings = plain(source.bindings)
    bindings["gradients"].append(
        {
            **bindings["gradients"][0],
            "parameter_name": "linear.second",
            "gradient_object_id": "gradient.second",
        }
    )
    bindings["gradient_geometry"] = {
        "gradient.second": tensor_geometry(source)["gradient"]
    }
    for example in bindings["training_templates"]:
        for stage in example:
            for form in stage:
                form["backward"]["contributions"].append(
                    {"leaf_index": 1, "object_id": "gradient.second"}
                )
                if form["accumulates"]:
                    inputs = form["backward"]["inputs"]
                    inputs.append(
                        {"leaf_index": len(inputs), "object_id": "gradient.second"}
                    )
                    profile = bindings["task_inputs"][form["backward"]["profile_key"]]
                    profile["tensor_positions"].append(len(inputs) - 1)
                    profile["tensor_geometry"].append(
                        tensor_geometry(source)["gradient"]
                    )
    source = replace(
        source,
        bindings=bindings,
        objects=(
            *source.objects,
            replace(
                gradient, object_id="gradient.second", alias_group_id="gradient.second"
            ),
        ),
        alias_groups=(
            *source.alias_groups,
            replace(alias, alias_group_id="gradient.second"),
        ),
    )
    path = tmp_path / "source.ssp"
    source.save(path)
    identities = []
    for seed in (0, 1, 2, 3):
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from shadowspill import Program, backward_program, instantiate_examples
backward = backward_program(Program.load(sys.argv[1]))
print(backward.digest, instantiate_examples(backward, [0, 0]).digest)
""",
                str(path),
            ],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        identities.append(run.stdout)
    assert len(set(identities)) == 1
