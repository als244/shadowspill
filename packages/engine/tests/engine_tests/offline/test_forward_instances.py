"""Forward counts are saved task repetitions with private inputs and results."""

import subprocess
import sys
from dataclasses import replace

import pytest

from shadowspill import Program, instantiate_examples
from shadowspill.artifacts.examples import input_groups, output_groups, single_forward
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    MutationSpec,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)


def captured_forward():
    names = ("weight", "counter", "input", "constant", "activation")
    geometry = {"shape": [4], "stride": [1], "dtype": "torch.float32"}
    return Program(
        devices=(DeviceSpec("rank0", "worker", "cuda", 0),),
        alias_groups=tuple(AliasGroupSpec(name, "rank0", 16) for name in names),
        objects=(
            *tuple(ObjectSpec(name, name, 0, 16) for name in names),
            ObjectSpec("view", "activation", 0, 8),
        ),
        tasks=(
            TaskSpec(
                "start",
                ResourceSpec("rank0", ResourceKind.CONTROL),
                "invocation_start",
                requires_entrypoint=False,
                phase="control",
            ),
            TaskSpec(
                "forward",
                ResourceSpec("rank0", ResourceKind.COMPUTE),
                "input",
                dependencies=("start",),
                inputs=names[:-1],
                outputs=("activation", "view"),
                mutations=(MutationSpec("counter"),),
                phase="forward",
            ),
        ),
        task_definitions=(
            TaskDefinition("invocation_start", "start", 0),
            TaskDefinition("input", "compiled", 0),
        ),
        executables={"input": b"compiled forward"},
        initial_residency=tuple(
            ResidencySpec(name, MemoryLocation.SPILL) for name in names[:-1]
        ),
        final_residency=tuple(
            ResidencySpec(
                name,
                MemoryLocation.SPILL if name != "activation" else MemoryLocation.DEVICE,
            )
            for name in names
        ),
        sources={name: [name] for name in names},
        bindings={
            "kind": "forward",
            "registrations": [
                {"name": name, "object_id": name, "parameter": name == "weight"}
                for name in names[:2]
            ],
            "model_tensor_geometry": {name: geometry for name in names[:2]},
            "root_inputs": [
                {"leaf_index": i, "object_id": name}
                for i, name in enumerate(names[2:4])
            ],
            "root_sources": [
                {"kind": "USER_INPUT", "name": "x", "user_leaf": 0},
                {"kind": "CONSTANT_TENSOR", "name": "bias", "value": [1, 2, 3, 4]},
            ],
            "input_signature": {"tree": "input tree", "leaves": []},
            "public_outputs": ["activation", "view", None, None],
            "output_literals": [(2, "kept"), (3, None)],
            "output_tree": "output tree",
            "task_inputs": {
                "input": {
                    "tensor_positions": [0, 1, 2, 3],
                    "tensor_geometry": [geometry] * 4,
                }
            },
            "entrypoints": [
                {
                    "task_id": "forward",
                    "input_slots": [
                        {"leaf_index": i, "object_id": name}
                        for i, name in enumerate(names[:-1])
                    ],
                    "output_slots": [
                        {"leaf_index": 0, "object_id": "activation"},
                        {"leaf_index": 1, "object_id": "view"},
                    ],
                    "options": {"stage_index": 0},
                    "storage_handoffs": [
                        {
                            "source_object_id": "input",
                            "destination_object_id": "activation",
                        }
                    ],
                }
            ],
            "transient_outputs": {"forward": [2]},
            "fixed_inputs": [{"object_id": "constant", "fill": 1.0}],
        },
    )


@pytest.mark.parametrize("count", [1, 3])
def test_repeat_forward_preserves_state_profiles_alias_views_and_order(count, tmp_path):
    source = captured_forward()
    assert single_forward(source.bindings)
    result = instantiate_examples(source, [0] * count)
    assert not single_forward(result.bindings)
    assert program_state(result) == program_state(source)
    assert result.task_definitions == source.task_definitions
    assert result.executables == source.executables
    assert result.bindings["output_literals"] == source.bindings["output_literals"]
    assert (
        len(input_groups(result.bindings))
        == len(output_groups(result.bindings))
        == count
    )
    objects = {item.object_id: item for item in result.objects}
    roots = result.bindings["root_inputs"]
    assert len({slot[0]["object_id"] for slot in roots}) == count
    for index, group in enumerate(result.bindings["forward_templates"]):
        control, compute = result.tasks[2 * index : 2 * index + 2]
        assert tuple(task.task_id for task in (control, compute)) == group
        assert control.dependencies == (
            () if index == 0 else (result.tasks[2 * index - 1].task_id,)
        )
        assert compute.dependencies == (control.task_id,)
        assert compute.profile_id == "input"
        assert compute.inputs[:2] == ("weight", "counter")
        assert compute.mutations == (MutationSpec("counter"),)
        output, view, absent, literal = result.bindings["public_outputs"][index]
        assert absent is literal is None
        assert objects[output].alias_group_id == objects[view].alias_group_id
        assert output.startswith("object_") and output.removeprefix("object_").isdigit()
        assert result.bindings["transient_outputs"][compute.task_id] == (2,)
        entry = result.bindings["entrypoints"][index]
        assert entry["options"]["repetition"] == index
        assert (
            entry["storage_handoffs"][0]["source_object_id"]
            == roots[index][0]["object_id"]
        )
        assert entry["storage_handoffs"][0]["destination_object_id"] == output
    result.save(tmp_path / "forward.ssp")
    assert Program.load(tmp_path / "forward.ssp") == result
    repeated = instantiate_examples(result, [count - 1, 0, count - 1])
    assert repeated.bindings["num_examples"] == 3
    assert program_state(repeated) == program_state(source)
    assert len(repeated.objects) == 14


@pytest.mark.parametrize("indices", [[], [-1], [1], [True], [0.0]])
def test_invalid_forward_selection_fails(indices):
    with pytest.raises(ValueError, match="template_indices"):
        instantiate_examples(captured_forward(), indices)


def test_rejects_borrowed_state_outputs_and_malformed_groups():
    source = captured_forward()
    with pytest.raises(ValueError, match="separate from model state"):
        instantiate_examples(
            replace(
                source, bindings={**source.bindings, "public_outputs": ["counter"]}
            ),
            [0],
        )
    with pytest.raises(ValueError, match="example groups"):
        instantiate_examples(
            replace(
                source, bindings={**source.bindings, "forward_templates": [["start"]]}
            ),
            [0],
        )


def test_forward_count_instantiation_does_not_import_frameworks(tmp_path):
    captured_forward().save(tmp_path / "source.ssp")
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for module in ('torch', 'triton', 'mlops', 'shadowspill_client.service',
               'shadowspill_client.training',
               'shadowspill_client.transport',
               'shadowspill_workloads'):
    sys.modules[module] = None
from shadowspill import Program, instantiate_examples
source = Program.load('source.ssp')
count = instantiate_examples(source, [0, 0, 0])
result = instantiate_examples(count, [2, 0, 2])
result.save('count.ssp')
assert result == Program.load('count.ssp')
assert result.executables == source.executables
""",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
