"""Saved count variants clone private storage and retain one mutable inventory."""

import subprocess
import sys
from dataclasses import replace

import pytest

from engine_tests.offline.test_optimizer_program import step_program
from shadowspill import Program, instantiate_examples
from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.ir import (
    AliasGroupSpec,
    MemoryLocation,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)
from shadowspill.planning.training.bind import bind_training


def captured(*, reduction=False):
    source = step_program(reduction=reduction)
    aliases, objects, templates, roots, outputs, fixed = [], [], [], [], [], []
    definitions = list(source.task_definitions)
    task_inputs = plain(source.bindings["task_inputs"])
    for index in range(2):
        for role in ("input", "activation", "constant"):
            name = f"{role}{index}"
            aliases.append(AliasGroupSpec(name, "rank0", 64 * (index + 1)))
            objects.append(ObjectSpec(name, name, 0, 64 * (index + 1)))
        # The view must follow its whole alias closure when its example repeats.
        objects.append(ObjectSpec(f"view{index}", f"activation{index}", 0, 16))
        stages = []
        for accumulates in (False, True):
            inputs = [f"activation{index}", "linear.weight"]
            if accumulates:
                inputs.append("gradient")
            backward = f"backward{index}.{accumulates}"
            definitions.append(TaskDefinition(backward, backward, 0))
            stages.append(
                {
                    "variant": "saved",
                    "accumulates": accumulates,
                    "saved_aliases": [f"activation{index}"],
                    "forward": {
                        # Deliberately equal an object ID: identity remapping must
                        # not mistake a compiled profile key for storage.
                        "profile_key": f"input{index}",
                        "inputs": [
                            {"leaf_index": i, "object_id": name}
                            for i, name in enumerate(
                                (f"input{index}", f"constant{index}", "linear.weight")
                            )
                        ],
                        "outputs": [
                            {"leaf_index": 0, "object_id": f"activation{index}"}
                        ],
                        "mutations": [],
                        "replacements": [],
                        "handoffs": [
                            {
                                "leaf_index": 0,
                                "source_object_id": f"input{index}",
                                "destination_object_id": f"activation{index}",
                            }
                        ],
                        "public_outputs": [0],
                    },
                    "backward": {
                        "profile_key": backward,
                        "inputs": [
                            {"leaf_index": i, "object_id": name}
                            for i, name in enumerate(inputs)
                        ],
                        "contributions": [{"leaf_index": 0, "object_id": "gradient"}],
                        "handoffs": [],
                        "replacements": [],
                        "discarded": [],
                    },
                }
            )
        definitions.append(TaskDefinition(f"input{index}", f"forward{index}", 0))
        for form in stages:
            for phase in ("forward", "backward"):
                slots = form[phase]["inputs"]
                task_inputs[form[phase]["profile_key"]] = {
                    "tensor_positions": [slot["leaf_index"] for slot in slots],
                    "tensor_geometry": [
                        {
                            "shape": [4, 4],
                            "stride": [4, 1],
                            "dtype": "torch.float32",
                        }
                        for slot in slots
                    ],
                }
        templates.append([stages])
        roots.append([{"leaf_index": 0, "object_id": f"input{index}"}])
        outputs.append([f"view{index}", None])
        fixed.append({"object_id": f"constant{index}", "value": index + 1})
    signatures = [{"shape": [index + 1, 4]} for index in range(2)]
    root_sources = [[{"name": f"x{index}"}] for index in range(2)]
    result = replace(
        source,
        alias_groups=(*source.alias_groups, *aliases),
        objects=(*source.objects, *objects),
        tasks=(
            TaskSpec(
                "invocation_start",
                ResourceSpec("rank0", ResourceKind.CONTROL),
                "invocation_start",
                phase="control",
                requires_entrypoint=False,
            ),
        ),
        task_definitions=tuple(definitions),
        initial_residency=tuple(
            ResidencySpec(alias.alias_group_id, MemoryLocation.SPILL)
            for alias in (*source.alias_groups, *aliases)
            if alias.alias_group_id != "gradient"
            and not alias.alias_group_id.startswith("activation")
        ),
        final_residency=tuple(
            ResidencySpec(alias.alias_group_id, MemoryLocation.DEVICE)
            for alias in aliases
            if alias.alias_group_id.startswith("activation")
        ),
        bindings={
            **source.bindings,
            "num_examples": 2,
            "task_inputs": task_inputs,
            "training_templates": templates,
            "root_inputs": roots,
            "public_outputs": outputs,
            "fixed_inputs": fixed,
            "input_signature": signatures[0],
            "input_signatures": signatures,
            "root_sources": root_sources[0],
            "example_root_sources": root_sources,
            "ordering_constraints": ["mutable model buffer"],
        },
    )
    return bind_training(result)


@pytest.mark.parametrize("indices", [(0,), (0, 0, 0), (1, 0, 1)])
def test_private_storage_repeats_while_state_and_compiled_forms_stay_shared(
    indices, tmp_path
):
    source = captured()
    result = instantiate_examples(source, indices)
    assert program_state(result) == program_state(source)
    assert result.executables == source.executables
    assert result.task_definitions == source.task_definitions
    assert result.bindings["num_examples"] == len(indices)
    assert result.bindings["ordering_constraints"] == ("mutable model buffer",)
    assert result.bindings["optimizer_ordering"] == "stage_interleaved"
    assert result.bindings["example_instances"]["source_program"] == source.digest
    assert result.bindings["example_instances"]["template_indices"] == indices
    object_ids = {obj.object_id for obj in result.objects}
    objects = {obj.object_id: obj for obj in result.objects}
    assert not {f"input{i}" for i in range(2)} & object_ids
    assert result == instantiate_examples(source, indices)
    for position, index in enumerate(indices):
        root = result.bindings["root_inputs"][position][0]["object_id"]
        view, absent = result.bindings["public_outputs"][position]
        fixed = result.bindings["fixed_inputs"][position]
        assert absent is None and fixed["value"] == index + 1
        assert root.startswith("object_") and root.removeprefix("object_").isdigit()
        assert len({root, view, fixed["object_id"]}) == 3
        assert (
            result.bindings["input_signatures"][position]
            == source.bindings["input_signatures"][index]
        )
        form = result.bindings["training_templates"][position][0][0]
        assert form["forward"]["profile_key"] == f"input{index}"
        handoff = form["forward"]["handoffs"][0]
        assert handoff["source_object_id"] == root
        activation = form["forward"]["outputs"][0]["object_id"]
        assert handoff["destination_object_id"] == activation
        assert objects[activation].alias_group_id == objects[view].alias_group_id
        assert form["saved_aliases"] == (objects[view].alias_group_id,)
        assert result.sources[objects[root].alias_group_id] == (
            f"example[{position}].x{index}",
        )
    backwards = [t for t in result.tasks if t.phase == "backward"]
    assert backwards[0].outputs == ("gradient",)
    assert all(t.mutations[0].object_id == "gradient" for t in backwards[1:])
    optimizer = next(t for t in result.tasks if t.phase == "optimizer")
    assert {t.task_id for t in backwards} <= set(optimizer.dependencies)
    result.save(tmp_path / "count.ssp")
    assert Program.load(tmp_path / "count.ssp") == result
    repeated = instantiate_examples(result, [0, 0])
    assert program_state(repeated) == program_state(source)
    assert len({slot[0]["object_id"] for slot in repeated.bindings["root_inputs"]}) == 2


@pytest.mark.parametrize("indices", [[], [-1], [2], [True], [0.0], ["0"]])
def test_invalid_count_selection_rejected(indices):
    with pytest.raises(ValueError, match="template_indices"):
        instantiate_examples(captured(), indices)


def test_missing_additive_forms_rejected_without_preparation():
    source = captured()
    bindings = plain(source.bindings)
    bindings["training_templates"] = [
        [[form for form in stage if not form["accumulates"]] for stage in example]
        for example in bindings["training_templates"]
    ]
    source = replace(source, bindings=bindings)
    instantiate_examples(source, [0])
    with pytest.raises(ValueError, match="required gradient form"):
        instantiate_examples(source, [0, 0])
    with pytest.raises(ValueError, match="saved forward bindings"):
        instantiate_examples(replace(source, bindings={"kind": "forward"}), [0])


def test_collective_descriptions_survive_rebinding_and_wait_for_every_contribution():
    source = captured(reduction=True)
    source = replace(
        source,
        bindings={
            **source.bindings,
            "collective_tasks": {
                "reduce": {
                    "operation": "reduce_scatter",
                    "groups": [[0]],
                    "tensor_bytes": 64,
                    "dtype": "torch.float32",
                }
            },
        },
    )
    result = instantiate_examples(source, [1, 0, 1])
    assert result.bindings["collective_tasks"] == source.bindings["collective_tasks"]
    reduction = next(t for t in result.tasks if t.phase == "gradient_communication")
    assert {t.task_id for t in result.tasks if t.phase == "backward"} <= set(
        reduction.dependencies
    )
    update = next(t for t in result.tasks if t.phase == "optimizer")
    assert reduction.task_id in update.dependencies


def test_instantiation_roundtrip_without_framework_or_client(tmp_path):
    captured().save(tmp_path / "source.ssp")
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'shadowspill_client.service',
               'shadowspill_client.training', 'shadowspill_client.transport',
               'shadowspill_workloads', 'mlops'):
    sys.modules[name] = None
from shadowspill import Program, instantiate_examples
source = Program.load('source.ssp')
result = instantiate_examples(source, [1, 0, 1])
result.save('count.ssp')
assert Program.load('count.ssp') == result
assert result.executables == source.executables
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
