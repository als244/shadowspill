"""Host outputs stay off the device pool in every saved training form."""

from dataclasses import replace

import pytest

from engine_tests.offline._example import program
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.build.output_devices import resolve_output_devices
from shadowspill.errors import CompilationError
from shadowspill.ir import (
    AliasGroupSpec,
    ObjectSpec,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)


def fixture():
    aliases = tuple(AliasGroupSpec(name, "rank0", 8) for name in ("fresh", "add"))

    def slot(name):
        return {"leaf_index": 0, "object_id": name}

    def form(name):
        return {
            "forward": {
                "profile_key": "produce",
                "inputs": [],
                "outputs": [slot(name)],
            },
            "backward": {
                "profile_key": name + "_consume",
                "inputs": [slot(name)],
                "contributions": [],
            },
        }

    return replace(
        program(),
        objects=(
            ObjectSpec("fresh", "fresh", 0, 8),
            ObjectSpec("add", "add", 0, 8),
            ObjectSpec("add_view", "add", 4, 4),
        ),
        alias_groups=aliases,
        task_definitions=(TaskDefinition("produce", "produce"),),
        tasks=(
            TaskSpec(
                "selected", ResourceSpec("rank0", ResourceKind.COMPUTE), "produce"
            ),
        ),
        initial_residency=(),
        final_residency=(),
        sources={},
        bindings={
            "entrypoints": [
                {
                    "task_id": "selected",
                    "input_slots": [],
                    "output_slots": [slot("fresh")],
                }
            ],
            "training_templates": [[[form("fresh"), form("add")]]],
            "compiler_memory": {"produce": {"off_device_output_leaves": []}},
            "task_inputs": {
                name + "_consume": {
                    "tensor_positions": [0],
                    "tensor_geometry": [{"device_type": "cuda"}],
                }
                for name in ("fresh", "add")
            },
        },
    )


def test_host_resolution_covers_unselected_forms_consumers_and_alias_views():
    original = fixture()
    resolved = resolve_output_devices(original, {"produce": ("cpu",)})
    assert all(obj.size_bytes == obj.offset_bytes == 0 for obj in resolved.objects)
    assert all(alias.size_bytes == 0 for alias in resolved.alias_groups)
    for inputs in resolved.bindings["task_inputs"].values():
        assert inputs["tensor_geometry"][0]["device_type"] == "cpu"
    assert original.objects[1].size_bytes == 8
    assert (
        original.bindings["task_inputs"]["add_consume"]["tensor_geometry"][0][
            "device_type"
        ]
        == "cuda"
    )


def test_unselected_form_cannot_conflict_with_selected_output_device():
    from shadowspill.artifacts._values import plain

    original = fixture()
    bindings = plain(original.bindings)
    alternate = bindings["training_templates"][0][0][1]["forward"]
    alternate.update(
        profile_key="alternate", outputs=[{"leaf_index": 0, "object_id": "fresh"}]
    )
    bindings["compiler_memory"]["alternate"] = {"off_device_output_leaves": []}
    with pytest.raises(CompilationError, match="conflicting output devices"):
        resolve_output_devices(
            replace(original, bindings=bindings),
            {"produce": ("cpu",), "alternate": ("cuda",)},
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradient_copy_devices_use_the_complete_entrypoint_slots(device):
    from shadowspill.artifacts._values import plain

    original = fixture()
    bindings = plain(original.bindings)
    bindings["training_templates"][0][0] = bindings["training_templates"][0][0][:1]
    bindings["gradient_templates"] = [
        {
            "source": "fresh",
            "padding": {"profile_key": "copy", "output": "add"},
        }
    ]
    bindings["compiler_memory"]["copy"] = {"off_device_output_leaves": []}
    bindings["entrypoints"].append(
        {
            "task_id": "gradient_copy",
            "input_slots": [{"leaf_index": 0, "object_id": "fresh"}],
            "output_slots": [{"leaf_index": 0, "object_id": "add"}],
        }
    )
    bindings["task_inputs"]["copy"] = {
        "tensor_positions": [0],
        "tensor_geometry": [{"device_type": "cuda"}],
    }
    original = replace(
        original,
        bindings=bindings,
        task_definitions=(*original.task_definitions, TaskDefinition("copy", "copy")),
        tasks=(
            *original.tasks,
            TaskSpec(
                "gradient_copy",
                original.tasks[0].resource,
                "copy",
                dependencies=("selected",),
                inputs=("fresh",),
                outputs=("add",),
            ),
        ),
    )
    resolved = resolve_output_devices(
        original, {"produce": (device,), "copy": (device,)}
    )
    assert resolved.objects[1].size_bytes == (0 if device == "cpu" else 8)
    assert (
        resolved.bindings["task_inputs"]["copy"]["tensor_geometry"][0]["device_type"]
        == device
    )


@pytest.mark.cuda
def test_replicated_dp_step_finalizes_gradient_copy_kernels():
    import torch
    from torch import nn

    from shadowspill import Parallelism, build_step, measure_program

    if not torch.cuda.is_available():
        pytest.skip("requires one representative GPU")
    model = nn.Linear(7, 5, dtype=torch.bfloat16, device="meta")
    program = build_step(
        model,
        optimizer=torch.optim.AdamW(model.parameters(), foreach=False),
        objective=lambda model, x: model(x).float().square().mean(),
        example_args=(torch.zeros(2, 7, dtype=torch.bfloat16),),
        partition="whole",
        grad_dtype=torch.float32,
        master_dtype=torch.float32,
        parallelism=Parallelism.dp(
            ranks_per_replica=2,
            weights="replicated",
            gradients="replicated",
            optimizer="replicated",
        ),
    )
    copies = [row["padding"] for row in program.bindings["gradient_templates"]]
    assert copies and all(copy is not None for copy in copies)
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    for copy in copies:
        assert copy["profile_key"] in profiles.tasks
        assert not program.bindings["compiler_memory"][copy["profile_key"]][
            "off_device_output_leaves"
        ]
