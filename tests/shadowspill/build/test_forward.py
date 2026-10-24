import os
import subprocess
import sys
import textwrap
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch.utils._pytree import tree_flatten, tree_unflatten, treespec_loads

from shadowspill import Program
from shadowspill.build import build_forward


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    def forward(self, x, *, options):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return {"result": x * options["scale"], "note": None}


def replay_for_test(program, model, args, kwargs):
    """Numerical oracle for task bindings; this is not the runtime Executor."""
    objects = {
        registration["object_id"]: model.get_parameter(registration["name"])
        for registration in program.bindings["registrations"]
    }
    leaves, _ = tree_flatten((args, kwargs))
    sources = program.bindings["root_sources"]
    for slot in program.bindings["root_inputs"]:
        source = sources[slot["leaf_index"]]
        if source["kind"] == "USER_INPUT":
            objects[slot["object_id"]] = leaves[source["user_leaf"]]
    profile_by_task = {task.task_id: task.profile_id for task in program.tasks}
    for entry in program.bindings["entrypoints"]:
        arguments = [objects[slot["object_id"]] for slot in entry["input_slots"]]
        function = AOTCompiledArtifact.deserialize(
            program.executables[profile_by_task[entry["task_id"]]]
        )
        outputs, _ = tree_flatten(function(*arguments))
        for slot in entry["output_slots"]:
            objects[slot["object_id"]] = outputs[slot["leaf_index"]]
    output_values = [
        None if object_id is None else objects[object_id]
        for object_id in program.bindings["public_outputs"]
    ]
    for index, value in program.bindings["output_literals"]:
        output_values[index] = value
    return tree_unflatten(
        output_values, treespec_loads(program.bindings["output_tree"])
    )


@pytest.mark.cuda
def test_meta_build_roundtrips_and_executes_bound_stages(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("finalizing CUDA task kernels needs a representative GPU")
    with torch.device("meta"):
        model = Network()
    inputs = torch.empty(4, 8)
    program = build_forward(
        model, example_args=(inputs,), example_kwargs={"options": {"scale": 2}}
    )
    assert all(parameter.is_meta for parameter in model.parameters())
    assert len(program.bindings["entrypoints"]) == 2
    assert any("layers.0" in task.module_paths for task in program.task_definitions)
    assert all(task.workspace_bytes is None for task in program.task_definitions[1:])
    program.save(tmp_path / "program.ssp")
    loaded = Program.load(tmp_path / "program.ssp")
    assert loaded.to_dict() == program.to_dict()
    assert loaded.executables == program.executables
    with pytest.raises(TypeError):
        loaded.executables["other"] = b"changed"
    actual_model, x = Network().cuda(), torch.randn(4, 8, device="cuda")
    with torch.no_grad():
        actual = replay_for_test(loaded, actual_model, (x,), {"options": {"scale": 2}})
        expected = actual_model(x, options={"scale": 2})
    torch.testing.assert_close(actual, expected)
    script = textwrap.dedent("""
        import importlib, sys, torch, triton
        from shadowspill import Program
        from tests.shadowspill.build.test_forward import Network, replay_for_test
        def forbidden(*args, **kwargs):
            raise AssertionError("restored task attempted compilation")
        module = importlib.import_module("torch._inductor.compile_fx")
        module.compile_fx = module.compile_fx_inner = forbidden
        triton.compile = forbidden
        model, x = Network().cuda(), torch.randn(4, 8, device="cuda")
        with torch.no_grad():
            program = Program.load(sys.argv[1])
            result = replay_for_test(program, model, (x,), {"options": {"scale": 2}})
            torch.testing.assert_close(result, model(x, options={"scale": 2}))
        print("fresh cache replay correct")
    """)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(tmp_path / "program.ssp")],
        env={
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
            "TRITON_CACHE_DIR": str(tmp_path / "triton"),
        },
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fresh cache replay correct" in result.stdout


def test_archive_preserves_opaque_code_without_loading_it(tmp_path):
    from tests.shadowspill.offline._example import program

    value = replace(program(), executables={"task": b"opaque executable bytes"})
    value.save(tmp_path / "program.ssp")
    assert Program.load(tmp_path / "program.ssp").executables == value.executables
