"""CPU output envelopes must match both saved compiler views and declared outputs."""

import io
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from shadowspill.artifacts.tensors import public_tensor_geometry
from shadowspill.execution.results import TrainingResult
from shadowspill.training.worker.output_frame import WorkerOutputFrame
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.definitions import OutputSet, TensorOutput
from shadowspill_client.protocol.payload_io import read_payload


def description(*, forward=False, dtype="float32", count=2):
    """Synthetic metadata only; the GPU test checks actual compiler artifacts."""
    contract = SimpleNamespace(
        outputs=OutputSet(
            metrics={"count": "sum"},
            tensors={"prediction": TensorOutput(dtype, (3,), "datum")},
        )
    )
    bindings = [
        SimpleNamespace(
            buffer=4 + index * 3,
            contract=contract,
            tensor_shapes={"prediction": (2, 3)},
        )
        for index in range(count)
    ]
    groups, views, slots = [], [], []
    for index in range(count):
        group = []
        for name, shape, kind in (
            ("loss", (), "float32"),
            ("count", (), "int64"),
            ("prediction", (2, 3), dtype),
        ):
            if forward and name == "loss":
                group.append(None)
                continue
            identity, leaf = f"result.{index}.{name}", len(views)
            group.append(identity)
            slots.append({"object_id": identity, "leaf_index": leaf})
            views.append(
                {
                    "leaf_index": leaf,
                    "dtype": "torch." + kind,
                    "shape": shape,
                    "layout": "torch.strided",
                }
            )
        groups.append(group)
    program = SimpleNamespace(
        tasks=(SimpleNamespace(task_id="task", profile_id="compiled"),),
        bindings={
            "kind": "forward" if forward else "step",
            "forward_templates": (),
            "public_outputs": groups,
            "entrypoints": [{"task_id": "task", "output_slots": slots}],
            "manifests": {"compiled": {"storage_contract": {"output_views": views}}},
        },
    )
    return program, bindings


def result(*, forward=False, dtype=torch.float32, count=2):
    groups = tuple(
        (
            None if forward else torch.tensor(float(index)),
            {
                "prediction": torch.arange(6).reshape(3, 2).T.to(dtype),
                "count": torch.tensor(index + 2),
            },
        )
        for index in range(count)
    )
    return (
        groups
        if forward
        else TrainingResult(
            tuple(loss for loss, _ in groups), tuple(values for _, values in groups)
        )
    )


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_exact_saved_output_frame_size_and_chunked_roundtrip(forward, dtype):
    program, bindings = description(forward=forward, dtype=dtype)
    frame = WorkerOutputFrame(program, bindings)
    assert frame.buffers == (4, 7)
    value = result(forward=forward, dtype=getattr(torch, dtype))
    fields = frame.fields(value)
    output = io.BytesIO()
    budget = BufferBudget(frame.frame.write_bytes)
    ref = frame.write(value, output, budget, "result")
    assert ref.size_bytes == frame.frame.size.wire == len(output.getvalue())
    assert budget.used == (0, 0)
    output.seek(0)
    with read_payload(
        output, ref, BufferBudget(frame.frame.size.staging), limits=frame.frame.size
    ) as decoded:
        for path, tensor in fields.items():
            leaf = decoded.datums[0].inputs
            for key in path[1:]:
                leaf = leaf[key]
            assert leaf.dtype == str(tensor.dtype).removeprefix("torch.")
            assert (
                leaf.data
                == tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )


def test_optimizer_only_empty_output_frame_is_still_exact_and_decodable():
    program, bindings = description(count=0)
    frame = WorkerOutputFrame(program, bindings)
    assert not frame.frame.shapes and not frame.frame.size.tensors
    stream = io.BytesIO()
    reference = frame.write(
        result(count=0), stream, BufferBudget(frame.frame.write_bytes), "empty"
    )
    assert reference.size_bytes == frame.frame.size.wire


@pytest.mark.parametrize(
    "failure", ["leaf", "dtype", "shape", "metric", "producer", "ungrouped"]
)
def test_compiler_outputs_must_match_declared_abi_before_any_execution(failure):
    program, bindings = description()
    saved = program.bindings
    views = saved["manifests"]["compiled"]["storage_contract"]["output_views"]
    if failure == "leaf":
        saved["public_outputs"][0].append(None)
    elif failure == "dtype":
        views[2]["dtype"] = "torch.int32"
    elif failure == "shape":
        views[2]["shape"] = (3, 2)
    elif failure == "metric":
        views[1]["shape"] = (1,)
    elif failure == "producer":
        saved["entrypoints"][0]["output_slots"].pop()
    else:
        saved["kind"] = "forward"
        del saved["forward_templates"]
    with pytest.raises(ValueError):
        WorkerOutputFrame(program, bindings)


def test_conflicting_writer_geometry_and_non_strided_outputs_fail():
    program, _ = description()
    second = deepcopy(program.bindings["manifests"]["compiled"])
    second["storage_contract"]["output_views"][0]["shape"] = (1,)
    program.tasks += (SimpleNamespace(task_id="other", profile_id="other"),)
    program.bindings["manifests"]["other"] = second
    program.bindings["entrypoints"].append(
        {
            "task_id": "other",
            "output_slots": [{"object_id": "result.0.loss", "leaf_index": 0}],
        }
    )
    with pytest.raises(ValueError, match="disagree"):
        public_tensor_geometry(program)
    second["storage_contract"]["output_views"][0]["layout"] = "torch.sparse_coo"
    with pytest.raises(ValueError, match="strided"):
        public_tensor_geometry(program)


@pytest.mark.parametrize(
    "failure", ["count", "names", "shape", "dtype", "loss", "parameters"]
)
def test_actual_result_cannot_change_the_prepared_output_contract(failure):
    program, bindings = description(forward=failure == "loss")
    frame = WorkerOutputFrame(program, bindings)
    value = result(forward=failure == "loss")
    if failure == "count":
        value = result(count=1)
    elif failure == "names":
        value.metrics[0]["extra"] = torch.tensor(0)
    elif failure == "shape":
        value.metrics[0]["prediction"] = torch.zeros(3, 2)
    elif failure == "dtype":
        value.metrics[0]["prediction"] = torch.zeros(2, 3, dtype=torch.int64)
    elif failure == "loss":
        value = ((torch.tensor(0), value[0][1]), value[1])
    else:
        value.parameter_metrics["unexpected"] = torch.tensor(0)
    with pytest.raises(ValueError):
        frame.fields(value)


def test_output_layout_import_stays_framework_and_catalog_free():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'torch', 'triton', 'shadowspill_workloads'}:
   raise AssertionError(name)
sys.meta_path.insert(0, Block())
from shadowspill.training.worker.output_frame import WorkerOutputFrame
from shadowspill.artifacts.tensors import public_tensor_geometry
"""
    process = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr
