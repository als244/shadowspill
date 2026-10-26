"""Host controls have zero GPU extent but survive until their final consumer."""

import pytest
import torch
from torch import nn

from shadowspill import build_forward, measure_program
from tests.shadowspill.execution._host_controls import host_control, read_host_control
from tests.shadowspill.execution.helpers import run_worker, save_plan


class Produce(nn.Module):
    def forward(self, x):
        # Cover both compiler-visible CPU allocations and opaque CPU results.
        fixed = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cpu")
        return x.sin(), host_control(x) + fixed


class Consume(nn.Module):
    def forward(self, x, control):
        return read_host_control(x, control)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((8,), 0.5))
        self.produce = Produce()
        self.first, self.second = Consume(), Consume()

    def forward(self, x):
        x, control = self.produce(x * self.weight)
        return control, self.second(self.first(x, control), control)


@pytest.mark.cuda
def test_host_outputs_across_tasks_and_cold_execution(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model, x = Network(), torch.randn(4, 8)
    with torch.device("meta"):
        template = Network()
    program = build_forward(template, example_args=(x,))
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    tasks = {task.task_id: task for task in program.tasks}
    objects = {obj.object_id: obj for obj in program.objects}
    observed = 0
    for entry in program.bindings["entrypoints"]:
        record = profiles.tasks[tasks[entry["task_id"]].profile_id]
        host = record.metadata["off_device_output_leaves"]
        for slot in entry["output_slots"]:
            if slot["leaf_index"] in host:
                assert objects[slot["object_id"]].size_bytes == 0
                observed += 1
    assert observed > 0
    # A stage-only profile still resolves its controls through earlier producers.
    subset = measure_program(
        program,
        stages=[len(program.bindings["entrypoints"]) - 1],
        warmup_seconds=0.001,
        measure_seconds=0.002,
    )
    assert subset.tasks.keys() <= profiles.tasks.keys()
    save_plan(program, tmp_path, profiles=profiles)
    expected = model(x)
    state = {"weight": model.weight.detach().clone()}
    torch.save(
        dict(
            initial=state,
            examples=[((x,), {})],
            expected=[(expected, state)] * 3,
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
