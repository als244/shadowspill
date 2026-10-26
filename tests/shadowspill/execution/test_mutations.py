"""Functionalized buffer replacements retain their full physical allocation."""

import copy

import pytest
import torch
from torch import nn

from shadowspill import build_forward, measure_program
from tests.shadowspill.execution.helpers import run_worker, save_plan


class Stateful(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running", torch.zeros(64))

    def forward(self, value):
        self.running.add_(value.sum(0))
        return self.running[8:56] * 0.5


@pytest.mark.cuda
def test_saved_buffer_replacement_preserves_state_and_workspace(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(317)
    model, value = Stateful(), torch.randn(4, 64)
    initial = copy.deepcopy(model.state_dict())
    program = build_forward(
        copy.deepcopy(model).to("meta"), example_args=(value,), partition="whole"
    )
    (task,) = [task for task in program.tasks if task.requires_entrypoint]
    assert len(task.mutations) == 1
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    record = profiles.tasks[task.profile_id]
    storage = program.bindings["manifests"][task.profile_id]["storage_contract"]
    (mutation,) = storage["mutations"]
    leaf = mutation["replacement_output_leaf"]
    assert leaf is not None
    root = next(
        view["root_id"]
        for view in storage["output_views"]
        if view["leaf_index"] == leaf
    )
    layout = record.metadata["compiled_layout"]
    assert layout["roots"][root]["requested_bytes"] == model.running.nbytes
    assert record.metadata["allocation_contract"]
    plan = save_plan(program, tmp_path, profiles=profiles)
    costed = plan._costed_program
    compiled = next(item for item in costed.tasks if item.task_id == task.task_id)
    workspace = next(
        p.workspace_bytes
        for p in costed.profiles
        if p.profile_id == compiled.profile_id
    )
    assert workspace == record.workspace_bytes + model.running.nbytes
    expected = []
    for _ in range(3):
        output = model(value).clone()
        expected.append((output, copy.deepcopy(model.state_dict())))
    torch.save(
        dict(
            initial=initial,
            examples=[((value,), {})],
            expected=expected,
            queued_outputs=[model(value).clone() for _ in range(2)],
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
