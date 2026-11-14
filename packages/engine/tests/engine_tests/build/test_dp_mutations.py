"""A distributed replacement must allocate the complete padded storage root."""

import pytest
import torch
from torch import nn

from shadowspill import Parallelism, build_step, measure_program
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.task_inputs import prepare_inputs


class MutableBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(5, 5)
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("running", torch.zeros(5))

    def forward(self, value):
        self.count.add_(1)
        self.running.add_(2)
        return self.layer(value)


@pytest.mark.cuda
def test_sharded_mutations_allocate_padding_before_profiling(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires one representative GPU")

    def forbidden(*args, **kwargs):
        raise AssertionError("local build/profiling must not create process groups")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    with torch.device("meta"):
        model = MutableBuffers()
    program = build_step(
        model,
        optimizer=torch.optim.AdamW(model.parameters(), foreach=False),
        objective=lambda model, x: model(x).square().mean(),
        example_args=(torch.zeros(2, 5),),
        num_examples=2,
        partition="whole",
        parallelism=Parallelism.dp(ranks_per_replica=2),
    )
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    assert profiles.tasks
    checked = set()
    for job in stage_jobs(program, None):
        if "forward" not in job:
            continue
        key = job["forward"]["profile_key"]
        function = restore_executable(
            program.executables[key],
            dependencies=program.bindings["operator_dependencies"].get(key, ()),
        )
        arguments = prepare_inputs(program, key, torch.device("cuda:0"))
        outputs = function(*arguments)
        for slot in job["forward"]["outputs"]:
            obj = next(o for o in program.objects if o.object_id == slot["object_id"])
            names = program.sources.get(obj.alias_group_id, ())
            for name, increment, elements in (("count", 1, 2), ("running", 2, 6)):
                if name not in names:
                    continue
                source = next(
                    s
                    for s in job["forward"]["inputs"]
                    if s["object_id"] == obj.object_id
                )
                value = outputs[slot["leaf_index"]]
                torch.testing.assert_close(
                    value, arguments[source["leaf_index"]] + increment
                )
                assert (
                    value.untyped_storage().nbytes() == elements * value.element_size()
                )
                checked.add(name)
    assert checked == {"count", "running"}
