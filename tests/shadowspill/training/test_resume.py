"""Checkpoint progress preserves source position, schedules and RNG state."""

import json
import random

import numpy as np
import pytest
import torch

from shadowspill import Trainer
from shadowspill.training import _state


class Schedule:
    def __init__(self):
        self.scale = torch.tensor(0.2)

    def __call__(self, step):
        return self.scale / (step + 1)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, values):
        self.scale = values["scale"]


class Source:
    def __init__(self):
        self.position = 0

    def __iter__(self):
        while self.position < 3:
            self.position += 1
            yield {"args": [(self.position,)]}

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, value):
        self.position = value["position"]


def test_resume_restores_source_schedule_rng_and_checks_before_mutation(
    executor, tmp_path
):
    schedule, source = Schedule(), Source()
    trainer = Trainer(executor.plan, executor=executor, schedules={"lr": schedule})
    trainer.fit(source, steps=1, log_every=0)
    trainer.save(tmp_path / "resume.ssp", chunk_bytes=20)
    expected = random.random(), torch.rand(3), np.random.rand(3)
    executor.values["step"].zero_()
    schedule.scale.zero_()
    resumed = Trainer(executor.plan, executor=executor, schedules={"lr": Schedule()})
    resumed.load(tmp_path / "resume.ssp", chunk_bytes=28)
    assert resumed.step_count == 1 and executor.values["step"].item() == 3
    assert resumed.schedules["lr"].scale.item() == pytest.approx(0.2)
    assert random.random() == expected[0]
    torch.testing.assert_close(torch.rand(3), expected[1])
    np.testing.assert_equal(np.random.rand(3), expected[2])
    with pytest.raises(ValueError, match="resumable source"):
        resumed.fit([], steps=3)
    fresh_source = Source()
    resumed.fit(fresh_source, steps=3, log_every=0)
    assert fresh_source.position == 3
    assert [call["args"][0][0] for call in executor.calls] == [1, 2, 3]
    bad = Trainer(executor.plan, executor=executor)
    executor.values["step"].fill_(99)
    with pytest.raises(ValueError, match="schedules differ"):
        bad.load(tmp_path / "resume.ssp")
    assert executor.values["step"].item() == 99


def test_loop_state_codec_preserves_types_without_pickle():
    value = {
        1: (torch.ones(3, dtype=torch.bfloat16), np.arange(4, dtype=np.uint32)),
        "empty": torch.empty(0, 3),
        "nested": [False, None, {2: "text"}],
    }
    decoded = _state.decode(json.loads(json.dumps(_state.encode(value))))
    torch.testing.assert_close(decoded[1][0], value[1][0])
    np.testing.assert_equal(decoded[1][1], value[1][1])
    assert decoded["nested"] == value["nested"]
    assert decoded["empty"].shape == (0, 3)
    with pytest.raises(TypeError, match="object arrays"):
        _state.encode(np.array([{}], dtype=object))
