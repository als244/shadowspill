"""Legacy startup diagnostics remain covered until that behavior is migrated."""

import torch
from torch import nn

from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import PyTorch


def objective(model, batch):
    error = model(batch["features"]) - batch["target"]
    return error.square().mean(), {"error_sum": error.detach().sum()}


def data(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "features": torch.randn(5, 3, generator=generator),
        "target": torch.randn(5, 2, generator=generator),
    }


class Source:
    def __init__(self):
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        result = data(self.position)
        self.position += 1
        return result

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def test_startup_diagnostics_reuse_first_data_without_advancing_loop(tmp_path):
    source = Source()
    calls, logs, schedule_calls = [], [], []

    def schedule(step):
        schedule_calls.append(step)
        return 0.01

    with Trainer(
        nn.Linear(3, 2),
        objective=objective,
        optimizer=torch.optim.SGD,
        schedules={"lr": schedule},
        backend=PyTorch(compile=False, device="cpu"),
    ) as trainer:
        trainer.prepare(data())

        def diagnose(batches, directory, *, warmup):
            calls.append((batches, directory, warmup, trainer.step_count))
            torch.rand(5)
            return {"traced_seconds": 0.1}

        trainer._execution.diagnose = diagnose
        before_rng = torch.get_rng_state()
        trainer.fit(
            source,
            steps=2,
            run_dir=tmp_path,
            logger=logs.append,
            startup_diagnostics=True,
        )
        assert source.position == trainer.step_count == 2
        assert schedule_calls == [0, 1]
        assert [item["step"] for item in logs] == [1, 2]
        assert torch.equal(torch.get_rng_state(), before_rng)
    assert len(calls) == 1
    batches, directory, warmup, step = calls[0]
    assert directory == tmp_path / "startup" and warmup == 1 and step == 0
    torch.testing.assert_close(batches[0][0]["features"], data()["features"])
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
