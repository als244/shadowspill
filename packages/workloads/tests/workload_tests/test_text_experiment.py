"""Packed text keeps exact resume and observations across saved phase workers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from shadowspill import Trainer
from shadowspill.pytorch.guards import capture_input_signature
from shadowspill.training import config
from shadowspill_workloads.recipes.text.experiment.inputs import (
    call_inputs,
    candidates,
    source,
)
from shadowspill_workloads.recipes.text.heterogeneous.checkpoint_paths import (
    checkpoint_directory,
    resume_path,
)
from tests.integration.training._text_recipe import _config


def test_packed_normalizer_is_a_runtime_tensor():
    parts = [[torch.ones(1, 8, dtype=torch.int64)] * 3] * 2
    first, second = [
        call_inputs({"candidates": {"batch": parts}, "normalizer": n}, "batch")
        for n in (7, 13)
    ]
    for calls, denominator in ((first, 7), (second, 13)):
        assert len(calls["args"]) == 2
        assert all(
            args[1].item() == pytest.approx(1 / denominator) for args in calls["args"]
        )
        assert all(args[1].shape == () for args in calls["args"])


def test_build_examples_match_actual_packed_layouts(tmp_path):
    options = json.loads(_config(tmp_path).read_text())
    data = config.resolve(options["data"])
    example, _, geometries, _ = candidates(options, data)
    for name, (tokens, count) in geometries.items():
        expected = capture_input_signature(call_inputs(example, name)["args"][0])
        actual = next(
            source(
                data,
                options,
                {
                    "candidate": name,
                    "tokens_per_microbatch": tokens,
                    "microbatches": count,
                },
            )
        )
        for args in actual["args"]:
            expected.validate(args)


def test_budgets_left_out_come_from_the_profiled_hardware(tmp_path):
    import math

    from shadowspill_workloads.recipes.text.experiment import plan, profile

    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       263000000 kB\nMemFree:        1000 kB\n"
        "MemAvailable:   200000000 kB\n"
    )
    host = profile.host_memory(meminfo)
    assert host == {
        "host_total_bytes": 263000000 << 10,
        "host_available_bytes": 200000000 << 10,
    }
    hardware = {"device": "card", "device_total_bytes": 34_000_000_000, **host}
    assert plan.budgets({"execution_gib": 4, "spill_gib": 0.5}, hardware) == (4, 0.5)
    assert plan.budgets({"execution_gib": 4, "spill_gib": 0.5}, None) == (4, 0.5)
    assert plan.budgets({"spill_gib": 8}, hardware) == (31 - 2, 8)
    assert plan.budgets({}, hardware) == (
        29,
        math.floor((200000000 << 10) * 0.75 / (1 << 30)),
    )
    with pytest.raises(ValueError, match=r"hardware\.json"):
        plan.budgets({}, None)
    with pytest.raises(ValueError, match="positive"):
        plan.budgets({"execution_gib": 0, "spill_gib": 1}, None)


def test_the_calibration_a_candidate_plans_with_follows_the_rank_count(tmp_path):
    from shadowspill_workloads.recipes.text.experiment.plan import communication_path

    assert communication_path(tmp_path, 1, "6144_tokens", "training") == (
        tmp_path / "communication.ssp"
    )
    assert communication_path(tmp_path, 2, "6144_tokens", "evaluation") == (
        tmp_path / "communication/6144_tokens/evaluation.ssp"
    )


@pytest.mark.parametrize(
    "ranks,named,expected",
    [
        (1, False, ["run/checkpoints"]),
        (1, True, ["saves"]),
        (2, False, ["run/rank-00000/checkpoints", "run/rank-00001/checkpoints"]),
        (2, True, ["saves/rank-00000", "saves/rank-00001"]),
    ],
)
def test_each_rank_saves_prunes_and_resumes_a_step_in_its_own_directory(
    tmp_path, monkeypatch, ranks, named, expected
):
    def step(self, **inputs):
        self.step_count += 1
        return SimpleNamespace(step=self.step_count)

    def save(self, path):
        # The archive names the rank that wrote it, so one rank's save
        # replacing another's would show.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(self.rank))

    monkeypatch.setattr(Trainer, "step", step)
    monkeypatch.setattr(Trainer, "save", save)
    options = {"ranks": ranks}
    if named:
        options["checkpoint_dir"] = str(tmp_path / "saves")
    for rank in range(ranks):
        home = tmp_path / "run" / f"rank-{rank:05d}" if ranks > 1 else tmp_path / "run"
        directory = checkpoint_directory(options, home, rank)
        assert directory == tmp_path / expected[rank]
        trainer = object.__new__(Trainer)
        trainer.rank, trainer.step_count = rank, 0
        trainer._source_state, trainer._closed = None, False
        trainer.fit(
            [{}] * 3,
            steps=3,
            run_dir=home,
            log_every=0,
            tables_every=0,
            checkpoint_every=1,
            checkpoint_dir=directory,
            keep_last=2,
        )
    for rank, name in enumerate(expected):
        directory = tmp_path / name
        # Retention prunes the rank's own directory alone.
        assert sorted(path.name for path in directory.iterdir()) == [
            "step_00000002.ssp",
            "step_00000003.ssp",
        ]
        resumed = resume_path("step_00000003", directory)
        assert resumed == directory / "step_00000003.ssp"
        assert resumed.read_text() == str(rank)
    given = Path("runs/text/checkpoints/step_00000003.ssp")
    assert resume_path(str(given), tmp_path / expected[0]) == given
