"""The launcher's resolution of a case and the runner's reading of it: what
a trainer starts from, the tokens it trains on, and the row it ends with."""

import json

import numpy
import pytest
import torch

from baselines.common import runner
from baselines.common.launch import resolve, trainer_environment
from baselines.common.results import validate_row
from baselines.fsdp2 import settings as fsdp2
from baselines.fsdp2.settings import Settings

TINY = {
    "model": "llama3-tiny",
    "sequence_length": 64,
    "tokens_per_step": 512,
    "tokens_per_microbatch": 128,
    "warmup_steps": 1,
    "steps": 2,
}


def write_case(tmp_path, world_size=2, **changes):
    settings = Settings.from_dict({**TINY, **changes})
    case = resolve("fsdp2", settings, world_size)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    return settings, case, path


def open_rank(monkeypatch, tmp_path, path, *, rank=0, world=2, engine="fsdp2"):
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    arguments = runner.parse(engine, ["--output", str(tmp_path), "--case", str(path)])
    return runner.open_case(engine, fsdp2, arguments, standalone=True)


def test_a_resolved_case_describes_and_counts_what_the_trainer_runs(tmp_path):
    settings, case, _ = write_case(tmp_path)
    assert case["engine"] == "fsdp2" and case["size"] == "tiny"
    assert case["settings"] == settings.as_dict()
    assert case["description"] == settings.describe(2)
    assert case["useful_flops_per_step"] > 0
    summary = case["model_summary"]
    assert summary["model"] == "llama3-tiny" and summary["parameters"] > 0
    assert summary["trainable_parameters"] == summary["parameters"]


def test_the_runner_opens_the_case_the_launcher_resolved(monkeypatch, tmp_path):
    settings, case, path = write_case(tmp_path)
    opened = open_rank(monkeypatch, tmp_path, path, rank=1)
    assert opened.settings == settings
    assert opened.geometry == settings.geometry(2)
    assert opened.description == case["description"]
    assert opened.directory == tmp_path / "rank-00001"
    record = opened.record
    assert record["rank"] == 1 and record["world_size"] == 2
    assert record["summary"] == case["model_summary"]
    assert record["microbatches_per_step"] == opened.geometry.accumulation == 2
    assert record["shadowspill"] is False


def test_a_case_runs_only_where_it_was_resolved(monkeypatch, tmp_path):
    _, _, path = write_case(tmp_path)
    with pytest.raises(ValueError, match="another world"):
        open_rank(monkeypatch, tmp_path, path, world=4)
    with pytest.raises(ValueError, match="not a deepspeed case"):
        open_rank(monkeypatch, tmp_path, path, engine="deepspeed")


def test_a_case_runs_only_in_the_environment_it_sets(monkeypatch, tmp_path):
    _, case, path = write_case(tmp_path)
    case["environment"] = {"BASELINES_CASE_SWITCH": "1", "BASELINES_CASE_UNSET": None}
    path.write_text(json.dumps(case))
    monkeypatch.delenv("BASELINES_CASE_SWITCH", raising=False)
    monkeypatch.setenv("BASELINES_CASE_UNSET", "x")
    with pytest.raises(ValueError, match="environment is not its case's"):
        open_rank(monkeypatch, tmp_path, path)
    environment = trainer_environment(case["environment"])
    assert environment["BASELINES_CASE_SWITCH"] == "1"
    assert "BASELINES_CASE_UNSET" not in environment
    monkeypatch.setenv("BASELINES_CASE_SWITCH", "1")
    monkeypatch.delenv("BASELINES_CASE_UNSET")
    opened = open_rank(monkeypatch, tmp_path, path)
    assert opened.record["environment"] == case["environment"]


def test_synthetic_tokens_are_one_draw_per_replica(monkeypatch, tmp_path):
    _, _, path = write_case(tmp_path)
    opened = open_rank(monkeypatch, tmp_path, path)
    data = opened.data(512)
    tokens, targets = data(0)
    count = opened.geometry.tokens_per_rank_per_step
    assert tokens.shape == targets.shape == (1, count)
    again, _ = data(5)
    assert again is tokens
    other, _ = opened.data(512, replica=1)(0)
    assert not torch.equal(tokens, other)
    assert "corpus" not in opened.record


def test_corpus_tokens_are_the_replicas_window_of_each_step(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    numpy.arange(4096, dtype=numpy.uint32).tofile(corpus / "train.bin")
    (corpus / "meta.json").write_text(
        json.dumps({"tokenizer": "test", "vocab_size": 512, "dtype": "uint32"})
    )
    _, _, path = write_case(tmp_path, data=str(corpus))
    opened = open_rank(monkeypatch, tmp_path, path, rank=1)
    data = opened.data(512)
    count = opened.geometry.tokens_per_rank_per_step
    tokens, targets = data(0)
    assert int(tokens[0, 0]) == count and int(targets[0, 0]) == count + 1
    tokens, _ = data(1)
    assert int(tokens[0, 0]) == 3 * count
    assert opened.record["corpus"]["vocab_size"] == 512


class StoppedClock(runner.Clock):
    """A clock whose steps were measured elsewhere: nothing touches a device."""

    def __init__(self, case, cycles, losses):
        super().__init__(case)
        self.cycles, self.losses, self.head_losses = cycles, losses, losses
        self.setup_seconds = 1.0
        self.closed = True


def test_the_first_rank_writes_the_results_and_the_row(monkeypatch, tmp_path):
    _, case, path = write_case(tmp_path, world_size=1)
    opened = open_rank(monkeypatch, tmp_path, path, world=1)
    opened.record["hardware"] = {"host": "test"}
    # The test process holds ShadowSpill's modules, which a standalone run
    # refuses.
    opened.standalone = False
    opened.directory.mkdir()
    outcome = runner.Outcome(
        peaks={"peak_allocated_bytes": 7, "peak_reserved_bytes": 9},
        provenance={"extra": "kept"},
        result={"extra_seconds": 2.0},
    )
    clock = StoppedClock(opened, [0.5, 0.25], [3.0, 2.5])
    runner.finish(opened, clock, outcome, fsdp2.SOURCE_PATHS)
    result = json.loads((opened.directory / "result.json").read_text())
    assert result["cycles"] == [0.5, 0.25] and result["extra_seconds"] == 2.0
    assert result["peak_allocated_bytes"] == 7
    assert result["performance_result"] is False
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["useful_flops_per_step"] == case["useful_flops_per_step"]
    assert results["losses"] == [[3.0, 2.5]]
    row = json.loads((tmp_path / "row.json").read_text())
    validate_row(row)
    assert row["engine"] == "fsdp2" and row["status"] == "completed"
    for key in ("model", "geometry", "precision", "configuration"):
        assert row[key] == case["description"][key]
    assert row["measurement"]["step_seconds"] == pytest.approx(0.375)
    assert row["memory"]["peak_reserved_bytes"] == 9
    assert row["provenance"]["extra"] == "kept"
    assert row["provenance"]["model_summary"] == case["model_summary"]
    assert row["provenance"]["environment"]["packages"]["torch"]
