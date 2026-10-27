"""Reject incomparable configurations and partial or failed distributed
results, for any engine that exposes its settings."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pytest

from baselines.common import write_json
from baselines.common.launch import command, engine_module, trainer_command
from baselines.common.provenance import ROOT
from baselines.common.results import (
    memory_peaks,
    read_rows,
    row,
    summarize_cycles,
    varied,
)
from baselines.engines.fsdp2.settings import Settings, add_arguments, from_namespace
from baselines.sweep import config as sweep
from baselines.sweep.config import comparable, expand, keep_names
from baselines.sweep.process import run_case
from baselines.sweep.reporting import (
    duration,
    ended_block,
    labels,
    started_block,
    tally,
)
from baselines.sweep.results import classify, failure_row, latest, to_attempt
from baselines.sweep.run import main

#: A tiny case and the eight-GPU Qwen3-30B-A3B pretraining case.
TINY = {
    "model": "llama3-tiny",
    "sequence_length": 64,
    "tokens_per_step": 2048,
    "tokens_per_microbatch": 64,
}
BIG = {"model": "qwen3moe-30b-a3b", "tokens_per_step": 2097152}


def test_grid_keeps_global_step_constant_and_roundtrips_every_control():
    cases = expand(
        {
            "schema_version": 1,
            "defaults": {
                "model": "qwen3moe-30b-a3b",
                "tokens_per_step": 2097152,
                "forward_prefetch": 2,
                "backward_prefetch": "2",
            },
            "grid": {
                "tokens_per_microbatch": [4096, 131072],
                "offload_policy": ["none", "cpu"],
                "gradient_sync": ["per-microbatch", "last-microbatch"],
            },
        },
        8,
        Settings,
    )
    assert len(cases) == 8
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    for _, settings in cases:
        accumulation = settings.describe(8)["geometry"]["accumulation"]
        assert settings.tokens_per_microbatch * accumulation * 8 == 2097152
        assert from_namespace(parser.parse_args(settings.arguments())) == settings


@pytest.mark.parametrize(
    "override",
    [
        {"tokens_per_step": 2097152 + 1},
        {"tokens_per_microbatch": 5000},
        {"backward_prefetch": 0},
        {"forward_prefetch": -1},
        {"offload_policy": "typo"},
        {"recompute": float("nan")},
        {"pin_memory": "false"},
        {"cpu_threads": 0},
        {"allocator_cap_gib": float("inf")},
        {"verify_reference": True},
    ],
)
def test_invalid_controls_are_rejected_before_launch(override):
    with pytest.raises(ValueError):
        expand({"schema_version": 1, "defaults": BIG, "cases": [override]}, 8, Settings)


def test_a_list_of_grids_runs_their_union():
    specification = {
        "schema_version": 1,
        "defaults": TINY,
        "grid": [
            {"recompute": [0], "activation_offload": ["none"]},
            {"recompute": [0.5, 1], "activation_offload": ["none", "cpu"]},
        ],
    }
    cases = [settings for _, settings in expand(specification, 1, Settings)]
    assert [(case.recompute, case.activation_offload) for case in cases] == [
        (0, "none"),
        (0.5, "none"),
        (0.5, "cpu"),
        (1, "none"),
        (1, "cpu"),
    ]
    with pytest.raises(ValueError, match="nonempty"):
        expand({**specification, "grid": []}, 1, Settings)


def test_duplicate_effective_cases_run_once():
    cases = [{}, {"tokens_per_step": 2097152}]
    specification = {"schema_version": 1, "defaults": BIG, "cases": cases}
    assert len(expand(specification, 8, Settings)) == 1


def test_unknown_knob_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown"):
        expand(
            {"schema_version": 1, "defaults": BIG, "grid": {"prefetch_typo": [1]}},
            8,
            Settings,
        )


def test_a_engine_is_its_settings_module_launched_through_the_launcher():
    module = engine_module("fsdp2")
    assert module.Settings is Settings
    line = command("fsdp2", "py", 2, Settings(**TINY), "out")
    if sys.flags.no_site:
        assert line.pop(1) == "-S"
    assert line[:11] == [
        sys.executable,
        "-m",
        "baselines.common.launch",
        "--engine",
        "fsdp2",
        "--python",
        "py",
        "--gpus",
        "2",
        "--output",
        "out",
    ]
    assert line[11:] == Settings(**TINY).arguments()
    assert trainer_command("fsdp2", "py", 2, Path("out")) == [
        "py",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--max-restarts=0",
        "-m",
        "baselines.engines.fsdp2.train",
        "--output",
        "out",
        "--case",
        "out/case.json",
    ]
    with pytest.raises(ValueError, match="no engine"):
        engine_module("nonesuch")


def test_settings_describe_the_model_geometry_precision_and_configuration():
    settings = Settings(**TINY)
    description = settings.describe(2)
    assert description["model"]["model"] == "llama3-tiny"
    assert description["geometry"]["accumulation"] == 16
    assert description["precision"]["master_dtype"] == "float32"
    assert "recompute" in description["configuration"]
    assert "tokens_per_step" not in description["configuration"]


def completed_attempt(attempt, settings, world_size=2, status="completed"):
    description = settings.describe(world_size)
    measurement = summarize_cycles(
        [[1, 10], [10, 1]], global_tokens=settings.tokens_per_step
    )
    measurement["performance_result"] = True
    peaks = {
        "peak_allocated_bytes": 123,
        "peak_reserved_bytes": 456,
        "host_peak_rss_bytes": 789,
    }
    value = row(
        engine="fsdp2",
        status=status,
        **description,
        hardware={"host": "test"},
        provenance={},
        measurement=measurement if status == "completed" else None,
        memory=memory_peaks([peaks, peaks]) if status == "completed" else None,
        failure=None if status == "completed" else {"reason": "x"},
    )
    write_json(attempt / "row.json", value)


def test_result_uses_slowest_rank_per_cycle(tmp_path):
    settings = Settings(**TINY, steps=2)
    settings.validate(2)
    completed_attempt(tmp_path, settings)
    outcome = classify(tmp_path, 0, 2, settings)
    assert outcome["status"] == "completed"
    assert outcome["step_seconds"] == 10
    assert outcome["host_peak_rss_bytes"] == 2 * 789
    assert outcome["host_peak_bytes"] == 2 * 789
    assert outcome["performance_result"] is True


def test_existing_shadowspill_outcome_uses_plan_host_peak_when_read(tmp_path):
    attempt = tmp_path / "attempt-001"
    settings = Settings(**TINY, steps=2)
    completed_attempt(attempt, settings)
    path = attempt / "row.json"
    value = json.loads(path.read_text())
    value["engine"] = "shadowspill"
    value["memory"]["shadowspill"] = [{"predicted_spill_peak_bytes": 42}]
    write_json(path, value)
    outcome = classify(attempt, 0, 2, settings)
    assert outcome["host_peak_bytes"] == 42
    del outcome["host_peak_bytes"]
    write_json(attempt / "outcome.json", outcome)
    assert latest(tmp_path)["host_peak_bytes"] == 42
    assert "host_peak_bytes" not in json.loads((attempt / "outcome.json").read_text())


@pytest.mark.parametrize(
    "problem",
    [
        "no_row",
        "other_configuration",
        "other_world",
        "child_error",
        "not_completed",
        "invalid_row",
    ],
)
def test_partial_or_changed_runs_are_not_results(tmp_path, problem):
    settings = Settings(**TINY, steps=2)
    settings.validate(2)
    if problem == "other_configuration":
        completed_attempt(tmp_path, dataclasses.replace(settings, offload_policy="cpu"))
    elif problem == "other_world":
        completed_attempt(tmp_path, settings, world_size=4)
    elif problem == "child_error":
        completed_attempt(tmp_path, settings)
    elif problem == "not_completed":
        completed_attempt(tmp_path, settings, status="failed")
    elif problem == "invalid_row":
        write_json(tmp_path / "row.json", {"schema": "other"})
    outcome = classify(tmp_path, int(problem == "child_error"), 2, settings)
    assert outcome["status"] == "failed"


def test_oom_distinct_from_generic_rank_failure(tmp_path):
    write_json(tmp_path / "rank-00001/failure.json", {"status": "cuda_out_of_memory"})
    settings = Settings(**TINY)
    settings.validate(2)
    assert classify(tmp_path, 1, 2, settings)["status"] == "cuda_out_of_memory"


def test_a_failed_attempt_gets_a_row_from_the_launcher(tmp_path):
    settings = Settings(**TINY)
    settings.validate(2)
    write_json(tmp_path / "rank-00000/request.json", {"hardware": {"host": "h"}})
    manifest = {"source": {"git_head": "x"}, "environment": {"python": "3"}}
    outcome = {"status": "cuda_out_of_memory", "failures": [{"rank": 1}]}
    value = failure_row(tmp_path, outcome, settings, 2, "fsdp2", manifest)
    assert value["status"] == "cuda_out_of_memory"
    assert value["hardware"] == {"host": "h"}
    assert value["model"]["model"] == "llama3-tiny"
    assert value["failure"]["failures"] == [{"rank": 1}]
    assert value["provenance"]["source"] == {"git_head": "x"}


def test_interrupted_new_attempt_cannot_reuse_an_older_result(tmp_path):
    write_json(tmp_path / "attempt-001/outcome.json", {"status": "completed"})
    (tmp_path / "attempt-002").mkdir()
    assert latest(tmp_path) == {"status": "interrupted", "attempt": "attempt-002"}


def test_a_resume_attempts_a_case_with_no_outcome_and_a_failed_one_on_a_retry():
    assert to_attempt({"status": "pending"}, False)
    assert to_attempt({"status": "interrupted", "attempt": "attempt-002"}, False)
    assert not to_attempt({"status": "failed"}, False)
    assert to_attempt({"status": "cuda_out_of_memory"}, True)
    assert not to_attempt({"status": "completed"}, True)


def test_the_console_names_what_varies_and_what_each_case_measured(tmp_path):
    cases = expand(
        {"schema_version": 1, "defaults": TINY, "grid": {"recompute": [0, 1]}},
        1,
        Settings,
    )
    knobs, label_of = labels(cases, 1)
    assert knobs == ["recompute"]
    assert sorted(label_of.values()) == ["recompute=0", "recompute=1"]
    measured = {
        "status": "completed",
        "global_tokens_per_second": 1476.31,
        "step_seconds": 44.4,
        "peak_reserved_bytes": 12 * 2**30,
        "peak_allocated_bytes": 11 * 2**30,
        "host_peak_rss_bytes": 71 * 2**30,
        "wall_seconds": 390.2,
    }
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    ended = ended_block("[1/2]", "rc0__ab", measured, attempt)
    assert ended[0] == "CASE COMPLETED [1/2] rc0__ab"
    assert "  throughput   1,476.3 tok/s, 44.40 s/update" in ended
    assert "  memory       GPU 12.00 GiB held (11.00 in use), host 71.00 GiB" in ended
    assert ended[3].startswith("  duration     6m 30s, ended ")
    failed = ended_block(
        "[2/2]", "rc1__cd", {"status": "failed", "reason": "exit status 1"}, attempt
    )
    assert failed[:2] == ["CASE FAILED [2/2] rc1__cd", "  reason       exit status 1"]
    assert failed[2] == f"  log          {attempt / 'console.log'}"
    started = started_block("[1/2]", "rc0__ab", "recompute=0", cases[0][1], 1, attempt)
    assert started[:2] == ["CASE START [1/2] rc0__ab", "  settings     recompute=0"]
    assert started[3] == f"  log          {attempt / 'console.log'}"
    assert tally(["pending", "completed", "completed"]) == (
        "2 completed, 1 not yet run"
    )
    assert duration(3725) == "1h 02m 05s"


def test_timeout_terminates_child_and_preserves_log(tmp_path):
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    outcome = run_case(
        [
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(60)",
        ],
        attempt,
        Settings(**TINY),
        1,
        0.5,
    )
    assert outcome["status"] == "timeout"
    assert (attempt / "outcome.json").is_file()
    assert "started" in (attempt / "console.log").read_text()


def test_varied_keys_are_the_columns_of_the_index():
    assert varied({"a": {"x": 1, "y": 2}, "b": {"x": 1, "y": 3}}) == ["y"]
    assert varied({"a": {"x": 1}}) == []


def test_a_resume_refuses_another_world_but_takes_a_grown_matrix(tmp_path, monkeypatch):
    config, output = tmp_path / "config.json", tmp_path / "sweep"
    write_json(config, {"schema_version": 1, "defaults": BIG, "cases": [{}]})
    argv = [
        "--engine",
        "fsdp2",
        "--config",
        str(config),
        "--output",
        str(output),
        "--gpus",
        "8",
        "--dry-run",
    ]
    main(argv)
    assert (output / "README.md").is_file()
    (first,) = [path.name for path in (output / "cases").iterdir()]
    with pytest.raises(SystemExit):
        main(argv)
    with pytest.raises(SystemExit):
        main([*argv[:-3], "4", "--dry-run", "--resume"])
    with pytest.raises(SystemExit):
        main([*argv[:1], "nonesuch", *argv[2:]])
    write_json(
        config,
        {"schema_version": 1, "defaults": BIG, "cases": [{"recompute": 0}, {}]},
    )
    main([*argv, "--resume"])
    manifest = json.loads((output / "manifest.json").read_text())
    names = [case["name"] for case in manifest["cases"]]
    # The grown matrix varies recompute: the new case's name says so, and
    # the case that ran keeps the name it was given.
    assert first.startswith("case__") and names[1] == first
    assert names[0].startswith("rc0__")
    assert manifest["cases"][0]["settings"]["recompute"] == 0
    write_json(config, {"schema_version": 1, "defaults": BIG, "cases": [{}]})
    main([*argv, "--resume"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert [case["name"] for case in manifest["cases"]] == [first]
    assert len(list((output / "cases").iterdir())) == 2


def test_the_code_may_change_under_a_resume_but_not_the_environment():
    held = {
        "engine": "fsdp2",
        "world_size": 1,
        "python": "/py",
        "source": {"git_head": "a", "file_sha256": {"x.py": "1"}},
        "environment": {
            "python": "3.12",
            "packages": {"torch": "2.13"},
            "mlops_source": {"a.py": "1"},
        },
    }
    changed_code = {
        **held,
        "source": {"git_head": "b", "file_sha256": {"x.py": "2"}},
        "environment": {**held["environment"], "mlops_source": {"a.py": "2"}},
    }
    assert comparable(held, changed_code) == []
    upgraded = {
        **held,
        "environment": {**held["environment"], "packages": {"torch": "2.14"}},
    }
    assert comparable(held, upgraded) == ["environment"]
    assert comparable(held, {**held, "world_size": 2}) == ["world_size"]


def test_a_case_keeps_the_name_its_digest_was_given(tmp_path):
    (tmp_path / "cases" / "rc1__aaaaaaaa").mkdir(parents=True)
    settings = Settings(**TINY)
    named = keep_names(
        [("rc0_mb128__bbbbbbbb", settings), ("rc1_mb64__aaaaaaaa", settings)],
        tmp_path,
    )
    assert [name for name, _ in named] == ["rc0_mb128__bbbbbbbb", "rc1__aaaaaaaa"]


def test_a_case_is_named_for_the_settings_its_matrix_varies():
    matrix = {
        "schema_version": 1,
        "defaults": TINY,
        "grid": {"recompute": [0, 0.25], "activation_offload": ["none", "cpu"]},
    }
    names = [name for name, _ in expand(matrix, 1, Settings)]
    assert [name.split("__")[0] for name in names] == [
        "rc0_act-none",
        "rc0_act-cpu",
        "rc0.25_act-none",
        "rc0.25_act-cpu",
    ]
    assert all(len(name.split("__")[1]) == 8 for name in names)
    assert len(set(sweep.CODES.values())) == len(sweep.CODES)


def test_only_the_cases_a_sweep_names_count(tmp_path):
    sweep = tmp_path / "sweep"
    for name in ("rc0__aaaaaaaa", "rc1__bbbbbbbb"):
        write_json(
            sweep / "cases" / name / "attempt-001" / "row.json",
            failure_row(
                sweep / "cases" / name / "attempt-001",
                {"status": "failed", "reason": "x"},
                Settings(**TINY),
                1,
                "fsdp2",
                {"source": {}, "environment": {}},
            ),
        )
    write_json(sweep / "manifest.json", {"cases": [{"name": "rc1__bbbbbbbb"}]})
    rows = read_rows(tmp_path)
    assert [Path(row["_path"]).parts[-3] for row in rows] == ["rc1__bbbbbbbb"]


def test_the_guide_lists_every_case_name_code():
    guide = (ROOT / "baselines" / "experiments" / "README.md").read_text()
    for setting, code in sweep.CODES.items():
        assert f"| `{code}` | `{setting}` |" in guide
