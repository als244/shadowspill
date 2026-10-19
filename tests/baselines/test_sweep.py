"""Reject incomparable configurations and partial/failed distributed results."""

import argparse
import dataclasses
import json
import sys

import pytest

from baselines.common import write_json
from baselines.fsdp2.settings import Settings, add_arguments
from baselines.fsdp2.sweep import classify, expand, latest, main, run_case


def test_grid_keeps_global_step_constant_and_roundtrips_every_control():
    cases = expand(
        {
            "schema_version": 1,
            "defaults": {"forward_prefetch": 2, "backward_prefetch": "2"},
            "grid": {
                "microbatch_tokens": [4096, 131072],
                "offload": ["none", "cpu"],
                "gradient_sync": ["per-microbatch", "last-microbatch"],
            },
        },
        8,
    )
    assert len(cases) == 8
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    for _, settings in cases:
        assert (
            settings.microbatch_tokens * settings.gradient_accumulation * 8 == 2097152
        )
        assert (
            Settings.from_dict(vars(parser.parse_args(settings.arguments())))
            == settings
        )


@pytest.mark.parametrize(
    "override",
    [
        {"gradient_accumulation": 1},
        {"microbatch_tokens": 5000},
        {"backward_prefetch": 0},
        {"forward_prefetch": -1},
        {"offload": "typo"},
        {"recompute": float("nan")},
        {"pin_memory": "false"},
        {"cpu_threads": 0},
        {"allocator_cap_gib": float("inf")},
        {"verify_reference": True},
    ],
)
def test_invalid_controls_are_rejected_before_launch(override):
    with pytest.raises(ValueError):
        expand({"schema_version": 1, "cases": [override]}, 8)


def test_duplicate_effective_cases_run_once():
    assert (
        len(
            expand(
                {"schema_version": 1, "cases": [{}, {"gradient_accumulation": 64}]}, 8
            )
        )
        == 1
    )


def test_unknown_knob_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown"):
        expand({"schema_version": 1, "grid": {"prefetch_typo": [1]}}, 8)


def result_fixture(tmp_path):
    settings = Settings(steps=2, gradient_accumulation=256)
    ranks = [
        {
            "rank": rank,
            "status": "completed",
            "cycles": cycles,
            "performance_result": True,
            "qualified_against_reference": False,
            "peak_allocated_bytes": 123,
            "peak_reserved_bytes": 456,
            "host_peak_rss_bytes": 789,
        }
        for rank, cycles in enumerate([[1, 10], [10, 1]])
    ]
    result = {
        "status": "completed",
        "request": dataclasses.asdict(settings),
        "ranks": ranks,
    }
    write_json(tmp_path / "results.json", result)
    return settings, result


def test_result_uses_slowest_rank_per_cycle(tmp_path):
    settings, _ = result_fixture(tmp_path)
    outcome = classify(tmp_path, 0, 2, settings)
    assert outcome["status"] == "completed"
    assert outcome["step_seconds"] == 10


@pytest.mark.parametrize(
    "problem",
    [
        "missing_rank",
        "duplicate_rank",
        "partial_steps",
        "wrong_settings",
        "child_error",
    ],
)
def test_partial_or_changed_runs_are_not_results(tmp_path, problem):
    settings, result = result_fixture(tmp_path)
    if problem == "missing_rank":
        result["ranks"].pop()
    elif problem == "duplicate_rank":
        result["ranks"][1]["rank"] = 0
    elif problem == "partial_steps":
        result["ranks"][0]["cycles"].pop()
    elif problem == "wrong_settings":
        result["request"]["offload"] = "cpu"
    write_json(tmp_path / "results.json", result)
    assert (
        classify(tmp_path, int(problem == "child_error"), 2, settings)["status"]
        == "failed"
    )


def test_oom_distinct_from_generic_rank_failure(tmp_path):
    write_json(tmp_path / "rank-00001/failure.json", {"status": "cuda_out_of_memory"})
    assert classify(tmp_path, 1, 2, Settings())["status"] == "cuda_out_of_memory"


def test_interrupted_new_attempt_cannot_reuse_an_older_result(tmp_path):
    write_json(tmp_path / "attempt-001/outcome.json", {"status": "completed"})
    (tmp_path / "attempt-002").mkdir()
    assert latest(tmp_path) == {"status": "interrupted", "attempt": "attempt-002"}


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
        Settings(),
        1,
        0.5,
    )
    assert outcome["status"] == "timeout"
    assert (attempt / "outcome.json").is_file()
    assert "started" in (attempt / "console.log").read_text()


def test_dry_run_and_resume_refuse_different_settings(tmp_path):
    config, output = tmp_path / "config.json", tmp_path / "sweep"
    write_json(config, {"schema_version": 1, "cases": [{}]})
    argv = [
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
    main([*argv, "--resume"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cases"][0]["settings"]["gradient_accumulation"] == 64
    write_json(config, {"schema_version": 1, "cases": [{"recompute": 0}]})
    with pytest.raises(SystemExit):
        main([*argv, "--resume"])
