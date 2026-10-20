"""Reject incomparable configurations and partial or failed distributed
results, for any harness that exposes its settings."""

import argparse
import dataclasses
import json
import sys

import pytest

from baselines.common import write_json
from baselines.common.results import memory_peaks, row, summarize_cycles, varied
from baselines.fsdp2.settings import Settings, add_arguments, from_namespace
from baselines.sweep import (
    classify,
    expand,
    failure_row,
    harness_module,
    latest,
    main,
    run_case,
)

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
                "offload": ["none", "cpu"],
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


def test_a_harness_is_its_settings_module():
    module = harness_module("fsdp2")
    assert module.Settings is Settings
    command = module.command("py", 2, Settings(**TINY), "out")
    assert command[:10] == [
        "py",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--max-restarts=0",
        "-m",
        "baselines.fsdp2.train",
        "--output",
        "out",
    ]
    assert command[10:] == Settings(**TINY).arguments()
    with pytest.raises(ValueError, match="no harness"):
        harness_module("nonesuch")


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
        harness="fsdp2",
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
    assert outcome["performance_result"] is True


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
        completed_attempt(tmp_path, dataclasses.replace(settings, offload="cpu"))
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


def test_dry_run_and_resume_refuse_different_settings(tmp_path):
    config, output = tmp_path / "config.json", tmp_path / "sweep"
    write_json(config, {"schema_version": 1, "defaults": BIG, "cases": [{}]})
    argv = [
        "--harness",
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
    main([*argv, "--resume"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["harness"] == "fsdp2"
    assert manifest["cases"][0]["settings"]["tokens_per_step"] == 2097152
    with pytest.raises(SystemExit):
        main([*argv[:-3], "4", "--dry-run", "--resume"])
    write_json(
        config, {"schema_version": 1, "defaults": BIG, "cases": [{"recompute": 0}]}
    )
    with pytest.raises(SystemExit):
        main([*argv, "--resume"])
    with pytest.raises(SystemExit):
        main([*argv[:1], "nonesuch", *argv[2:]])
