"""Tracing one case of an experiment: which case, the command nsys runs it
under, and where its trace goes; nothing is launched."""

import json
import shlex
from pathlib import Path

import pytest

from baselines import trace
from baselines.common import write_json
from baselines.engines.fsdp2.settings import Settings

TINY = {
    "model": "llama3-tiny",
    "sequence_length": 64,
    "tokens_per_step": 2048,
    "tokens_per_microbatch": 64,
}


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "sweeps" / "fsdp2"
    write_json(
        root / "manifest.json",
        {"engine": "fsdp2", "world_size": 1, "python": "/opt/py/bin/python"},
    )
    write_json(
        root / "cases" / "rc1__1a2b3c4d" / "config.json",
        Settings.from_dict({**TINY, "recompute": 1}).as_dict(),
    )
    return tmp_path


def traced(experiment, *arguments):
    case = [str(experiment), "fsdp2", "1a2b3c4d"]
    trace.main([*case, "--nsys", "/opt/nsys", "--dry-run", *arguments])
    if "--output" in arguments:
        output = Path(arguments[arguments.index("--output") + 1])
    else:
        traces = experiment / "traces" / "fsdp2" / "rc1__1a2b3c4d"
        (output,) = traces.iterdir()
    return output, shlex.split((output / "command.txt").read_text())


def test_the_case_runs_as_its_sweep_ran_it_under_nsys(experiment):
    output, command = traced(experiment, "--steps", "2")
    assert command[:2] == ["/opt/nsys", "profile"]
    assert f"--output={output / 'trace'}" in command
    assert "--capture-range=cudaProfilerApi" in command
    assert "--gpu-metrics-devices=all" in command
    launched = command[command.index("baselines.common.launch") - 2 :]
    assert launched[launched.index("--python") + 1] == "/opt/py/bin/python"
    assert "--nsys-capture" in launched
    assert launched[launched.index("--steps") + 1] == "2"
    asked = json.loads((output / "trace.json").read_text())
    assert asked["settings"]["nsys_capture"] and asked["settings"]["recompute"] == 1


def test_the_options_reach_nsys(experiment, tmp_path):
    output, command = traced(
        experiment,
        "--capture",
        "whole",
        "--gpu-metrics-devices",
        "0",
        "--cuda-graph-trace",
        "node",
        "--no-cuda-memory-usage",
        "--export",
        "sqlite",
        "--nsys-arg=--delay=5",
        "--name",
        "fsdp2-case7",
        "--output",
        str(tmp_path / "elsewhere"),
    )
    assert output == tmp_path / "elsewhere"
    assert "--capture-range=cudaProfilerApi" not in command
    for option in (
        f"--output={output / 'fsdp2-case7'}",
        "--gpu-metrics-devices=0",
        "--cuda-graph-trace=node",
        "--cuda-memory-usage=false",
        "--export=sqlite",
        "--delay=5",
    ):
        assert option in command


def test_a_case_is_named_once(experiment):
    with pytest.raises(SystemExit):
        trace.main([str(experiment), "fsdp2", "nonesuch", "--nsys", "/opt/nsys"])
    with pytest.raises(SystemExit):
        trace.main([str(experiment), "deepspeed", "1a2b3c4d5e6f", "--nsys", "/x"])
