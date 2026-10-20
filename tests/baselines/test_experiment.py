"""An experiment spec, its sweep commands and a dry run."""

import json
import sys
from pathlib import Path

import pytest

from baselines import experiment

SPEC = {
    "schema_version": 1,
    "gpus": 2,
    "python": {"fsdp2": "baselines"},
    "sweeps": [
        {"harness": "fsdp2", "config": "baselines/fsdp2/configs/qualification.json"},
        {
            "name": "deep",
            "harness": "deepspeed",
            "config": "baselines/deepspeed/configs/qualification.json",
            "python": "/opt/py/bin/python",
            "case_timeout": 60,
        },
    ],
}


def write_spec(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_the_spec_resolves_names_and_interpreters(tmp_path):
    spec = experiment.load_spec(write_spec(tmp_path, SPEC), tmp_path / "out")
    first, second = spec["sweeps"]
    assert first["name"] == "fsdp2-qualification"
    assert first["python"] == str(
        (Path(sys.prefix).resolve().parent / "baselines" / "bin" / "python").resolve()
    )
    assert first["case_timeout"] == 3600
    assert second["python"] == "/opt/py/bin/python" and second["case_timeout"] == 60
    matrix = tmp_path / "matrix.json"
    command = experiment.sweep_command(spec, first, matrix, tmp_path / "sweep", limit=3)
    assert command[1:4] == ["-m", "baselines.sweep", "--harness"]
    assert command[command.index("--config") + 1] == str(matrix)
    assert "--resume" not in command and command[-2:] == ["--limit", "3"]
    (tmp_path / "sweep").mkdir()
    (tmp_path / "sweep" / "manifest.json").write_text("{}")
    resumed = experiment.sweep_command(spec, first, matrix, tmp_path / "sweep")
    assert "--resume" in resumed


def write_matrix(tmp_path, matrix):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({"schema_version": 1, **matrix}))
    return path


SETTINGS = {"model": "llama3-tiny", "sequence_length": 64, "tokens_per_step": 1024}


def test_the_settings_join_every_case_of_a_matrix(tmp_path):
    path = write_matrix(
        tmp_path,
        {"defaults": {"zero_stage": 2}, "grid": {"tokens_per_microbatch": [64, 128]}},
    )
    matrix = experiment.sweep_matrix(path, SETTINGS)
    assert matrix["defaults"] == {**SETTINGS, "zero_stage": 2}
    assert matrix["grid"] == {"tokens_per_microbatch": [64, 128]}


@pytest.mark.parametrize(
    "matrix",
    [
        {"defaults": {"model": "llama3-1b"}},
        {"grid": {"tokens_per_step": [1024, 2048]}},
        {"cases": [{}, {"sequence_length": 128}]},
    ],
)
def test_a_matrix_that_names_a_shared_setting_is_refused(tmp_path, matrix):
    path = write_matrix(tmp_path, matrix)
    with pytest.raises(ValueError, match="experiment's settings"):
        experiment.sweep_matrix(path, SETTINGS)
    spec = {
        **SPEC,
        "settings": SETTINGS,
        "sweeps": [{"harness": "fsdp2", "config": str(path)}],
    }
    with pytest.raises(ValueError, match="experiment's settings"):
        experiment.load_spec(write_spec(tmp_path, spec), tmp_path / "out")


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"gpus": 0},
        {"python": "baselines"},
        {"settings": ["model"]},
        {"sweeps": []},
        {"sweeps": [{"harness": "fsdp2"}]},
        {"sweeps": [{"harness": "torchtitan", "config": "x.json"}]},
        {
            "sweeps": [
                {
                    "harness": "fsdp2",
                    "config": "baselines/fsdp2/configs/qualification.json",
                    "name": "a/b",
                }
            ]
        },
    ],
)
def test_bad_specs_are_refused(tmp_path, change):
    with pytest.raises(ValueError):
        experiment.load_spec(write_spec(tmp_path, {**SPEC, **change}), tmp_path / "out")


def test_a_dry_run_runs_each_sweeps_own_dry_run(tmp_path, monkeypatch):
    launched = []

    def sweep_dry_run(command, **options):
        assert command[1:3] == ["-m", "baselines.sweep"] and "--dry-run" in command
        launched.append(command)
        return experiment.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(experiment.subprocess, "run", sweep_dry_run)
    experiment.main(
        [
            "--spec",
            str(write_spec(tmp_path, SPEC)),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert len(launched) == 2
    record = json.loads((tmp_path / "out" / "experiment.json").read_text())
    assert [sweep["status"] for sweep in record["sweeps"]] == ["planned", "planned"]
    assert record["report"]["status"] == "planned"
    log = (tmp_path / "out" / "sweeps" / "deep" / "launcher.log").read_text()
    assert "baselines.sweep --harness deepspeed" in log
    matrix = tmp_path / "out" / "matrices" / "deep.json"
    assert f"--config {matrix}" in log
    source = experiment.ROOT / "baselines/deepspeed/configs/qualification.json"
    assert json.loads(matrix.read_text()) == json.loads(source.read_text())


def test_a_sweep_whose_dry_run_fails_fails_the_experiment(tmp_path, monkeypatch):
    def refused(command, **options):
        return experiment.subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(experiment.subprocess, "run", refused)
    arguments = ["--spec", str(write_spec(tmp_path, SPEC))]
    with pytest.raises(SystemExit, match="failed"):
        experiment.main([*arguments, "--output", str(tmp_path / "out"), "--dry-run"])
    record = json.loads((tmp_path / "out" / "experiment.json").read_text())
    assert [sweep["status"] for sweep in record["sweeps"]] == ["failed", "failed"]
