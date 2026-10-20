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
    command = experiment.sweep_command(spec, first, tmp_path / "sweep", limit=3)
    assert command[1:4] == ["-m", "baselines.sweep", "--harness"]
    assert "--resume" not in command and command[-2:] == ["--limit", "3"]
    (tmp_path / "sweep").mkdir()
    (tmp_path / "sweep" / "manifest.json").write_text("{}")
    assert "--resume" in experiment.sweep_command(spec, first, tmp_path / "sweep")


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"gpus": 0},
        {"python": "baselines"},
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


def test_a_dry_run_writes_the_plan_and_runs_nothing(tmp_path, monkeypatch):
    def forbidden(*arguments, **options):
        raise AssertionError("a dry run launches nothing")

    monkeypatch.setattr(experiment.subprocess, "run", forbidden)
    experiment.main(
        [
            "--spec",
            str(write_spec(tmp_path, SPEC)),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    record = json.loads((tmp_path / "out" / "experiment.json").read_text())
    assert [sweep["status"] for sweep in record["sweeps"]] == ["planned", "planned"]
    assert record["report"]["status"] == "planned"
    log = (tmp_path / "out" / "sweeps" / "deep" / "launcher.log").read_text()
    assert "baselines.sweep --harness deepspeed" in log
