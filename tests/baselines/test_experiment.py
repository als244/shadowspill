"""An experiment spec, its sweep commands, its runs one engine at a time or
whole, and its records."""

import json
import sys
from pathlib import Path

import pytest

from baselines.common.provenance import ROOT
from baselines.experiment import commands, config, process
from baselines.experiment import run as runner

SPEC = {
    "schema_version": 1,
    "gpus": 2,
    "python": {"fsdp2": "baselines"},
    "sweeps": [
        {
            "engine": "fsdp2",
            "config": "baselines/engines/fsdp2/configs/qualification.json",
        },
        {
            "name": "deep",
            "engine": "deepspeed",
            "config": "baselines/engines/deepspeed/configs/qualification.json",
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
    spec = config.load_spec(write_spec(tmp_path, SPEC), tmp_path / "out")
    first, second = spec["sweeps"]
    assert first["name"] == "fsdp2-qualification"
    assert first["python"] == str(
        (Path(sys.prefix).resolve().parent / "baselines" / "bin" / "python").resolve()
    )
    assert first["case_timeout"] == 3600
    assert second["python"] == "/opt/py/bin/python" and second["case_timeout"] == 60
    matrix = tmp_path / "matrix.json"
    command = commands.sweep_command(spec, first, matrix, tmp_path / "sweep", limit=3)
    assert command[1:4] == ["-m", "baselines.sweep", "--engine"]
    assert command[command.index("--config") + 1] == str(matrix)
    assert "--resume" not in command and command[-2:] == ["--limit", "3"]
    (tmp_path / "sweep").mkdir()
    (tmp_path / "sweep" / "manifest.json").write_text("{}")
    resumed = commands.sweep_command(spec, first, matrix, tmp_path / "sweep")
    assert "--resume" in resumed and "--retry-failed" not in resumed
    retried = commands.sweep_command(
        spec, first, matrix, tmp_path / "sweep", retry_failed=True
    )
    assert retried[-2:] == ["--resume", "--retry-failed"]
    report = commands.report_command([tmp_path / "sweep"], tmp_path / "report")
    assert report[-2:] == ["--memory", "reserved"]


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
    matrix = config.sweep_matrix(path, SETTINGS)
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
        config.sweep_matrix(path, SETTINGS)
    spec = {
        **SPEC,
        "settings": SETTINGS,
        "sweeps": [{"engine": "fsdp2", "config": str(path)}],
    }
    with pytest.raises(ValueError, match="experiment's settings"):
        config.load_spec(write_spec(tmp_path, spec), tmp_path / "out")


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"gpus": 0},
        {"python": "baselines"},
        {"settings": ["model"]},
        {"sweeps": []},
        {"sweeps": [{"engine": "fsdp2"}]},
        {"sweeps": [{"engine": "torchtitan", "config": "x.json"}]},
        {"summary_budgets_gib": [8, 0]},
        {"summary_budgets_gib": "8"},
        {"summary_budgets_gib": [True]},
        {
            "sweeps": [
                {
                    "engine": "fsdp2",
                    "config": "baselines/engines/fsdp2/configs/qualification.json",
                    "name": "a/b",
                }
            ]
        },
    ],
)
def test_bad_specs_are_refused(tmp_path, change):
    with pytest.raises(ValueError):
        config.load_spec(write_spec(tmp_path, {**SPEC, **change}), tmp_path / "out")


class Sweeps:
    """Stands in for the sweep launchers: records each command and gives the
    lines of its console."""

    def __init__(self, returncode=0, lines=("a sweep's console",)):
        self.commands, self.returncode, self.lines = [], returncode, lines

    def __call__(self, command, **options):
        assert command[1:3] == ["-m", "baselines.sweep"]
        self.commands.append(command)
        returncode, lines = self.returncode, self.lines

        class Process:
            stdout = iter([f"{line}\n" for line in lines])

            def wait(self):
                return returncode

        return Process()

    def engines(self):
        return [command[command.index("--engine") + 1] for command in self.commands]


def run(tmp_path, monkeypatch, *arguments, sweeps=None):
    sweeps = sweeps or Sweeps()
    monkeypatch.setattr(process.subprocess, "Popen", sweeps)
    spec = write_spec(tmp_path, SPEC)
    runner.main(["--spec", str(spec), "--output", str(tmp_path / "out"), *arguments])
    return sweeps


def record(tmp_path):
    return json.loads((tmp_path / "out" / "experiment.json").read_text())


def test_a_dry_run_runs_each_sweeps_own_dry_run(tmp_path, monkeypatch):
    sweeps = run(tmp_path, monkeypatch, "--dry-run")
    assert sweeps.engines() == ["fsdp2", "deepspeed"]
    assert all("--dry-run" in command for command in sweeps.commands)
    assert [sweep["status"] for sweep in record(tmp_path)["sweeps"]] == [
        "planned",
        "planned",
    ]
    assert record(tmp_path)["report"]["status"] == "planned"
    log = (tmp_path / "out" / "sweeps" / "deep" / "launcher.log").read_text()
    assert "baselines.sweep --engine deepspeed" in log and "a sweep's console" in log
    matrix = tmp_path / "out" / "matrices" / "deep.json"
    assert f"--config {matrix}" in log
    source = ROOT / "baselines/engines/deepspeed/configs/qualification.json"
    assert json.loads(matrix.read_text()) == json.loads(source.read_text())
    (console,) = (tmp_path / "out" / "runs").glob("*.log")
    text = console.read_text()
    assert "── Engine 2/2: deep (deepspeed) ─" in text
    assert "\na sweep's console\n" in text
    assert "SUMMARY" in text and (tmp_path / "out" / "summary.md").is_file()


def test_a_sweep_whose_dry_run_fails_fails_the_experiment(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="failed"):
        run(tmp_path, monkeypatch, "--dry-run", sweeps=Sweeps(returncode=2))
    assert [sweep["status"] for sweep in record(tmp_path)["sweeps"]] == [
        "failed",
        "failed",
    ]


def test_one_engine_at_a_time_keeps_what_the_others_hold(tmp_path, monkeypatch):
    first = run(tmp_path, monkeypatch, "--dry-run", "--engine", "fsdp2")
    assert first.engines() == ["fsdp2"]
    assert [sweep["status"] for sweep in record(tmp_path)["sweeps"]] == [
        "planned",
        "not run",
    ]
    second = run(tmp_path, monkeypatch, "--dry-run", "--only", "deep")
    assert second.engines() == ["deepspeed"]
    held = record(tmp_path)
    assert [sweep["status"] for sweep in held["sweeps"]] == ["planned", "planned"]
    assert [item["selected"] for item in held["runs"]] == [
        ["fsdp2-qualification"],
        ["deep"],
    ]
    runs = tmp_path / "out" / "runs"
    assert len(list(runs.glob("*.json"))) == len(list(runs.glob("*.log"))) == 2


@pytest.mark.parametrize(
    "selection", [["--engine", "megatron_core"], ["--only", "nonesuch"]]
)
def test_a_selection_the_spec_does_not_hold_is_refused(
    tmp_path, monkeypatch, selection
):
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, *selection)


def spec_folder(tmp_path, settings):
    folder = tmp_path / "experiment"
    folder.mkdir()
    (folder / "fsdp2.json").write_text(
        json.dumps({"schema_version": 1, "defaults": {"recompute": 1}})
    )
    spec = {
        "schema_version": 1,
        "gpus": 1,
        "variables": {"models": "where the checkpoints are"},
        "settings": settings,
        "sweeps": [{"name": "fsdp2", "engine": "fsdp2", "config": "fsdp2.json"}],
    }
    (folder / "experiment.json").write_text(json.dumps(spec))
    return folder / "experiment.json"


def test_a_spec_finds_its_matrices_beside_it_and_its_variables_at_launch(tmp_path):
    path = spec_folder(tmp_path, {**SETTINGS, "checkpoint": "{models}/tiny"})
    with pytest.raises(ValueError, match="--variable models="):
        config.load_spec(path, tmp_path / "out")
    with pytest.raises(ValueError, match="declares no variable"):
        config.load_spec(path, tmp_path / "out", {"models": "/m", "other": "x"})
    spec = config.load_spec(path, tmp_path / "out", {"models": "/m"})
    assert spec["settings"]["checkpoint"] == "/m/tiny"
    assert spec["variables"] == {
        "models": {"description": "where the checkpoints are", "value": "/m"}
    }
    assert spec["sweeps"][0]["config"] == str(path.parent / "fsdp2.json")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "experiment.json").write_text(json.dumps({"spec": spec}))
    resumed = config.load_spec(path, tmp_path / "out")
    assert resumed["settings"]["checkpoint"] == "/m/tiny"


def test_a_reference_to_an_undeclared_variable_is_refused(tmp_path):
    path = spec_folder(tmp_path, {**SETTINGS, "checkpoint": "{weights}/tiny"})
    with pytest.raises(ValueError, match="not a variable"):
        config.load_spec(path, tmp_path / "out", {"models": "/m"})


def test_a_run_publishes_its_summary_without_its_paths(tmp_path, monkeypatch):
    target = tmp_path / "results"
    run(tmp_path, monkeypatch, "--dry-run", "--publish", str(target))
    published = (target / "summary.md").read_text()
    assert "# out: results" in published or "results" in published
    assert str(tmp_path / "out") not in published
    data = json.loads((target / "summary.json").read_text())
    assert "experiment" not in data
    held = record(tmp_path)
    assert held["summary"]["published"] == str(target)


def test_an_overwrite_discards_only_the_engines_it_runs(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, "--dry-run")
    held = tmp_path / "out" / "sweeps"
    (held / "fsdp2-qualification" / "kept.txt").write_text("a result")
    (held / "deep" / "kept.txt").write_text("a result")
    run(tmp_path, monkeypatch, "--dry-run", "--overwrite", "--engine", "deepspeed")
    assert (held / "deep" / "kept.txt").is_file()
    run(tmp_path, monkeypatch, "--overwrite", "--engine", "deepspeed", "--skip-report")
    assert not (held / "deep" / "kept.txt").exists()
    assert (held / "fsdp2-qualification" / "kept.txt").is_file()
    console = sorted((tmp_path / "out" / "runs").glob("*.log"))[-1]
    assert "discarded" in console.read_text()


def test_an_interrupted_engine_is_recorded_so(tmp_path, monkeypatch):
    class Interrupted(Sweeps):
        def __call__(self, command, **options):
            process = super().__call__(command, **options)

            class Stopped:
                @property
                def stdout(self):
                    raise KeyboardInterrupt

                def wait(self):
                    return process.wait()

            return Stopped()

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, monkeypatch, sweeps=Interrupted())
    held = record(tmp_path)
    assert held["sweeps"][0]["status"] == "interrupted"
    assert held["runs"][0]["finished_at"] is not None
