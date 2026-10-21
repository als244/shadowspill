"""Generating an experiment from a template: the values derived from the
inputs, the folder written and checked, and what the generator refuses."""

import argparse
import json

import pytest

from baselines import generate_experiment

TINY = [
    "--model",
    "llama3-tiny",
    "--sequence-length",
    "64",
    "--tokens-per-step",
    "1024",
    "--gpu-memory-gib",
    "32",
    "--host-memory-gib",
    "188",
]


@pytest.fixture
def experiments(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    root.mkdir()
    monkeypatch.setattr(generate_experiment, "EXPERIMENTS", root)
    return root


def inputs(**changes):
    values = {
        "model": "llama3-8b",
        "implementation": "mlops",
        "lora": {"rank": 16, "alpha": 16.0},
        "gpus": 1,
        "sequence_length": 4096,
        "tokens_per_step": 65536,
        "gpu_memory_gib": 32,
        "host_memory_gib": 188,
        "checkpoint": None,
        "corpus": None,
    }
    return argparse.Namespace(**{**values, **changes})


def test_the_values_follow_from_the_geometry_and_the_machine():
    values = generate_experiment.derived(inputs())
    assert values["microbatches"] == [4096, 8192, 16384, 32768, 65536]
    assert values["accumulated_microbatches"] == [4096, 8192, 16384, 32768]
    assert values["single_microbatch"] == [65536]
    assert values["execution_budgets"] == [8, 12, 16, 24, 28]
    assert values["summary_budgets"] == [8, 12, 16, 24, 28, 32]
    assert values["spill_gib"] == 144
    assert values["checkpoint"] is generate_experiment.OMIT
    two = generate_experiment.derived(
        inputs(gpus=2, gpu_memory_gib=80, host_memory_gib=1024)
    )
    assert two["microbatches"] == [4096, 8192, 16384, 32768]
    assert two["execution_budgets"] == [20, 30, 40, 60, 70] and two["spill_gib"] == 816
    with pytest.raises(ValueError):
        generate_experiment.derived(inputs(tokens_per_step=6000))


def test_a_left_out_value_and_an_empty_grid_go():
    template = {"a": "@checkpoint", "b": "@microbatches", "c": [1, "@spill_gib"]}
    values = {
        "checkpoint": generate_experiment.OMIT,
        "microbatches": [1],
        "spill_gib": 16,
    }
    assert generate_experiment.fill(template, values) == {"b": [1], "c": [1, 16]}
    matrix = {"grid": [{"x": [1]}, {"x": []}]}
    assert generate_experiment.without_empty_grids(matrix) == {"grid": [{"x": [1]}]}
    with pytest.raises(ValueError):
        generate_experiment.fill("@nothing", values)


def test_a_full_experiment_is_written_checked_and_described(experiments, capsys):
    generate_experiment.main(
        [
            *["round1/tiny", "--template", "full", *TINY, "--setting", "steps=5"],
            *["--description", "A tiny check."],
        ]
    )
    folder = experiments / "round1" / "tiny"
    spec = json.loads((folder / "experiment.json").read_text())
    assert spec["settings"]["model"] == "llama3-tiny" and spec["settings"]["steps"] == 5
    assert "checkpoint" not in spec["settings"] and "variables" not in spec
    assert spec["summary_budgets_gib"] == [8, 12, 16, 24, 28, 32]
    record = json.loads((folder / "generation.json").read_text())
    assert record["template"] == "full"
    assert record["cases"] == {"shadowspill": 25, "fsdp2": 206, "deepspeed": 90}
    readme = (folder / "README.md").read_text()
    assert readme.startswith("# tiny\n\nA tiny check.")
    assert "### fsdp2: 206 cases" in readme and "run once" in readme
    assert "fsdp2: 206 cases" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        generate_experiment.main(["round1/tiny", "--template", "full", *TINY])


def test_a_lora_experiment_declares_its_variables(experiments, capsys):
    generate_experiment.main(
        [
            *["tiny-lora", "--template", "lora", *TINY, "--checkpoint", "Tiny"],
            *["--corpus", "llama3"],
        ]
    )
    spec = json.loads((experiments / "tiny-lora" / "experiment.json").read_text())
    assert spec["settings"]["checkpoint"] == "{models}/Tiny"
    assert spec["settings"]["data"] == "{corpora}/llama3"
    assert set(spec["variables"]) == {"models", "corpora"}
    assert "the dry run reads them" in capsys.readouterr().out


@pytest.mark.parametrize("path", ["/abs", "../out", "."])
def test_a_folder_outside_the_experiments_is_refused(experiments, path):
    with pytest.raises(SystemExit):
        generate_experiment.main([path, "--template", "full", *TINY])
