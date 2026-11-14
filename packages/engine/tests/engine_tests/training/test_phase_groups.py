"""Fresh phase ordering follows logical ownership and never launches execution early."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.training.test_preparation_templates import inventory
from shadowspill.orchestration.launch.phases import PhaseResult
from shadowspill.parallelism import (
    GroupRef,
    InputFieldPlacement,
    InputPlacement,
    OutputPlacement,
    Parallelism,
)
from shadowspill.training.preparation import phases
from shadowspill.training.preparation.groups import capture_groups


def test_capture_groups_deduplicate_counts_but_preserve_all_worker_ownership():
    base = inventory()
    groups = capture_groups(base)
    assert len(groups) == len(base.templates) == 9
    assert {g[0] for g in groups} == base.templates.keys()
    placement = InputPlacement(
        GroupRef("cooperators", ((2, 0), (3, 1))),
        outputs={"prediction": OutputPlacement()},
        fields=[
            InputFieldPlacement(tuple(f.path))
            for f in base.layouts.setup.config.inputs.fields
        ],
    )
    distributed = inventory(
        parallelism=Parallelism(ranks_per_replica=4, inputs=placement)
    )
    groups = capture_groups(distributed)
    assert len(groups) * 4 == len(distributed.templates)
    assert all(
        distributed.templates[key].binding.logical_rank == rank
        for group in groups
        for rank, key in enumerate(group)
    )
    assert groups == capture_groups(distributed)
    with pytest.raises(ValueError, match="exactly the prepared templates"):
        capture_groups(replace(distributed, families={}))


def test_phase_driver_can_prepare_and_calibrate_without_starting_execution(
    tmp_path, monkeypatch
):
    calls = []

    def worker(module, args, *, ranks):
        calls.append((module, ranks))
        return [module, str(args[0])]

    environments = []

    def run(command, *, env, **kwargs):
        environments.append((command[0], env["CUDA_VISIBLE_DEVICES"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    monkeypatch.setattr(phases, "worker_command", worker)
    monkeypatch.setattr(phases.subprocess, "run", run)
    phases.run_phases(
        "prepared",
        tmp_path,
        "all",
        ranks=2,
        preparation_ranks=2,
        phases=("build", "profile", "communicate", "plan"),
    )
    assert calls == [
        ("prepared." + phase, 1 if phase == "plan" else 2)
        for phase in ("build", "profile", "communicate", "plan")
    ]
    assert environments[-1] == ("prepared.plan", "")
    assert all(mask == "2,4" for _, mask in environments[:-1])
    calls.clear()
    phases.run_phases("existing", tmp_path, "all")
    assert [module for module, _ in calls] == [
        "existing." + p for p in ("build", "profile", "plan", "execute")
    ]


@pytest.mark.parametrize(
    "sequence", [(), ("plan", "build"), ("build", "build"), ("unknown",)]
)
def test_invalid_phase_sequences_fail_before_any_process(tmp_path, sequence):
    with pytest.raises(ValueError, match="ordered, distinct"):
        phases.run_phases("unused", tmp_path, "all", phases=sequence)


def test_explicit_launch_environment_is_snapshotted_once_without_global_mutation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "outside")
    environment = {"CUDA_VISIBLE_DEVICES": "assigned", "PHASE_VALUE": "original"}
    calls = []

    def launch(invocation):
        calls.append(invocation)
        environment["PHASE_VALUE"] = "changed by caller"
        with pytest.raises(TypeError):
            invocation.environment["PHASE_VALUE"] = "mutation"
        return PhaseResult(0, ())

    phases.run_phases(
        "installed",
        tmp_path,
        "all",
        phases=("build", "plan"),
        environment=environment,
        launch=launch,
    )
    assert [dict(call.environment) for call in calls] == [
        {"CUDA_VISIBLE_DEVICES": "assigned", "PHASE_VALUE": "original"},
        {"CUDA_VISIBLE_DEVICES": "", "PHASE_VALUE": "original"},
    ]
    assert phases.os.environ["CUDA_VISIBLE_DEVICES"] == "outside"
