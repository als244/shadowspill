"""Independent CPU Program searches preserve selection and failure semantics."""

import json
import zipfile
from dataclasses import replace

import pytest

from engine_tests.offline._example import communication, profiled
from shadowspill import Plan, TaskProfile, apply_profiles
from shadowspill.artifacts import TaskDefinition
from shadowspill.errors import PlanInfeasibleError
from shadowspill.training.preparation import api
from shadowspill.training.preparation.phases import run_phases
from shadowspill.training.preparation.plan import run
from shadowspill_workloads.recipes import LoRA, Model, train
from tests.benchmarks._profiled import measured


def _prepare(root, *, workers, evaluation=False):
    candidates = []
    for name, duration, workspace in (
        ("slow", 4e-6, 16),
        ("fast", 1e-6, 16),
        ("oversized", 1e-6, 16384),
    ):
        original = profiled()
        profiles = original.profiles
        profiles.tasks["update_impl"] = replace(
            profiles.tasks["update_impl"],
            duration_seconds=duration,
            workspace_bytes=workspace,
        )
        profiles.tasks["invocation_start"] = TaskProfile(0, 0)
        program = replace(
            original.program,
            task_definitions=(
                *original.program.task_definitions,
                TaskDefinition("invocation_start", "invocation-start-v1", 0),
            ),
        )
        observed = measured(apply_profiles(program, profiles))
        forms = [
            {
                "variant": "save",
                "accumulates": accumulates,
                "saved_aliases": [],
                "forward": {
                    "profile_key": "boundary",
                    "inputs": [],
                    "outputs": [],
                    "mutations": [],
                    "replacements": [],
                    "handoffs": [],
                },
                "backward": {
                    "profile_key": "boundary",
                    "inputs": [],
                    "contributions": [],
                    "replacements": [],
                    "discarded": [],
                    "handoffs": [],
                },
            }
            for accumulates in (False, True)
        ]
        geometry = dict(shape=[16], stride=[1], dtype="float32", storage_offset=0)
        bindings = {
            **observed.program.bindings,
            "registrations": [
                {"name": "weight", "object_id": "weight", "parameter": True}
            ],
            "model_tensor_geometry": {"weight": geometry},
            "task_inputs": {
                key: {
                    "tensor_positions": [0] if key == "update_impl" else [],
                    "tensor_geometry": [geometry] if key == "update_impl" else [],
                }
                for key in profiles.tasks
            },
        }
        program = replace(
            observed.program,
            bindings={
                **bindings,
                "kind": "step",
                "gradients": [],
                "training_templates": [[forms], [forms]],
                "optimizer_templates": [
                    {
                        "task_id": "update",
                        "profile_key": "update_impl",
                        "inputs": [{"object_id": "weight", "leaf_index": 0}],
                        "mutations": ["weight"],
                        "host_inputs": [],
                        "replacements": [],
                    }
                ],
            },
        )
        replace(observed, program=program).save(
            root / "profiled" / name / "training.ssp"
        )
        communication().save(root / "communication" / name / "training.ssp")
        if evaluation:
            forward = replace(
                observed.program,
                bindings={**bindings, "kind": "forward"},
                tasks=tuple(
                    replace(task, mutations=(), phase="forward")
                    for task in observed.program.tasks
                ),
            )
            replace(observed, program=forward).save(
                root / "profiled" / name / "evaluation.ssp"
            )
            communication().save(root / "communication" / name / "evaluation.ssp")
        metadata = root / "programs" / name / "metadata.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({"name": name}))
        candidates.append(dict(path=name, num_examples=2))
    options = dict(
        candidates=candidates,
        eval_every=int(evaluation),
        execution_gib=4096 / (1 << 30),
        spill_gib=4096 / (1 << 30),
        minimum_evict_bytes=0,
        search_workers=1,
        program_workers=workers,
        plots=False,
    )
    (root / "config.json").write_text(json.dumps(options))


def test_spawned_program_workers_preserve_all_cases_and_minimum_selection(tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    for root, workers in ((serial, 1), (parallel, 2)):
        _prepare(root, workers=workers)
        if workers == 1:
            run(root)
        else:
            run_phases("shadowspill.training.preparation", root, "plan")

    def read(root, name):
        return json.loads((root / name).read_text())

    rows = read(parallel, "search.json")
    assert rows == read(serial, "search.json")
    assert len(rows) == 6
    assert sum("duration_seconds" in row for row in rows) == 4
    selected = read(parallel, "selected.json")
    assert selected == read(serial, "selected.json")
    assert selected["path"] == "fast" and selected["ordering"] == "1x2rp"
    assert selected["duration_seconds"] == min(
        row["duration_seconds"] for row in rows if "duration_seconds" in row
    )
    assert read(parallel, "model.json") == {"name": "fast"}
    work = read(parallel, "planning-work.json")
    assert work["program_workers"] == 2 and work["search_workers_per_program"] == 1
    assert len(work["programs"]) == 3
    assert sum(row["cases"] for row in work["programs"]) == 6
    winner = parallel / "plans/candidates/fast/1x2rp/training.ssp"
    assert winner.read_bytes() == (parallel / "plans/training.ssp").read_bytes()
    first, second = (
        Plan.load(root / "plans/training.ssp") for root in (serial, parallel)
    )
    assert first.summary == second.summary
    assert first.schedule == second.schedule
    assert first.selections == second.selections


def test_worker_contract_failure_does_not_leave_a_runnable_selection(tmp_path):
    _prepare(tmp_path, workers=2)
    (tmp_path / "selected.json").write_text('{"old":true}')
    (tmp_path / "search.json").write_text('[{"old":true}]')
    (tmp_path / "planning-work.json").write_text('{"old":true}')
    (tmp_path / "profiled/fast/training.ssp").write_bytes(b"invalid archive")
    with pytest.raises(zipfile.BadZipFile):
        run(tmp_path)
    assert not (tmp_path / "selected.json").exists()
    assert not (tmp_path / "plans/training.ssp").exists()
    assert not (tmp_path / "search.json").exists()
    assert not (tmp_path / "planning-work.json").exists()


def test_parallel_workers_preserve_joint_evaluation_admission_and_archives(tmp_path):
    from shadowspill.planning.shared import shared_memory

    _prepare(tmp_path, workers=2, evaluation=True)
    run(tmp_path)
    rows = json.loads((tmp_path / "search.json").read_text())
    assert len(rows) == 5
    assert "evaluation_error" in rows[-1]
    first = tmp_path / "plans/candidates/fast/1x2rp/evaluation.ssp"
    second = tmp_path / "plans/candidates/fast/2x1rp/evaluation.ssp"
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == (tmp_path / "plans/evaluation.ssp").read_bytes()
    training = Plan.load(tmp_path / "plans/training.ssp")
    evaluation = Plan.load(tmp_path / "plans/evaluation.ssp")
    assert shared_memory(training, {"evaluation": evaluation}).aliases[
        "evaluation"
    ] == {"weight_storage": "weight_storage"}


def test_all_failed_preparations_remain_diagnostic_rows(tmp_path):
    _prepare(tmp_path, workers=8)
    (tmp_path / "candidate-failures.json").write_text(
        json.dumps(
            {
                name: {"phase": "profile", "kind": "device_capacity"}
                for name in ("slow", "fast", "oversized")
            }
        )
    )
    with pytest.raises(PlanInfeasibleError, match="no training/evaluation pair"):
        run(tmp_path)
    rows = json.loads((tmp_path / "search.json").read_text())
    assert len(rows) == 3 and all("failure" in row for row in rows)
    assert not (tmp_path / "selected.json").exists()


@pytest.mark.parametrize("family", ["catalog", "qwen3_5_moe", "glm5_next"])
def test_public_worker_controls_are_independent_and_change_without_rebuild(
    tmp_path, monkeypatch, family
):
    monkeypatch.setattr(api, "run_phases", lambda *args, **kwargs: None)
    if family == "catalog":
        model = Model.from_config("qwen3moe-tiny")
    else:
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps({"model_type": family}))
        model = Model.from_checkpoint(checkpoint, lora=LoRA())
    root = train(
        model,
        {
            "run_dir": tmp_path / "run",
            "data": tmp_path / "data",
            "initialization": "random",
        },
        phase="build",
    )
    initial = json.loads((root / "config.json").read_text())
    assert initial["program_workers"] == initial["search_workers"] == 8
    train(
        model,
        {"run_dir": root, "program_workers": 8, "search_workers": 16},
        phase="plan",
    )
    options = json.loads((root / "config.json").read_text())
    assert options["program_workers"] == 8 and options["search_workers"] == 16
    assert options["candidates"] == initial["candidates"]
    for value in (0, -1, True):
        with pytest.raises(
            ValueError, match="program_workers must be a positive integer"
        ):
            train(model, {"run_dir": root, "program_workers": value}, phase="plan")
