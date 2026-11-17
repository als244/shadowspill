"""Equivalent rank captures share frozen binaries while retaining local bindings."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.offline._example import program as example
from shadowspill import Profiles, TaskProfile
from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import Program
from shadowspill.profiling.distributed import combine_profiles
from shadowspill.training.preparation import capture, job_phases
from shadowspill.training.preparation.compiled_reuse import reuse_compiled_tasks


def fixture(tile, *, rank=0, contract=None):
    identity = {
        "task": "graph-layout-dependencies-resources",
        "triton_choices": {},
        "inductor_kernels": {"pointwise": f"selected-tile-{tile}"},
        "output_devices": ["cuda"],
        **(contract or {}),
    }
    source = example()
    return replace(
        source,
        executables={"update_impl": f"frozen-{tile}".encode()},
        task_definitions=tuple(
            replace(row, compatibility_digest=digest(identity))
            if row.profile_key == "update_impl"
            else row
            for row in source.task_definitions
        ),
        bindings={
            "compiled_device_index": 0,
            "finalized_tasks": {"update_impl": identity},
            "inductor_kernels": {"update_impl": identity["inductor_kernels"]},
            "compiler_memory": {"update_impl": {"known_workspace_bytes": 16}},
            "task_inputs": {"update_impl": {"shape": [512]}},
            "operator_dependencies": {"update_impl": []},
            "preparation_template": {"logical_rank": rank, "buffer": rank},
            "training_operation": {"logical_rank": rank, "buffer": rank},
            "parameter_ownership": [{"names": [f"rank-{rank}"]}],
        },
    )


def profiles(source):
    return Profiles(
        tasks={
            row.profile_key: TaskProfile(1.0, 16, row.compatibility_digest)
            for row in source.task_definitions
        },
        fixed_device_bytes=32,
    )


def combine(first, second):
    values = [profiles(first), profiles(second)]
    return combine_profiles(
        values[1],
        SimpleNamespace(gather=lambda _: [value.to_dict() for value in values]),
    )


def test_shared_binaries_restore_strict_profile_agreement_and_keep_rank_metadata(
    tmp_path,
):
    canonical, local = fixture(256), fixture(128, rank=1)
    with pytest.raises(ValueError, match="different implementations"):
        combine(canonical, local)
    shared = reuse_compiled_tasks(local, canonical)
    assert shared.executables == canonical.executables
    assert shared.task_definitions == canonical.task_definitions
    for name in ("preparation_template", "training_operation", "parameter_ownership"):
        assert shared.bindings[name] == local.bindings[name]
        assert shared.bindings[name] != canonical.bindings[name]
    assert shared.tasks == local.tasks and shared.objects == local.objects
    shared.save(tmp_path / "rank1.ssp")
    restored = Program.load(tmp_path / "rank1.ssp")
    combine(canonical, restored)
    assert restored.executables == canonical.executables
    assert reuse_compiled_tasks(restored, canonical) is restored
    assert local.executables != canonical.executables  # Immutable input artifacts.


@pytest.mark.parametrize(
    "contract",
    [
        {"task": "different-graph-or-resource"},
        {"triton_choices": {"custom_operator": "different_choice"}},
        {"output_devices": ["cpu"]},
        {"operator_preparation": {"providers": ["different-binary"]}},
        {"inductor_kernels": {"different_kernel": "binary"}},
    ],
)
def test_different_computations_or_provider_choices_are_not_replaced(contract):
    canonical, local = fixture(256), fixture(128, rank=1, contract=contract)
    assert reuse_compiled_tasks(local, canonical) is local
    with pytest.raises(ValueError, match="different implementations"):
        combine(canonical, local)


@pytest.mark.parametrize(
    "field", ["compiler_memory", "task_inputs", "operator_dependencies"]
)
def test_different_physical_inputs_and_requirements_are_not_replaced(field):
    local = fixture(128, rank=1)
    local = replace(
        local, bindings={**local.bindings, field: {"update_impl": "different"}}
    )
    assert reuse_compiled_tasks(local, fixture(256)) is local


@pytest.mark.parametrize("change", ["target", "device", "unknown", "unproved"])
def test_heterogeneous_and_unproved_captures_keep_their_own_implementations(change):
    local = fixture(128, rank=1)
    if change == "target":
        local = replace(local, setup={"target": "different"})
    elif change == "device":
        local = replace(local, bindings={**local.bindings, "compiled_device_index": 1})
    elif change == "unknown":
        canonical = replace(fixture(256), executables={})
        assert reuse_compiled_tasks(local, canonical) is local
        return
    else:
        local = replace(local, bindings={**local.bindings, "finalized_tasks": {}})
    assert reuse_compiled_tasks(local, fixture(256)) is local


@pytest.mark.parametrize("field", ["finalized_tasks", "inductor_kernels"])
def test_modified_identity_is_rejected_before_a_binary_is_reused(field):
    local = fixture(128, rank=1)
    saved = local.to_dict()["bindings"]
    if field == "finalized_tasks":
        saved[field]["update_impl"]["task"] = "tampered"
    else:
        saved[field]["update_impl"]["pointwise"] = "tampered"
    local = replace(local, bindings=saved)
    with pytest.raises(ValueError, match="invalid finalized"):
        reuse_compiled_tasks(local, fixture(256))


@pytest.mark.parametrize("resume", [False, True])
def test_build_shares_frozen_binaries_and_refreshes_local_allocation_evidence(
    monkeypatch, tmp_path, resume
):
    canonical, local = fixture(256), fixture(128, rank=1)
    canonical.save(tmp_path / "programs" / "rank0.ssp")
    local_path = tmp_path / "programs" / "rank1.ssp"
    if resume:
        local.save(local_path)
    group = SimpleNamespace(
        count=2, logical_rank=1, rank_map=(1, 0), gather=lambda value: [value, value]
    )
    inventory = SimpleNamespace(
        layouts=SimpleNamespace(
            setup=SimpleNamespace(context=SimpleNamespace(parallelism=group))
        ),
        templates={"rank1": "local-template"},
    )
    group.world_size = 2
    monkeypatch.setattr(job_phases, "load_job", lambda _: (tmp_path, None, inventory))
    monkeypatch.setattr(job_phases, "workers", lambda _: nullcontext(group))
    monkeypatch.setattr(job_phases, "capture_groups", lambda _: (("rank0", "rank1"),))
    monkeypatch.setattr(job_phases, "admitted_inventory", lambda *args: None)
    monkeypatch.setattr(
        job_phases, "program_at", lambda *args: Program.load(local_path)
    )
    monkeypatch.setattr(job_phases, "candidate_title", lambda *args: "Candidate")
    monkeypatch.setattr(job_phases, "describe_candidate", lambda *args: None)
    captures = []

    def build(*args, rank_map):
        assert args == (None, inventory, "rank1") and rank_map == (1, 0)
        captures.append(True)
        return local

    def coverage(template, program):
        assert template == "local-template"
        assert program.executables == canonical.executables
        assert program.bindings["preparation_template"]["logical_rank"] == 1
        return {
            "executables": digest(
                {key: value.hex() for key, value in program.executables.items()}
            )
        }

    monkeypatch.setattr(capture, "build_template", build)
    monkeypatch.setattr(job_phases, "allocation_coverage", coverage)
    job_phases.build(tmp_path)
    saved = Program.load(local_path)
    assert len(captures) == (0 if resume else 1)
    assert saved.executables == canonical.executables
    assert saved.bindings["allocation_coverage"] == coverage("local-template", saved)
    assert (
        saved.bindings["parameter_ownership"] == local.bindings["parameter_ownership"]
    )
