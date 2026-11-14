"""Installed operation bindings through real build/profile/plan/cold execution."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from engine_tests.fixtures import external_definition
from engine_tests.fixtures.definition_operations import definitions, example
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill import CommunicationInfo
from shadowspill.training.preparation.capture import build_template
from shadowspill.training.preparation.family import plan_family
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.operation_build import build_operation
from shadowspill.training.preparation.profile import profile_template
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import (
    BuildCandidates,
    Memory,
    PlanningOptions,
    Preparation,
    ProfilingOptions,
)
from shadowspill_client.protocol.operations import OperationKey


@pytest.mark.cuda
def test_definition_bound_operations_in_cold_worker(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    resolved, objective, _ = definitions()
    registry = installed()
    requested = {**arguments(), "memory": Memory(device_gib=2, spill_gib=0.0625)}
    setup = resolve_training(
        registry,
        **requested,
        operations={
            "step": ["metrics"],
            "forward_backward": ["metrics"],
            "forward": ["predictions"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[2, 4]),
            planning=PlanningOptions(microbatch_counts=[1, 2]),
            profiling=ProfilingOptions(
                conditioning_seconds=0.001, measurement_seconds=0.002
            ),
        ),
    )
    inventory = capture_inventory(prepare_layouts(setup))
    assert len(inventory.templates) == 6
    measured = {}
    for key, template in inventory.templates.items():
        program = build_template(registry, inventory, key)
        assert program.bindings["operation_mode"] == template.binding.contract.key.mode
        saved = program.bindings["training_operation"]
        assert saved["objective"]["source"] == objective.source.to_dict()
        assert saved["contributes_loss"]
        program.save(tmp_path / f"capture-{key}.ssp")
        profiled = profile_template(inventory, key, program)
        measured[key] = profiled
        assert profiled.profiles.metadata["preparation_template"] == key
        assert (
            profiled.profiles.metadata["profiling_scenario"]
            == template.signature()["scenario"]
        )
        profiled.save(tmp_path / f"profiled-{key}.ssp")
    family = plan_family(
        inventory,
        measured,
        {key: CommunicationInfo(1e10, 1e10) for key in measured},
    )
    assert len(family.coverage) == 3 and len(family.plans[0]) == 14
    assert all(len(rows) == 3 for rows in family.families.values())
    assert family.memory[0].execution_bytes <= 2 << 30
    assert family.memory[0].spill_bytes <= 64 << 20

    def full_buffer(operation):
        return next(
            row.workers[0]
            for row in family.families[operation]
            if row.source.layout.candidate.schedule.buffers[0].max_datums == 4
        )

    step = full_buffer(OperationKey("step", "metrics"))
    backward = full_buffer(OperationKey("forward_backward", "metrics"))
    evaluation = full_buffer(OperationKey("forward", "predictions", "evaluation"))
    used = set()
    for name, identity in (
        ("step", step.main),
        ("backward", backward.main),
        ("masked", backward.masked),
        ("evaluation", evaluation.main),
        ("optim", backward.optimizer),
    ):
        family.plans[0][identity].save(tmp_path / f"{name}.ssp")
        used.add(identity)
    extra = sorted(family.plans[0].keys() - used)
    for identity in extra:
        family.plans[0][identity].save(tmp_path / f"{identity}.ssp")
    (tmp_path / "extra-plans.json").write_text(json.dumps(extra))

    reference = resolved.create()
    initial = {
        name: value.detach().clone() for name, value in reference.named_parameters()
    }
    oracle = {"initial": initial}
    eager = torch.optim.SGD(reference.parameters(), lr=0.03, foreach=False)
    with example(resolved, objective, "step") as (binding, adapted):
        args = binding.arguments(adapted, {"temperature": 2.0})
        loss = binding.objective(reference, *args).loss
        oracle["step_loss"] = loss.detach().clone()
        loss.backward()
        for p in reference.parameters():
            p.grad.mul_(0.5)
        eager.step()
        oracle["after_step"] = {
            n: p.detach().clone() for n, p in reference.named_parameters()
        }
        eager.zero_grad(set_to_none=True)
        for temperature in (0.5, 1.0):
            binding.objective(
                reference, *binding.arguments(adapted, {"temperature": temperature})
            ).loss.backward()
        oracle["gradients"] = {
            n: p.grad.detach().clone() for n, p in reference.named_parameters()
        }
        for p in reference.parameters():
            p.grad.mul_(-0.5)
        eager.param_groups[0]["lr"] = 0.02
        eager.step()
        oracle["after_split"] = {
            n: p.detach().clone() for n, p in reference.named_parameters()
        }
    torch.save(oracle, tmp_path / "oracle.pt")
    worker = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_definition_operations.py")),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cold-inductor"),
            "TRITON_CACHE_DIR": str(tmp_path / "cold-triton"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert worker.returncode == 0, worker.stdout + worker.stderr
    assert "Qualified definition-bound operations" in worker.stdout


def ignore_temperature(model, inputs, loss_inputs, loss_params):
    return external_definition.classification_loss(
        model, inputs, loss_inputs, {"temperature": 1.0}
    )


def ignore_lr(model, config, controls):
    return torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)


@pytest.mark.cuda
def test_preparation_rejects_controls_without_real_captured_reads():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    resolved, objective, optimizer = definitions()
    with torch.device("meta"):
        model = resolved.create()
    with example(resolved, objective, "forward_backward") as (binding, adapted):
        binding.computation = ignore_temperature
        with pytest.raises(ValueError, match="objective control is not consumed"):
            build_operation(model, binding, adapted, options={"partition": "whole"})
    optimizer = replace(
        optimizer, definition=replace(optimizer.definition, create=ignore_lr)
    )
    with (
        example(resolved, objective, "step") as (binding, adapted),
        pytest.raises(ValueError, match="optimizer control lacks"),
    ):
        build_operation(
            model,
            binding,
            adapted,
            optimizer=optimizer,
            options={"partition": "whole"},
        )
    assert model.training


class MutatingClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 3)
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, pixels):
        self.calls.add_(1)
        return self.head(pixels.flatten(start_dim=1))


@pytest.mark.cuda
def test_evaluation_and_undeclared_buffer_mutations_fail_admission():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    resolved, objective, _ = definitions()
    with torch.device("meta"):
        model = MutatingClassifier()
    model.eval()
    key = OperationKey("forward", "predictions", "evaluation")
    with (
        example(resolved, objective, key) as (binding, adapted),
        pytest.raises(ValueError, match="evaluation must preserve"),
    ):
        build_operation(model, binding, adapted, options={"partition": "whole"})
    key = OperationKey("forward", "predictions", "training")
    with (
        example(resolved, objective, key) as (binding, adapted),
        pytest.raises(ValueError, match="undeclared model state"),
    ):
        build_operation(model, binding, adapted, options={"partition": "whole"})
    declared = replace(
        resolved, definition=replace(resolved.definition, mutable_buffers=("calls",))
    )
    with example(declared, objective, key) as (binding, adapted):
        program = build_operation(
            model, binding, adapted, options={"partition": "whole"}
        )
        assert any(task.mutations for task in program.tasks)
    assert not model.training
