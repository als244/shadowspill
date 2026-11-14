"""Installed operation bindings through real build/profile/plan/cold execution."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from engine_tests.fixtures import external_definition
from engine_tests.fixtures.definition_operations import definitions, example
from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    apply_profiles,
    measure_program,
    plan_program,
)
from shadowspill.planning.training.operations import optimizer_program
from shadowspill.training.preparation.operation_build import build_operation
from shadowspill_client.protocol.operations import OperationKey


@pytest.mark.cuda
def test_definition_bound_operations_in_cold_worker(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    resolved, objective, optimizer = definitions()
    with torch.device("meta"):
        model = resolved.create()
    programs = {}
    for name, key, output_only in (
        ("step", OperationKey("step", "metrics"), False),
        ("backward", OperationKey("forward_backward", "metrics"), False),
        ("masked", OperationKey("step", "metrics"), True),
        ("evaluation", OperationKey("forward", "predictions", "evaluation"), False),
    ):
        with example(resolved, objective, key) as (binding, adapted):
            programs[name] = build_operation(
                model,
                binding,
                adapted,
                optimizer=optimizer,
                output_only=output_only,
                options={"partition": "whole"},
            )
            assert model.training
            saved = programs[name].bindings["training_operation"]
            assert saved["objective"]["source"] == objective.source.to_dict()
            assert saved["contributes_loss"]
    for name, program in programs.items():
        profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
        options = {
            "memory_budgets": MemoryBudgets(2 << 30, 64 << 20),
            "communication_info": CommunicationInfo(1e10, 1e10),
        }
        plan_program(apply_profiles(program, profiles), **options).save(
            tmp_path / f"{name}.ssp"
        )
        if name == "step":
            plan_program(
                apply_profiles(optimizer_program(program), profiles), **options
            ).save(tmp_path / "optim.ssp")

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
