"""A composed row task follows the ordinary saved-phase execution path."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from shadowspill import (
    CommunicationInfo,
    Program,
    apply_profiles,
    measure_program,
)
from shadowspill.build import build_forward, build_forward_backward, build_step
from shadowspill.pytorch.chunking import AxisChunks
from shadowspill.pytorch.partition import IsolateOperations
from shadowspill.task.inputs import TaskInputRole
from tests.shadowspill.build.test_training import Network, ResidualNetwork, objective
from tests.shadowspill.execution._oracle import oracle


class ChunkRows:
    def __init__(self, variant):
        self.variant = variant

    def configure_stage(self, stage):
        if stage.terminal:
            return None
        axes = {
            i: 1
            for i, (value, role) in enumerate(
                zip(stage.inputs, stage.input_roles, strict=True)
            )
            if value is not None
            and len(value.shape) == 3
            and role not in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}
        }
        return AxisChunks(7, axes, variants=(self.variant,)) if axes else None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "network, with_optimizer, variant",
    [
        pytest.param(Network, None, "save", id="linear-forward"),
        pytest.param(ResidualNetwork, None, "save", id="residual-forward"),
        pytest.param(Network, False, "recompute", id="linear-backward-recompute"),
        pytest.param(ResidualNetwork, False, "save", id="residual-backward-save"),
        pytest.param(Network, True, "save", id="linear-step-save"),
        pytest.param(ResidualNetwork, True, "recompute", id="residual-step-recompute"),
    ],
)
def test_chunked_plan_reloads_and_matches_repeated_training(
    tmp_path, variant, with_optimizer, network
):
    # Keep both topologies through each public builder, and both retention
    # variants through backward and optimizer execution. A forward-only Plan
    # has no retention decision, so repeating it for RECOMPUTE adds no coverage.
    # The CPU VJP/accumulation matrices cover the combinatorial math; these
    # cases test saved-phase integration, mixed extents and tied parameters.
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = network(tied=True)
    examples = [
        ((torch.randn(2, rows, 8),), {"labels": torch.randn(2, rows, 8)})
        for rows in (17, 14)
    ]
    with torch.device("meta"):
        template = network(tied=True)
    common = dict(
        objective=objective, example_calls=examples, chunking=ChunkRows(variant)
    )
    if with_optimizer is None:
        examples = [(examples[0][0], {})]
        program = build_forward(
            template, example_args=examples[0][0], chunking=ChunkRows(variant)
        )
    elif with_optimizer:
        program = build_step(
            template,
            optimizer=torch.optim.AdamW(template.parameters(), foreach=False),
            **common,
        )
    else:
        program = build_forward_backward(template, **common)
    _verify_saved(
        program,
        model,
        examples,
        "forward"
        if with_optimizer is None
        else "step"
        if with_optimizer
        else "forward_backward",
        tmp_path,
    )


class SingleRows:
    def configure_stage(self, stage):
        axes = {
            i: 0
            for i, role in enumerate(stage.input_roles)
            if role is not TaskInputRole.PARAMETER
        }
        return AxisChunks(1, axes)


class SharedWeightBoundary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8) / 8)

    def forward(self, x):
        shared = self.weight.t().clone()
        hidden = (x @ self.weight).sin()
        return hidden @ shared


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_shared_boundary_cotangent_is_not_repeated_per_chunk(tmp_path, variant):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = SharedWeightBoundary()
    with torch.device("meta"):
        template = SharedWeightBoundary()
    examples = [
        ((torch.randn(2, rows, 8),), {"labels": torch.randn(2, rows, 8)})
        for rows in (17, 14)
    ]
    program = build_forward_backward(
        template,
        objective=objective,
        example_calls=examples,
        partition=IsolateOperations(("aten.sin.default",)),
        chunking=ChunkRows(variant),
    )
    assert any(
        "sum" in metadata.get("row_chunks", {}).get("inputs", ())
        for metadata in program.bindings["task_metadata"].values()
    )
    _verify_saved(program, model, examples, "forward_backward", tmp_path)


class HostControlNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, x):
        value = self.projection(x).sin()
        control = torch.tensor([0, 7, 17], dtype=torch.int64, device="cpu")
        return value, control


@pytest.mark.cuda
def test_chunked_plan_preserves_host_control_output_locations(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = HostControlNetwork()
    with torch.device("meta"):
        template = HostControlNetwork()
    examples = [((torch.randn(2, 17, 8),), {})]
    program = build_forward(
        template, example_args=examples[0][0], chunking=ChunkRows("save")
    )
    assert any(
        program.bindings["compiler_memory"][key]["off_device_output_leaves"]
        for key, metadata in program.bindings["task_metadata"].items()
        if "row_chunks" in metadata
    )
    _verify_saved(
        program, model, examples, "forward", tmp_path, host_output_leaves=(1,)
    )


@pytest.mark.cuda
def test_many_chunks_reserve_retirements_for_reused_allocation_generations(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = torch.nn.Linear(8, 8)
    with torch.device("meta"):
        template = torch.nn.Linear(8, 8)
    examples = [((torch.randn(1025, 8),), {})]
    program = build_forward(
        template, example_args=examples[0][0], chunking=SingleRows()
    )
    _verify_saved(program, model, examples, "forward", tmp_path)


def _verify_saved(
    program,
    model,
    examples,
    kind,
    tmp_path,
    *,
    host_output_leaves=(),
    bf16_reduction=None,
):
    program.save(tmp_path / "program.ssp")
    program = Program.load(tmp_path / "program.ssp")
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.003)
    profiles.save(tmp_path / "profiles.ssp")
    apply_profiles(program, profiles).save(tmp_path / "profiled.ssp")
    CommunicationInfo(fetch_bytes_per_second=1e10, evict_bytes_per_second=1e10).save(
        tmp_path / "communication.ssp"
    )
    (tmp_path / "settings.json").write_text(
        json.dumps({"execution_bytes": 2 << 30, "spill_bytes": 64 << 20})
    )
    planning = subprocess.run(
        [sys.executable, "-m", "qualification.pipeline.plan", str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert planning.returncode == 0, planning.stdout + planning.stderr
    expected = oracle(model, examples, kind)
    expected["host_output_leaves"] = host_output_leaves
    if bf16_reduction is not None:
        expected["bf16_reduction"] = bf16_reduction
    torch.save(expected, tmp_path / "oracle.pt")
    worker = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_execute.py")), str(tmp_path)],
        env={
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cold-inductor"),
            "TRITON_CACHE_DIR": str(tmp_path / "cold-triton"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert worker.returncode == 0, worker.stdout + worker.stderr
