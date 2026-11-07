"""Generated indexing kernels finalize with valid controls, including cold replay."""

import os
import shutil
import subprocess
import sys

import pytest
import torch
from torch import nn

from shadowspill import build_forward


class PackedPositions(nn.Module):
    def forward(self, lengths):
        ends = lengths.cumsum(0)
        tokens = torch.arange(64, device=lengths.device)
        sequence = torch.searchsorted(ends, tokens, right=True)
        return tokens - (ends - lengths)[sequence]


class NormalizeGradient(nn.Module):
    def forward(self, value, gradient):
        x, dy = value.float(), gradient.float()
        inverse = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return (
            (dy - x * (dy * x).mean(-1, keepdim=True) * inverse.square()) * inverse
        ).to(value.dtype)


@pytest.mark.cuda
def test_dependent_indices_finalize_before_profile_and_cold_execution(
    tmp_path, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cache = tmp_path / "build-cache"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache / "triton"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
    program = build_forward(
        PackedPositions(),
        example_args=(torch.tensor([20, 28, 16, 0], dtype=torch.int32),),
    )
    (key,) = program.executables
    assert len(program.bindings["inductor_kernels"][key]) >= 2
    program.save(tmp_path / "program.ssp")
    shutil.rmtree(cache)
    script = """
import sys
import torch
import triton
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from shadowspill import Program
from shadowspill.pytorch.executable import restore_executable

def forbidden(*args, **kwargs):
    raise AssertionError('execution attempted kernel tuning')

def compilation(**event):
    if not event['cache_hit']:
        raise AssertionError('execution compiled ' + event['src'].name)

CachingAutotuner.autotune_to_one_config = forbidden
triton.knobs.compilation.listener = compilation
program = Program.load(sys.argv[1])
key, = program.executables
function = restore_executable(program.executables[key])
for lengths in ([20, 28, 16, 0], [7, 13, 25, 19]):
    expected = torch.cat([torch.arange(n, device='cuda') for n in lengths])
    output, = function(torch.tensor(lengths, dtype=torch.int32, device='cuda'))
    torch.testing.assert_close(output, expected)
"""
    child = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            script,
            str(tmp_path / "program.ssp"),
        ],
        env={
            **os.environ,
            "TRITON_CACHE_DIR": str(tmp_path / "cold-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cold-inductor"),
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.cuda
def test_saved_reduction_does_not_expand_tiles_or_retune_on_cold_load(
    tmp_path, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cache = tmp_path / "build-cache"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache / "triton"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
    inputs = tuple(torch.randn(1024, 4096, dtype=torch.bfloat16) for _ in range(2))
    program = build_forward(NormalizeGradient(), example_args=inputs)
    program.save(tmp_path / "program.ssp")
    torch.save(inputs, tmp_path / "inputs.pt")
    shutil.rmtree(cache)
    script = """
import sys
from pathlib import Path
import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from shadowspill import Program
from shadowspill.pytorch.executable import restore_executable
from tests.shadowspill.build.test_inductor_tuning import NormalizeGradient

def forbidden(*args, **kwargs):
    raise AssertionError('saved reduction attempted tile expansion or tuning')

CachingAutotuner._iter_rblock_scale_candidates = forbidden
CachingAutotuner.autotune_to_one_config = forbidden
CachingAutotuner.coordinate_descent_tuning = forbidden
CachingAutotuner._combo_sequential_autotune = forbidden
root = Path(sys.argv[1])
program = Program.load(root / 'program.ssp')
key, = program.executables
function = restore_executable(program.executables[key])
inputs = tuple(x.cuda() for x in torch.load(root / 'inputs.pt', weights_only=True))
for values in (inputs, tuple(x * 0.5 for x in inputs)):
    output, = function(*values)
    expected = NormalizeGradient()(*values)
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
"""
    child = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            script,
            str(tmp_path),
        ],
        env={
            **os.environ,
            "TRITON_CACHE_DIR": str(tmp_path / "cold-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cold-inductor"),
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
