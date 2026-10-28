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
