"""One capture/compiled Plan accepts different sequence boundaries and empty slots."""

import os
import subprocess
import sys
from itertools import accumulate
from pathlib import Path

import pytest
import torch
from torch.export.graph_signature import InputKind

from shadowspill.pytorch.capture.export import capture_forward
from shadowspill_workloads.models.shared.decoder import packed_metadata

GEOMETRIES = ((65, 0, 64, 0), (1, 127, 1, 0), (129, 0, 0, 0), (0, 65, 0, 64))


class Metadata(torch.nn.Module):
    def forward(self, tokens, lengths):
        import mlops

        attention = packed_metadata(tokens, lengths, capacity=129)
        cumulative, chunks = mlops.prepare_packed_sequence_metadata(lengths, tokens)
        return attention.positions, attention.cu_seqlens, cumulative, chunks


def arguments(lengths, *, device="cpu"):
    return (
        torch.arange(129, device=device).reshape(1, 129),
        torch.tensor(lengths, dtype=torch.int32, device=device),
    )


def expected(lengths):
    offsets = [0, *accumulate(lengths)]
    chunks = [
        (sequence, chunk)
        for sequence, length in enumerate(lengths)
        for chunk in range((length + 63) // 64)
    ]
    # Extra rows belong to the final empty sentinel sequence.
    bound = (sum(lengths) + 63) // 64 + len(lengths)
    chunks.extend((len(lengths), index) for index in range(bound - len(chunks)))
    return (
        torch.tensor([position for length in lengths for position in range(length)]),
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor([*offsets, offsets[-1]], dtype=torch.int64),
        torch.tensor(chunks, dtype=torch.int64),
    )


def test_export_keeps_lengths_as_a_user_tensor_and_recomputes_all_metadata():
    captured = capture_forward(Metadata(), arguments(GEOMETRIES[0]))
    exported = captured.exported_program
    assert [spec.kind for spec in exported.graph_signature.input_specs] == [
        InputKind.USER_INPUT,
        InputKind.USER_INPUT,
    ]
    assert not exported.constants
    graph = exported.module()
    for lengths in GEOMETRIES:
        torch.testing.assert_close(
            graph(*arguments(lengths)), expected(lengths), rtol=0, atol=0
        )
    assert not torch.equal(expected(GEOMETRIES[0])[0], expected(GEOMETRIES[1])[0])
    assert not torch.equal(expected(GEOMETRIES[0])[3], expected(GEOMETRIES[1])[3])


@pytest.mark.cuda
def test_inductor_compiles_once_for_changing_length_tensor_values():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    compilations = []

    def backend(graph, values):
        compilations.append(graph)
        return torch._inductor.compile(graph, values)

    compiled = torch.compile(Metadata(), fullgraph=True, dynamic=False, backend=backend)
    for lengths in GEOMETRIES:
        actual = tuple(
            value.cpu() for value in compiled(*arguments(lengths, device="cuda"))
        )
        torch.testing.assert_close(actual, expected(lengths), rtol=0, atol=0)
    assert len(compilations) == 1


@pytest.mark.cuda
def test_saved_plan_reuses_runtime_lengths_in_a_worker_without_preparation(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from engine_tests.execution.helpers import save_plan
    from shadowspill import build_forward

    program = build_forward(
        Metadata(), example_args=arguments(GEOMETRIES[0]), partition="whole"
    )
    save_plan(program, tmp_path)
    torch.save(
        {
            "calls": [arguments(lengths) for lengths in GEOMETRIES],
            "expected": [expected(lengths) for lengths in GEOMETRIES],
        },
        tmp_path / "oracle.pt",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_runtime_sequence_metadata.py")),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "TRITON_CACHE_DIR": str(tmp_path / "empty-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "empty-inductor"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "worker.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verified four packings with one saved Plan" in result.stdout
