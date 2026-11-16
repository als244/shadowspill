"""The saved-plan text recipe keeps initialization bounded and data resumable."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from shadowspill.execution.tensors import TensorChunk
from shadowspill_workloads.initialization.decoder import Initialize
from shadowspill_workloads.inputs.text.calls import Inputs
from shadowspill_workloads.models.llama3.config import LLAMA_3_1_ROPE_SCALING, Config
from shadowspill_workloads.models.shared.decoder import RotaryEmbedding


@pytest.mark.parametrize("scaling", [None, LLAMA_3_1_ROPE_SCALING])
def test_initializer_chunks_preserve_tables_norms_and_random_values(scaling):
    config = Config(1, 128, 2, 1, 256, 512, max_seq_len=64, rope_scaling=scaling)
    initialize = Initialize("llama3", config, seed=123)
    initialize.block_elements = 257  # Cross RNG blocks with deliberately odd chunks.

    def fill(name, shape, dtype, width):
        target = torch.empty(math.prod(shape), dtype=dtype)
        for start in range(0, target.numel(), width):
            request = TensorChunk(name, shape, dtype, start, target.numel())
            initialize(request, target[start : start + width])
        return target.reshape(shape)

    rotary = RotaryEmbedding(
        config.head_dim,
        base=config.rope_base,
        capacity=config.max_seq_len,
        scaling=scaling,
    )
    for name, shape, dtype in (
        ("embed.weight", (512, 128), torch.bfloat16),
        ("blocks.0.attn.wq.weight", (128, 128), torch.bfloat16),
        ("final_norm.weight", (128,), torch.bfloat16),
        ("rotary.cosine", (64, 64), torch.float32),
        ("rotary.sine", (64, 64), torch.float32),
    ):
        whole = fill(name, shape, dtype, math.prod(shape))
        chunked = fill(name, shape, dtype, 113)
        torch.testing.assert_close(whole, chunked, rtol=0, atol=0)
        if name.startswith("rotary"):
            torch.testing.assert_close(whole, getattr(rotary, name.split(".")[1]))
        elif "norm" in name:
            assert torch.equal(whole, torch.ones_like(whole))
        else:
            # Hugging Face's N(0, initializer_range) for embeddings and
            # projections; another seed draws other values.
            assert 0.018 < whole.float().std().item() < 0.022
            other = Initialize("llama3", config, seed=124)
            target = torch.empty(math.prod(shape), dtype=dtype)
            other(TensorChunk(name, shape, dtype, 0, target.numel()), target)
            assert not torch.equal(whole, target.reshape(shape))


def test_inputs_use_normalized_calls_and_restore_next_example(tmp_path):
    values = np.arange(513, dtype=np.uint32) % 32
    values.tofile(tmp_path / "train.bin")
    (values + 1).tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 64}))
    options = {
        "data": str(tmp_path),
        "sequence_length": 16,
        "sequences_per_step": 4,
        "sequences_per_example": 2,
    }
    source = Inputs(options)
    assert source.data.next_step == 0
    expected = source.example()
    assert len(expected["args"]) == 2
    assert all(
        k == {"seq_lens": (16, 16), "normalizer": 1 / 64} for k in expected["kwargs"]
    )
    torch.testing.assert_close(next(source), expected)
    state = source.state_dict()
    expected = next(source)
    restored = Inputs(options)
    restored.load_state_dict(state)
    torch.testing.assert_close(next(restored), expected)
    assert not torch.equal(
        source.example()["args"][0][0],
        Inputs(options, split="val").example()["args"][0][0],
    )


@pytest.mark.cuda
@pytest.mark.slow
def test_llama_client_example_resumes_from_cold_saved_plans(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    (tmp_path / "setup.json").write_text(
        json.dumps(
            {
                "client_script": "_llama_example_client.py",
                "cold_llama_workers": True,
                "model_description": "Llama tiny public client application",
                "oracle_description": (
                    "exact resumed losses, schedules and all checkpoint tensors"
                ),
            }
        )
    )
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_catalog_service_server.py")),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=900,
    )
    (tmp_path / "server.stdout").write_text(child.stdout)
    (tmp_path / "server.stderr").write_text(child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["actual_example"] and receipt["exact_checkpoint_continuation"]
    assert receipt["all_server_claims_released"]
