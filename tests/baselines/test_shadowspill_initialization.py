"""Checkpoint loading retains wrapped names, bounded ranges and unused-key checks."""

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from baselines.engines.shadowspill.initialize import Initialize
from baselines.engines.shadowspill.settings import Settings
from benchmarking.quickstart.phases.common import write
from shadowspill.artifacts.tensors import TensorChunk
from tests.workloads.test_checkpoints import huggingface_state, materialized, metadata


@pytest.mark.parametrize("family", ["llama3", "qwen3moe"])
def test_wrapped_checkpoint_chunks_reproduce_base_and_initialize_adapters(
    tmp_path, family
):
    source = materialized(f"{family}-tiny")
    payload = huggingface_state(family, source, source.config)
    write(tmp_path / "config.json", metadata(family, source.config))
    save_file(payload, tmp_path / "model.safetensors")
    settings = Settings(model=f"{family}-tiny", checkpoint=str(tmp_path), mode="lora")
    with Initialize(settings) as fill:
        for name, parameter in source.named_parameters():
            actual = torch.empty(parameter.numel(), dtype=parameter.dtype)
            request = TensorChunk(
                "model." + name,
                tuple(parameter.shape),
                parameter.dtype,
                0,
                parameter.numel(),
                tuple(parameter.stride()),
            )
            for offset in range(0, actual.numel(), 4096):
                fill(replace(request, offset=offset), actual[offset : offset + 4096])
            torch.testing.assert_close(
                actual.view_as(parameter), parameter, rtol=0, atol=0
            )
        assert fill.report()["unused"] == []
        factor = torch.empty(128 * 32, dtype=torch.bfloat16)
        fill(
            TensorChunk(
                "model.blocks.0.attn.wq.lora_b",
                (128, 32),
                factor.dtype,
                0,
                factor.numel(),
                (32, 1),
            ),
            factor,
        )
        assert torch.count_nonzero(factor) == 0
    save_file(
        {**payload, "unexpected.weight": torch.ones(1)}, tmp_path / "model.safetensors"
    )
    with Initialize(settings) as fill, pytest.raises(ValueError, match="did not take"):
        fill.report()
