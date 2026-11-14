"""Changed backend configuration survives saved phases and is required at runtime."""

import pytest
import torch

from engine_tests.build.test_training import objective
from engine_tests.execution.test_chunked_saved_plan import _verify_saved
from engine_tests.execution.test_region_saved_plan import RegionNetwork
from engine_tests.pytorch.chunking.test_regions import RegionChunks
from shadowspill import measure_program
from shadowspill.build import build_step
from shadowspill.pytorch import ChunkRegions
from shadowspill.pytorch.partition import IsolateOperations


@pytest.mark.cuda
def test_alternate_numeric_configuration_nested_training_and_mismatch_refusal(
    tmp_path, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    matmul = torch.backends.cuda.matmul
    monkeypatch.setattr(matmul, "allow_bf16_reduced_precision_reduction", False)
    model = RegionNetwork(lora=True)
    with torch.device("meta"):
        template = RegionNetwork(lora=True)
    examples = [
        ((torch.randn(2, 17, 8),), {"labels": torch.randn(2, 17, 8)}) for _ in range(2)
    ]
    program = build_step(
        template,
        optimizer=torch.optim.AdamW(template.parameters(), foreach=False),
        objective=objective,
        example_calls=examples,
        partition="auto",
        chunking=ChunkRegions(
            IsolateOperations(
                ("aten._softmax.default", "aten._softmax_backward_data.default"),
                base="whole",
            ),
            RegionChunks(17, 2),
            variants=("recompute",),
        ),
    )
    matmul.allow_bf16_reduced_precision_reduction = True
    with pytest.raises(RuntimeError, match="allow_bf16_reduced_precision_reduction"):
        measure_program(program, warmup_seconds=0, measure_seconds=0.001)
    matmul.allow_bf16_reduced_precision_reduction = False
    _verify_saved(program, model, examples, "step", tmp_path, bf16_reduction=False)
