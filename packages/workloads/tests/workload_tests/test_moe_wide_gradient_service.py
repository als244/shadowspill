"""Public BF16 computation with FP32 gradients, master weights and AdamW state."""

import pytest

from shadowspill_client import BuildOptions, Policy
from workload_tests.test_moe_bf16_service import bf16_options, qualify


@pytest.mark.cuda
@pytest.mark.parametrize("lora", [False, True])
def test_public_bf16_compute_fp32_gradients_step_split_and_resume(tmp_path, lora):
    options = bf16_options(lora, grad_dtype="float32")
    options["build"] = BuildOptions(
        chunking=Policy(
            "shadowspill/text_chunks@1",
            {"token_capacity": 8, "chunk_size": 4, "expert_chunk_size": 4},
        ),
        graph_pair_policy=Policy(
            "shadowspill/text_retention@1", {"name": "cost-input", "chunked": True}
        ),
    ).to_dict()
    qualify(tmp_path, options, lora=lora)
