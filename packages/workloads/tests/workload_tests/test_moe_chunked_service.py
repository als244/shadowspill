"""Public packed-text policies execute real chunks with independent update oracles."""

import json

import pytest

from shadowspill_client import BuildOptions, Policy
from workload_tests.test_moe_bf16_service import bf16_options, qualify


@pytest.mark.cuda
@pytest.mark.parametrize("lora", [False, True])
def test_public_chunked_moe_step_split_and_saved_plan_resume(tmp_path, lora):
    options = bf16_options(lora)
    options["build"] = BuildOptions(
        chunking=Policy(
            "shadowspill/text_chunks@1",
            {"token_capacity": 8, "chunk_size": 4, "expert_chunk_size": 4},
        ),
        graph_pair_policy=Policy(
            "shadowspill/text_retention@1", {"name": "cost-input", "chunked": True}
        ),
    ).to_dict()
    options["model_description"] += "; installed four-token execution chunks"
    bundle = qualify(tmp_path, options, lora=lora)
    chunks = set()
    for plan in bundle.plans[0].values():
        for task in plan.program.bindings["task_metadata"].values():
            for region in task.get("execution_regions", {}).get("regions", ()):
                if "row_chunks" in region:
                    chunk = region["row_chunks"]
                    chunks.add((task["kind"], chunk["rows"], chunk["chunk_size"]))
    assert {"forward", "backward"} <= {
        kind for kind, rows, size in chunks if rows == 8 and size == 4
    }
    (tmp_path / "chunks.json").write_text(json.dumps(sorted(chunks), indent=2) + "\n")
