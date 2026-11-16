"""Aligned LoRA public training must use prepared destination-writing kernels."""

import json

import pytest
import torch

from shadowspill_client import BuildOptions, Policy
from shadowspill_workloads.definitions.moe_provider import DESTINATION_CONTRACTS
from workload_tests.test_moe_bf16_service import bf16_options, qualify


@pytest.mark.cuda
def test_public_aligned_lora_chunks_accumulate_into_prepared_destinations(tmp_path):
    options = bf16_options(True, lora_rank=16)
    options["build"] = BuildOptions(
        chunking=Policy(
            "shadowspill/text_chunks@1",
            {"token_capacity": 8, "chunk_size": 4, "expert_chunk_size": 4},
        ),
        graph_pair_policy=Policy(
            "shadowspill/text_retention@1", {"name": "cost-input", "chunked": True}
        ),
    ).to_dict()
    bundle = qualify(tmp_path, options, lora=True)
    calls, chunks = set(), set()
    for plan in bundle.plans[0].values():
        bindings = plan.program.bindings
        for task in bindings["allocation_coverage"]["tasks"].values():
            calls.update(set(task["contracts"]) & DESTINATION_CONTRACTS.keys())
        for task in bindings["task_metadata"].values():
            for region in task.get("execution_regions", {}).get("regions", ()):
                if "row_chunks" in region:
                    chunk = region["row_chunks"]
                    chunks.add((task["kind"], chunk["rows"], chunk["chunk_size"]))
    if torch.cuda.get_device_capability()[0] in (9, 10):
        assert calls == DESTINATION_CONTRACTS.keys()
    else:
        assert not calls
    assert ("backward", 8, 4) in chunks
    (tmp_path / "destinations.json").write_text(
        json.dumps({"calls": sorted(calls), "chunks": sorted(chunks)}, indent=2) + "\n"
    )
