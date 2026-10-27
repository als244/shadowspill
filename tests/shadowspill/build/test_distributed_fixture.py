"""Qualify the distributed fixture's stage choices before using an allocation."""

import pytest
import torch

from shadowspill import MemoryBudgets, ProfiledProgram, plan_program
from shadowspill.artifacts.tensors import registration_name, shared_model_aliases
from shadowspill.execution.tensor_regions import tensor_regions
from tests.shadowspill.execution._distributed_phases import (
    build,
    profile,
    select_variants,
)
from tests.shadowspill.offline._communication import estimated_communication


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["forward_backward", "step"])
def test_distributed_variants_keep_terminal_save_and_recompute_earlier_stages(
    tmp_path, kind
):
    if not torch.cuda.is_available():
        pytest.skip("requires one representative GPU")
    build(tmp_path, kind)
    profile(tmp_path)
    profiled = ProfiledProgram.load(tmp_path / "profiled.ssp")
    for variant in ("save", "recompute"):
        selected = select_variants(profiled, variant)
        result = plan_program(
            selected,
            memory_budgets=MemoryBudgets(
                profiled.fixed_device_bytes + (32 << 20), 64 << 20
            ),
            communication_info=estimated_communication(
                selected.program, topology={"node": (0, 1)}
            ),
        )
        variants = {choice.option_id for choice in result.selections}
        assert variant in variants
        assert "save" in variants
        assert not torch.distributed.is_initialized()


@pytest.mark.cuda
@pytest.mark.parametrize("cutoff", [False, True])
def test_frozen_fp8_shards_keep_physical_dtypes_and_shared_evaluation(tmp_path, cutoff):
    if not torch.cuda.is_available():
        pytest.skip("requires one representative GPU")
    build(tmp_path, "step", "fp8_cutoff" if cutoff else "fp8")
    profile(tmp_path)
    profiled = ProfiledProgram.load(tmp_path / "profiled.ssp")
    program = profiled.program
    objects = {o.object_id: o for o in program.objects}
    entries = {
        registration_name(row): row["object_id"]
        for row in program.bindings["registrations"]
    }
    data, scale = entries["weight._data"], entries["weight._scale"]
    geometry = program.bindings["model_tensor_geometry"]
    assert geometry[data]["dtype"] == "torch.float8_e4m3fn"
    assert geometry[scale]["dtype"] == "torch.float32"
    shards = program.bindings["weight_shards"]
    assert shards[objects[data].alias_group_id]["shard_bytes"] == 8192
    if cutoff:
        assert objects[scale].alias_group_id not in shards
        assert objects[entries["bias"]].alias_group_id not in shards
        assert not program.bindings["gradient_shards"]
        assert not program.bindings["owned_parameter_views"]
        bias = program.bindings["gradient_templates"][0]
        assert [p["operation"] for p in bias["phases"]] == ["all_reduce"]
    else:
        assert shards[objects[scale].alias_group_id]["shard_bytes"] == 4
        assert shards[objects[scale].alias_group_id]["logical_bytes"] == 4
    for rank in range(2):
        regions = tensor_regions(program, rank)
        assert (regions[data].offset, regions[data].length) == (rank * 8192, 8192)
        if cutoff:
            assert scale not in regions  # full local scale; no gather or padding
        else:
            assert regions[scale].length == 1 - rank
    evaluation = ProfiledProgram.load(tmp_path / "evaluation-profiled.ssp")
    assert len(shared_model_aliases(program, evaluation.program)) == 3
    result = plan_program(
        profiled,
        memory_budgets=MemoryBudgets(
            profiled.fixed_device_bytes + (32 << 20), 64 << 20
        ),
        communication_info=estimated_communication(program, topology={"node": (0, 1)}),
    )
    assert result.timeline.task_intervals
    assert result.simulation_only_reasons  # communication evidence is estimated
    assert not torch.distributed.is_initialized()
