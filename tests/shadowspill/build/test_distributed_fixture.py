"""Qualify the distributed fixture's stage choices before using an allocation."""

import pytest
import torch

from shadowspill import MemoryBudgets, ProfiledProgram, plan_program
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
