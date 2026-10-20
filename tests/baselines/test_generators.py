"""The recipe generators expose their frameworks' offload options by the
frameworks' own names."""

import pytest

from baselines.automodel.prepare import configuration
from baselines.megatron_core.prepare import RECIPE_LAYERS, recipe_overrides


def test_automodel_takes_the_fsdp2_offload_policy():
    value, metadata = configuration(routing="real", cpu_offload=True)
    policy = value["distributed"]["offload_policy"]
    assert policy["_target_"] == "torch.distributed.fsdp.CPUOffloadPolicy"
    assert policy["pin_memory"] is True and metadata["cpu_offload"] is True
    plain, _ = configuration(routing="real")
    assert "offload_policy" not in plain["distributed"]


def test_megatron_overrides_name_its_offloads(tmp_path):
    base = {
        "output": tmp_path,
        "tensor_parallel": 1,
        "recompute": False,
        "cpu_offload_layers": 8,
        "cpu_offload_weights": False,
        "optimizer_cpu_offload": True,
        "optimizer_offload_fraction": 0.5,
    }
    overrides = recipe_overrides(**base)
    assert "model.cpu_offloading=true" in overrides
    assert "model.cpu_offloading_num_layers=8" in overrides
    assert "model.cpu_offloading_weights=false" in overrides
    assert "optimizer.optimizer_cpu_offload=true" in overrides
    assert "optimizer.optimizer_offload_fraction=0.5" in overrides
    assert "model.recompute_granularity=null" in overrides
    with pytest.raises(ValueError, match="recomputation"):
        recipe_overrides(**{**base, "recompute": True})
    with pytest.raises(ValueError):
        recipe_overrides(**{**base, "cpu_offload_layers": RECIPE_LAYERS})
    with pytest.raises(ValueError):
        recipe_overrides(**{**base, "optimizer_offload_fraction": 2.0})
    none = recipe_overrides(
        **{**base, "cpu_offload_layers": 0, "optimizer_cpu_offload": False}
    )
    assert not any("offload" in item for item in none)
