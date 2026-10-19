"""Transfer-sharded code and its profile identity use the same order policy."""

from types import SimpleNamespace

import pytest
from torch._inductor import config

from shadowspill.pytorch import distributed
from shadowspill.pytorch.compilation.inductor.compiler import task_compiler_patches
from shadowspill.pytorch.profiling import environment


@pytest.mark.parametrize("sharded", [False, True])
def test_peak_reordering_policy_and_profile_identity(monkeypatch, sharded):
    bound = SimpleNamespace(transfer_sharding={"weight": object()} if sharded else {})
    monkeypatch.setattr(distributed, "current", lambda: bound)
    monkeypatch.setattr(environment, "distributed_preparation", lambda: None)
    monkeypatch.setattr(environment, "provider_version", lambda: "test")
    monkeypatch.setattr(
        environment.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="test", major=9, minor=0),
    )
    with config.patch(reorder_for_peak_memory=True):
        patches = task_compiler_patches()
        assert patches["shape_padding"] is False
        assert patches.get("reorder_for_peak_memory", True) is (not sharded)
        profile = environment.profile_environment(device_ordinal=0, provider_id="test")
        assert f":reorder_for_peak_memory={int(not sharded)}" in profile.compiler_id
        with config.patch(patches):
            assert config.reorder_for_peak_memory is (not sharded)
        assert config.reorder_for_peak_memory is True


def test_local_compilation_preserves_requested_order_policy(monkeypatch):
    monkeypatch.setattr(distributed, "current", lambda: None)
    assert task_compiler_patches() == {"shape_padding": False}
