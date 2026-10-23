import pytest

from baselines.moe_layer.candidates import expanded


@pytest.mark.parametrize("epilogue", [None, "gated", "lse"])
@pytest.mark.parametrize("cooperative", [False, True])
def test_expanded_pool_preserves_old_choices_and_adds_small_tiles(
    epilogue, cooperative
):
    gc = pytest.importorskip("quack.gemm_config")
    old = gc._get_sm90_configs(epilogue, cooperative)
    candidates = expanded(gc._get_sm90_configs, epilogue, cooperative)
    assert set(old).issubset(candidates)
    assert len(candidates) == len(set(candidates))
    assert any(
        c.tile_m == 64 and c.tile_n == 96 and c.cluster_m == c.cluster_n == 1
        for c in candidates
    )
    if epilogue in {"gated", "lse"}:
        assert all(not c.swap_ab for c in candidates)
    if not cooperative:
        assert all(c.pingpong for c in candidates)
