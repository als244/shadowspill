"""Prepare the framework-free inference decoder against real bounded archives."""

import numpy as np
import pytest

from engine_tests.checkpoint.test_conversion import WEIGHT, HostRank, family
from engine_tests.execution._weights_service_client import read_weights
from shadowspill.checkpoint import write_weights


@pytest.mark.parametrize("representation", ["master", "compute"])
@pytest.mark.parametrize("frozen", [False, True])
def test_numpy_inference_decoder_preserves_tied_weights_and_exact_frozen_base(
    tmp_path, representation, frozen
):
    state = family(1, 0, frozen=frozen)[0]
    worker = HostRank(state, initialize=True)
    (tmp_path / "ranks").mkdir()
    with (tmp_path / "ranks/rank-00000.ssp").open("wb") as stream:
        write_weights(
            worker,
            stream,
            weights=representation,
            chunk_bytes=16,
            frozen_state_id="exact-base" if frozen else None,
        )
    base = (
        {name: WEIGHT.numpy() for name in ("base.weight", "base.tied")}
        if frozen
        else {}
    )
    _, values = read_weights(tmp_path, base)
    expected = WEIGHT if representation == "master" else WEIGHT.bfloat16().float()
    for name in ("linear.weight", "head.weight"):
        np.testing.assert_array_equal(values[name], expected.numpy())
    for name in base:
        np.testing.assert_array_equal(values[name], base[name])
