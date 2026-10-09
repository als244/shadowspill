"""Preparation restores default streams without initializing unrelated devices."""

import random

import numpy as np
import pytest
import torch

from shadowspill.pytorch._rng import preserve_rng


@pytest.mark.parametrize("fail", [False, True])
def test_preserve_rng_restores_cpu_streams_on_success_and_failure(fail, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU preservation must not touch CUDA")

    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    random.seed(1009)
    np.random.seed(1009)
    torch.manual_seed(1009)

    def sample():
        return random.random(), np.random.rand(), torch.rand(5)

    with preserve_rng(torch.device("cpu")):
        expected = sample()
    try:
        with preserve_rng(torch.device("cpu")):
            sample()
            sample()
            if fail:
                raise RuntimeError("expected setup failure")
    except RuntimeError as error:
        assert str(error) == "expected setup failure"
    actual = sample()
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
