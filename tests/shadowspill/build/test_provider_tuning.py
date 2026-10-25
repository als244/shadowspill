"""Saved Triton choices survive cache loss and reject unseen specializations."""

import copy
import json

import pytest
import torch
import triton
import triton.language as tl
from triton.runtime.autotuner import Autotuner

from shadowspill.pytorch.tuning import capture_choices, use_choices


@triton.autotune(
    configs=[triton.Config({"BLOCK": size}) for size in (64, 128)], key=["N"]
)
@triton.jit
def square(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + indices, indices < N, 0)
    tl.store(Y + indices, value * value, indices < N)


def invoke(x):
    output = torch.empty_like(x)
    square[lambda cfg: (triton.cdiv(x.numel(), cfg["BLOCK"]),)](x, output, x.numel())
    return output


@pytest.mark.cuda
def test_frozen_choices_override_local_cache_and_refuse_missing_keys(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.randn(256, device="cuda")
    with capture_choices() as choices:
        torch.testing.assert_close(invoke(x), x.square())
    choices = json.loads(json.dumps(choices))
    assert len(choices) == 1
    selected = next(iter(choices.values()))["candidate"]
    key = next(iter(square.cache))
    square.cache.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("saved task tried to benchmark kernels")

    monkeypatch.setattr(Autotuner, "_bench", forbidden)
    restored = use_choices(invoke, choices)
    torch.testing.assert_close(restored(x), x.square())
    assert not square.cache
    square.cache[key] = square.configs[1 - selected]
    torch.testing.assert_close(restored(x), x.square())
    assert square.best_config is square.configs[selected]
    assert square.cache[key] is square.configs[1 - selected]
    with pytest.raises(ValueError, match="no finalized configuration"):
        restored(torch.ones(512, device="cuda"))
    invalid = copy.deepcopy(choices)
    next(iter(invalid.values()))["config"]["kwargs"]["BLOCK"] = 999
    with pytest.raises(ValueError, match="candidates differ"):
        use_choices(invoke, invalid)(x)
