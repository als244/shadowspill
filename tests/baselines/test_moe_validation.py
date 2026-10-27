import pytest
import torch

from baselines.moe_layer.validation import all_finite


@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf")])
def test_finite_check_bounds_scratch_and_checks_every_element(
    monkeypatch, noncontiguous, invalid
):
    value = torch.ones(13, 17)
    if invalid is not None:
        value[12, 16] = invalid
    if noncontiguous:
        value = value.t()
    pointer = value.untyped_storage().data_ptr()
    original = torch.isfinite
    visited = 0

    def bounded(part):
        nonlocal visited
        assert part.numel() <= 16
        assert part.untyped_storage().data_ptr() == pointer
        visited += part.numel()
        return original(part)

    monkeypatch.setattr(torch, "isfinite", bounded)
    assert all_finite(value, chunk_elements=16) is (invalid is None)
    assert visited == value.numel()


def test_empty_scalar_and_integer_values():
    assert all_finite(torch.empty(0))
    assert all_finite(torch.tensor(1.0))
    assert not all_finite(torch.tensor(float("nan")))
    assert all_finite(torch.arange(100))
    with pytest.raises(ValueError):
        all_finite(torch.ones(2), chunk_elements=0)
