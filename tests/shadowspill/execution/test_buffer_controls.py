"""Model controls write only declared buffer views and preserve omitted values."""

from types import SimpleNamespace

import pytest
import torch

from shadowspill.execution.controls import Controls
from tests.shadowspill.execution.test_initialization import storage


def controls(*, shape=(2,), stride=(2,), offset=1, region=None):
    owner = torch.arange(5, dtype=torch.float32)
    binding = dict(name="scale", object_id="buffer", parameter=False, persistent=True)
    state = storage(
        {"registrations": [binding, {**binding, "name": "tied"}]},
        {
            "buffer": dict(
                dtype="torch.float32", shape=shape, stride=stride, storage_offset=offset
            )
        },
        {"buffer": owner},
    )
    if region is not None:
        state.regions["buffer"] = region
    return Controls(state), owner


def test_buffer_fills_preserve_views_and_reject_invalid_calls_before_writing():
    control, owner = controls()
    control.apply({"scale": 9})
    torch.testing.assert_close(owner, torch.tensor([0.0, 9.0, 2.0, 9.0, 4.0]))
    control.apply({})
    before = owner.clone()
    for values, error in (
        ({"scale": 4, "unknown": 1}, ValueError),
        ({"scale": 4, "tied": 5}, ValueError),
        ({"scale": 4, "tied": [5]}, TypeError),
        ({"scale": 4, "tied": 1e100}, RuntimeError),
    ):
        with pytest.raises(error):
            control.apply(values)
        torch.testing.assert_close(owner, before)
    control.apply({"tied": 3})
    torch.testing.assert_close(owner, torch.tensor([0.0, 3.0, 2.0, 3.0, 4.0]))


@pytest.mark.parametrize("length", [0, 2, 5])
def test_sharded_buffer_fills_touch_owned_elements_but_not_padding(length):
    control, owner = controls(
        shape=(10,),
        stride=(1,),
        offset=0,
        region=SimpleNamespace(length=length, storage_elements=10),
    )
    before = owner.clone()
    control.apply({"scale": 7})
    assert torch.all(owner[:length] == 7)
    torch.testing.assert_close(owner[length:], before[length:])


def test_sharded_noncontiguous_buffers_are_refused():
    control, owner = controls(region=SimpleNamespace(length=2, storage_elements=5))
    before = owner.clone()
    with pytest.raises(ValueError, match="dense full-storage"):
        control.apply({"scale": 2})
    torch.testing.assert_close(owner, before)
