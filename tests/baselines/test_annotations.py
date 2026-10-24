"""Checkpoint replay labels must preserve outputs and gradients."""

from contextlib import contextmanager
from types import SimpleNamespace
import sys

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from baselines.common.annotations import checkpoint_contexts, phase


@pytest.mark.parametrize("reentrant", [False, True])
def test_only_checkpoint_replay_is_labeled_recompute(monkeypatch, reentrant):
    active, seen = [], []

    @contextmanager
    def label(name):
        active.append(name)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(torch.cuda.nvtx, "range", label)

    class Block(torch.nn.Module):
        def forward(self, value):
            seen.append(tuple(active))
            return value.sin().square()

    value = torch.linspace(-1, 1, 11, requires_grad=True)
    expected = value.detach().clone().requires_grad_()
    expected.sin().square().sum().backward()
    if reentrant:
        from baselines.engines.deepspeed.train import Checkpointed

        def ds_checkpoint(fn, *args):
            return checkpoint(fn, *args, use_reentrant=True)

        monkeypatch.setitem(
            sys.modules, "deepspeed",
            SimpleNamespace(checkpointing=SimpleNamespace(checkpoint=ds_checkpoint)),
        )
        result = Checkpointed(Block(), annotations=True)(value)
    else:
        result = checkpoint(
            Block(), value, use_reentrant=False, context_fn=checkpoint_contexts
        )
    assert seen == [("baseline.phase/checkpoint_forward",)]
    result.sum().backward()
    assert seen == [
        ("baseline.phase/checkpoint_forward",), ("baseline.phase/recompute",)
    ]
    assert active == []  # also handles non-reentrant early-stop exceptions
    torch.testing.assert_close(value.grad, expected.grad)
    torch.testing.assert_close(result, expected.sin().square())


def test_disabled_annotation_does_not_touch_cuda(monkeypatch):
    def fail(*args):
        raise AssertionError("disabled profiling entered CUDA")

    monkeypatch.setattr(torch.cuda.nvtx, "range", fail)
    with phase("recompute", False):
        pass
