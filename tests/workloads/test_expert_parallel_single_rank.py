"""An expert-parallel group of one rank builds local experts."""

from contextlib import nullcontext

import torch

from workloads.models.shared import expert_parallel as ep


def test_a_group_of_one_yields_local_experts(monkeypatch):
    """The factory returns None for a size-one group, as for no group: the
    local experts have no transport, the EP layer's would move every row for
    no one. Nothing about the group or a buffer is touched."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    config = object()  # nothing of the configuration is read on the local path
    with ep.expert_construction(
        config, renormalize_topk=False, ep_group=object(), token_capacity=256
    ) as factory:
        assert factory() is None
        assert factory.options is None and factory.buffer is None
