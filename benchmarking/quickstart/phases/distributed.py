"""Rendezvous for calibration/execution only; compute profiling stays local."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

from shadowspill.communication.binding.ranks import RankAssignment


@dataclass(frozen=True)
class Workers:
    rank: int
    rank_map: dict[int, int]

    @property
    def count(self):
        return len(self.rank_map)

    @property
    def logical_rank(self):
        return self.rank_map[self.rank]

    def gather(self, value):
        if self.count == 1:
            return [value]
        import torch.distributed as dist

        values = [None] * self.count
        dist.all_gather_object(values, value)
        return values


@contextmanager
def workers(request, index):
    count = index.get("world_size", 1)
    mapping = request.get("rank_map")
    mapping = (
        {i: i for i in range(count)}
        if mapping is None
        else {int(worker): rank for worker, rank in mapping.items()}
    )
    RankAssignment.resolve(count, mapping)
    if int(os.environ.get("WORLD_SIZE", "1")) != count:
        raise ValueError("worker count differs from the saved Program")
    if count == 1:
        yield Workers(0, mapping)
        return
    import torch.distributed as dist

    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        yield Workers(dist.get_rank(), mapping)
    finally:
        dist.destroy_process_group()
