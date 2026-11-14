"""Host rendezvous and rank mapping, independent of any client's file formats."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta


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
def workers(world_size, *, rank_map=None):
    """Own a Gloo control group; reject invalid bindings before rendezvous.

    Clients pass the saved artifact's participant count and optional launcher
    to logical rank mapping. CUDA/NCCL resources belong to the phase APIs.
    """
    from shadowspill.communication.binding.ranks import RankAssignment

    mapping = (
        {i: i for i in range(world_size)}
        if rank_map is None
        else {int(worker): rank for worker, rank in rank_map.items()}
    )
    RankAssignment.resolve(world_size, mapping)
    if int(os.environ.get("WORLD_SIZE", "1")) != world_size:
        raise ValueError("worker count differs from the saved artifact")
    if world_size == 1:
        yield Workers(0, mapping)
        return
    import torch.distributed as dist

    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        yield Workers(dist.get_rank(), mapping)
    finally:
        dist.destroy_process_group()
