"""Bad local input or invocation choices fail on every worker before dispatch."""

from datetime import timedelta
from types import SimpleNamespace


def worker(rank, rendezvous):
    import pytest
    import torch.distributed as dist

    from shadowspill.communication.binding.groups import GroupBindings
    from shadowspill.execution.distributed.checks import check_invocation, coordinated

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=20),
    )
    groups = None
    try:
        groups = GroupBindings(
            identity="preflight",
            groups=[(1, 0)],
            rank_map={0: 1, 1: 0},
            topology={},
            backend="gloo",
        )
        context = SimpleNamespace(groups=groups)
        with pytest.raises(ValueError, match="different Plans"):
            check_invocation(context, "evaluation" if rank else None, False)
        with pytest.raises(ValueError, match="accumulation modes"):
            check_invocation(context, None, bool(rank))

        def prepare_inputs():
            if rank == 1:
                raise ValueError("incorrect local input shape")
            return "ready"

        with pytest.raises(RuntimeError, match="worker 1: incorrect local input shape"):
            coordinated(context, prepare_inputs)
        # Recoverable validation errors leave the control group usable.
        check_invocation(context, None, False)
        with pytest.raises(ValueError, match="observations"):
            check_invocation(context, None, False, bool(rank))
        assert coordinated(context, lambda: rank) == rank
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()


def test_preflight_failures_are_coordinated(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
