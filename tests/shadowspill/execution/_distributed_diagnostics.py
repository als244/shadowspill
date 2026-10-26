"""Diagnostics leave every owned shard, local buffer and RNG unchanged."""

import torch

from shadowspill import Trainer


def check(executor, calls, directory):
    def snapshot():
        chunks = {}

        def collect(request, value):
            chunks[request.name, request.offset] = value.clone()

        executor.export_tensors(collect, chunk_bytes=32)
        return chunks

    initial = snapshot()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    with Trainer(executor.plan, executor=executor) as trainer:
        summary = trainer.diagnose(
            args=[args for args, _ in calls],
            kwargs=[kwargs for _, kwargs in calls],
            directory=directory,
            warmup=1,
        )
        assert trainer.step_count == 0
        assert summary["buffer_snapshot_bytes"] == 8
    torch.testing.assert_close(initial, snapshot(), rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    report = directory / f"rank_{executor.logical_rank:04d}"
    assert (report / "timelines/traced.html").is_file()
    print(
        f"logical rank {executor.logical_rank}: diagnostics preserved state", flush=True
    )
