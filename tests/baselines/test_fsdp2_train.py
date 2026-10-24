"""FSDP2 initialization and the trainer's pure helpers."""

import subprocess
import sys
import textwrap

import pytest
import torch

from baselines.engines.fsdp2.train import float8_filter


def test_float8_converts_block_linears_the_kernels_take():
    assert float8_filter(torch.nn.Linear(128, 256), "blocks.0.attn.wq")
    assert not float8_filter(torch.nn.Linear(128, 100), "blocks.0.mlp.w1")
    assert not float8_filter(torch.nn.Linear(128, 512), "lm_head")
    assert not float8_filter(torch.nn.Linear(128, 512), "embed")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA initialization")
def test_cpu_offloaded_initialization_matches_device_shards(tmp_path):
    # A fresh process isolates the process group and DTensor's RNG tracker.
    script = textwrap.dedent(
        """
        import sys
        import torch
        import torch.distributed as dist
        from torch import nn
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard
        from baselines.engines.fsdp2.model import initialize

        torch.cuda.set_device(0)
        dist.init_process_group('nccl', init_method='file://' + sys.argv[1],
                                rank=0, world_size=1,
                                device_id=torch.device('cuda:0'))
        mesh = init_device_mesh('cuda', (1,))
        try:
            initialized = []
            for device, offload in (('cuda', False), ('cpu', True), ('cuda', True)):
                with torch.device('meta'):
                    model = nn.Sequential(
                        nn.Embedding(128, 128, dtype=torch.bfloat16),
                        nn.Linear(128, 128, bias=False, dtype=torch.bfloat16),
                        nn.Linear(128, 128, bias=False, dtype=torch.bfloat16),
                    )
                policy = CPUOffloadPolicy() if offload else OffloadPolicy()
                for child in model:
                    fully_shard(child, mesh=mesh, offload_policy=policy)
                fully_shard(model, mesh=mesh, offload_policy=policy)
                model.to_empty(device=device)
                originals = list(model.parameters())
                pointers = [p.to_local().data_ptr() for p in originals]
                torch.manual_seed(17)
                with torch.device(device):
                    initialize(model)
                assert all(a is b for a, b in zip(originals, model.parameters()))
                assert pointers == [p.to_local().data_ptr() for p in originals]
                initialized.append([p.to_local().cpu().clone() for p in originals])
            assert all(torch.equal(a, b) and torch.equal(a, c)
                       for a, b, c in zip(*initialized))
            assert not torch.equal(initialized[1][1], initialized[1][2])
        finally:
            dist.destroy_process_group()
        """
    )
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "rendezvous")],
        check=True,
        timeout=60,
    )
