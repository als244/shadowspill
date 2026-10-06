"""Logical gradients and complete quantization survive sharded optimizer updates."""

from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from shadowspill.pytorch.distributed import Distributed
from shadowspill.pytorch.distributed._checkpoint import restore_compute_weights
from shadowspill.pytorch.distributed._optimizer import distribute_capture
from shadowspill.pytorch.distributed._shards import (
    fill_parameter,
    optimizer_parameters,
    parameter_layouts,
)
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.representations import component_at
from tests.shadowspill.pytorch.state.representations import model


def worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "store")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        net = model()
        with torch.no_grad():
            net.weight.scale.mul_(rank + 1)
        bound = Distributed(dist.group.WORLD)._bind(
            net, dist.group.WORLD, namespace="representations"
        )
        try:
            bound.synchronize_initial(net)
            torch.testing.assert_close(net.weight.scale, torch.tensor(0.025))
            with bound.activate():
                weights = dict(net.named_parameters())
                layouts = parameter_layouts(
                    bound, weights, set(weights), master_dtype=torch.float32
                )
                owned = optimizer_parameters(
                    weights, layouts, master_dtype=torch.float32
                )
                with torch.no_grad():
                    fill_parameter(
                        owned["weight"], weights["weight"], layouts["weight"]
                    )
                captured = distribute_capture(
                    capture_optimizer(
                        owned, torch.optim.SGD(owned.values(), lr=0.02, foreach=False)
                    ),
                    layouts,
                    gradient_dtype=torch.float32,
                    representative_values={"compute.weight": net.weight},
                )
                values = {}
                for binding in captured.bindings:
                    if binding.logical_name == "weight":
                        values[binding.name] = owned["weight"]
                    elif binding.logical_name == "gradient.weight":
                        values[binding.name] = torch.empty(net.weight.shape)
                    elif binding.logical_name == "compute.weight":
                        values[binding.name] = component_at(
                            net.weight, binding.component_path
                        )
                    else:
                        raise AssertionError(binding.name)
                expected = net.weight.dense().detach().clone()
                for step in range(3):
                    generator = torch.Generator().manual_seed(101 + step)
                    gradients = torch.randn(2, *expected.shape, generator=generator)
                    values["gradient.weight"].copy_(gradients[rank])
                    with torch.no_grad():
                        for task in captured.update_tasks:
                            task.artifact.graph_module(
                                *(values[n] for n in task.binding_names)
                            )
                        expected.sub_(gradients.sum(0), alpha=0.02)
                        quantized = net.weight.detach().clone()
                        quantized.copy_(expected)
                    torch.testing.assert_close(net.weight.payload, quantized.payload)
                    torch.testing.assert_close(net.weight.scale, quantized.scale)
                    torch.testing.assert_close(
                        owned["weight"], expected.flatten().chunk(2)[rank]
                    )
                restored = model()
                restore_compute_weights(
                    dict(restored.named_parameters()), owned, bound, chunk_bytes=12
                )
                torch.testing.assert_close(restored.weight.payload, net.weight.payload)
                torch.testing.assert_close(restored.weight.scale, net.weight.scale)
        finally:
            bound.close()
    finally:
        dist.destroy_process_group()


def test_quantized_state_with_sharded_masters_and_checkpoint_restore():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(root,), nprocs=2, join=True)
