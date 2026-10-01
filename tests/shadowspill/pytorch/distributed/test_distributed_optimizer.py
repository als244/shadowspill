from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from shadowspill.pytorch.distributed import Distributed
from shadowspill.pytorch.distributed._optimizer import distribute_capture
from shadowspill.pytorch.distributed._reductions import (
    before_observations,
    initialize_missing_gradients,
)
from shadowspill.pytorch.distributed._shards import (
    fill_parameter,
    optimizer_parameters,
    parameter_layouts,
    validate_optimizer,
)
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.optimizer.capture import declare_optimizer_state
from shadowspill.pytorch.optimizer.metrics import with_parameter_metrics
from shadowspill.pytorch.optimizer.starts import ConstantStart, ValueStart


class Model(nn.Module):
    def __init__(self, rank, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-1, 1, 15).reshape(3, 5).to(dtype))
        self.bias = nn.Parameter(torch.linspace(-0.5, 0.5, 5).to(dtype))
        self.local = nn.Parameter(torch.full((3,), float(rank + 1), dtype=dtype))


def check_training_lowering(model, captured, rank, bound):
    # Exercise actual training lowering with a parameter
    # absent from rank 1's objective, including new gradient
    # publication and subsequent metric/update dependencies.
    from torch._subclasses.fake_tensor import FakeTensorMode

    from shadowspill.pytorch.capture.aot import capture_training
    from shadowspill.pytorch.capture.fake import (
        fake_device_inputs,
        fake_device_model,
    )
    from shadowspill.pytorch.graph_pairs import (
        partition_training_capture,
    )
    from shadowspill.pytorch.lowering.training import (
        lower_partitioned_training_program,
    )
    from tests.shadowspill.pytorch.lowering.test_training_lowering import (
        _measurement,
    )

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake = fake_device_model(model, mode)

    def objective(m, x):
        value = x @ m.weight + m.local.sum()
        if rank == 0:
            value = value + m.bias
        return value.square().sum()

    with mode:
        paired = partition_training_capture(
            capture_training(
                fake,
                objective,
                fake_device_inputs(
                    (torch.ones(2, 3, dtype=model.weight.dtype),),
                    mode,
                ),
            ),
            accumulating=False,
            gradient_dtype=torch.float32,
        )
    artifacts = [task.artifact for task in captured.update_tasks]
    for stage in paired.stages:
        for option in stage.graph_pairs.variants:
            artifacts.extend((option.pair.forward, option.pair.backward))
    measures = {item.compatibility_digest: _measurement(item) for item in artifacts}
    lowered = lower_partitioned_training_program(
        fake,
        (paired,),
        measures,
        captured,
        optimizer_ordering="tail",
    )
    gradient = next(item for item in lowered.gradients if item.parameter_name == "bias")
    producers = [
        task
        for task in lowered.program.tasks
        if gradient.gradient_object_id in task.outputs
    ]
    assert producers
    if rank == 1:
        assert len(producers) == 1 and producers[0].phase == "optimizer"
    bound.control.agree(
        "lowered/task_roles",
        [task.phase for task in lowered.program.tasks],
    )


def worker(rank, root, capture_on_gpu=False):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "store")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for master in (False, True):
            for sharded in (False, True):
                model = Model(rank, torch.bfloat16 if master else torch.float32)
                spec = Distributed(
                    dist.group.WORLD,
                    replica_overrides=[([model.local], None)],
                    timeout=30,
                )
                bound = spec._bind(
                    model, dist.group.WORLD, namespace=f"optimizer/{master}/{sharded}"
                )
                bound.shard_optimizer = sharded
                try:
                    with bound.activate():
                        weights = dict(model.named_parameters())
                        layouts = parameter_layouts(
                            bound,
                            weights,
                            set(weights) - ({"bias"} if rank else set()),
                            master_dtype=torch.float32 if master else None,
                        )
                        owned = optimizer_parameters(
                            weights,
                            layouts,
                            master_dtype=torch.float32 if master else None,
                        )
                        if sharded and not master:
                            assert owned["weight"].is_meta
                        optimizer = torch.optim.AdamW(
                            owned.values(), lr=0.02, foreach=False
                        )
                        validate_optimizer(optimizer, layouts)
                        for name, layout in layouts.items():
                            if layout.master:
                                with torch.no_grad():
                                    fill_parameter(owned[name], weights[name], layout)
                        for item in declare_optimizer_state(owned, optimizer):
                            if isinstance(item.start, ConstantStart):
                                value = torch.full(
                                    item.shape, item.start.value, dtype=item.dtype
                                )
                            elif isinstance(item.start, ValueStart):
                                value = item.start.value.detach().clone()
                            else:
                                raise AssertionError(item)
                            optimizer.state[owned[item.parameter_name]][
                                item.entry_name
                            ] = value
                        captured = capture_optimizer(owned, optimizer)
                        assert not captured.update_is_opaque, captured.opaque_reason
                        staged = distribute_capture(
                            captured,
                            layouts,
                            gradient_dtype=torch.float32,
                            parameter_stage_owners={
                                "weight": (0,),
                                "bias": (1,),
                                "local": (2,),
                            },
                        )
                        assert sorted(
                            task.completion_stage_index for task in staged.update_tasks
                        ) == [0, 1, 2]
                        gathers = [
                            node
                            for task in staged.update_tasks
                            for node in task.artifact.graph_module.graph.nodes
                            if node.target
                            == torch.ops._c10d_functional.all_gather_into_tensor.default
                        ]
                        assert len(gathers) == (2 if sharded else 0)
                        for node in gathers:
                            # Optional FP32 masters are cast by their owner before
                            # communication. Only compute precision goes on wire.
                            assert node.args[0].meta["val"].dtype == model.weight.dtype
                            assert node.meta["val"].dtype == model.weight.dtype
                        if master and sharded:
                            for name in ("weight", "bias"):
                                assert owned[name].dtype == torch.float32
                                assert owned[name].numel() == layouts[name].capacity
                        captured = distribute_capture(
                            captured,
                            layouts,
                            gradient_dtype=torch.float32,
                            already_reduced=True,
                            parameter_stage_owners={
                                "weight": (0,),
                                "bias": (0,),
                                "local": (0,),
                            },
                        )
                        captured = with_parameter_metrics(
                            captured, lambda w, g: {"grad2": g.float().square().sum()}
                        )
                        captured = before_observations(captured, layouts)
                        captured = initialize_missing_gradients(captured, layouts)
                        assert all(
                            root.kind.value == "input"
                            for root in captured.update.storage_contract.roots
                        )
                        if capture_on_gpu:
                            check_training_lowering(model, captured, rank, bound)
                        expected_weights = {
                            name: nn.Parameter(value.float().detach().clone())
                            for name, value in weights.items()
                        }
                        expected = torch.optim.AdamW(
                            expected_weights.values(), lr=0.02, foreach=False
                        )
                        arguments = {}
                        for binding in captured.bindings:
                            name = binding.name
                            if name.startswith("gradient."):
                                arguments[name] = torch.empty(
                                    layouts[name.removeprefix("gradient.")].shape,
                                    dtype=binding.tensor.dtype,
                                )
                            elif name.startswith("compute."):
                                arguments[name] = weights[name.removeprefix("compute.")]
                            elif name in weights:
                                arguments[name] = (
                                    owned[name]
                                    if layouts[name].master
                                    else weights[name]
                                )
                            elif name.startswith("optimizer."):
                                parameter, field = name.removeprefix(
                                    "optimizer."
                                ).rsplit(".", 1)
                                arguments[name] = optimizer.state[owned[parameter]][
                                    field
                                ]
                            else:
                                raise AssertionError(name)
                        for step in range(3):
                            torch.manual_seed(777 + step)
                            for name, weight in weights.items():
                                gradients = torch.randn(2, *weight.shape)
                                if name == "bias":
                                    gradients[1].zero_()
                                arguments["gradient." + name].copy_(gradients[rank])
                                expected_weights[name].grad = (
                                    gradients[rank]
                                    if name == "local"
                                    else gradients.sum(0)
                                )
                            with torch.no_grad():
                                for task in captured.update_tasks:
                                    function = task.artifact.graph_module
                                    if task.output_names and not task.binding_names:
                                        # Exercise the captured zero producer on CPU;
                                        # Its deployment graph targets the GPU.
                                        import copy

                                        function = copy.deepcopy(function)
                                        for node in function.graph.nodes:
                                            if "device" in node.kwargs:
                                                node.kwargs = {
                                                    **node.kwargs,
                                                    "device": torch.device("cpu"),
                                                }
                                        function.recompile()
                                    result = function(
                                        *(
                                            arguments[name]
                                            for name in task.binding_names
                                        )
                                    )
                                    for name, value in zip(
                                        task.output_names, result, strict=False
                                    ):
                                        arguments[name] = value
                                    if task.metric_schema is not None:
                                        names = [
                                            name.removeprefix("gradient.")
                                            for name in task.binding_names
                                            if name.startswith("gradient.")
                                        ]
                                        for name, value in zip(
                                            names, result, strict=True
                                        ):
                                            torch.testing.assert_close(
                                                value,
                                                expected_weights[name]
                                                .grad.square()
                                                .sum(),
                                            )
                            expected.step()
                            for name, weight in weights.items():
                                torch.testing.assert_close(
                                    weight,
                                    expected_weights[name].to(weight.dtype),
                                    rtol=1e-6,
                                    atol=1e-6,
                                )
                                layout = layouts[name]
                                for field in ("exp_avg", "exp_avg_sq"):
                                    actual = optimizer.state[owned[name]][
                                        field
                                    ].reshape(-1)[: layout.length]
                                    reference = expected.state[expected_weights[name]][
                                        field
                                    ].reshape(-1)[
                                        layout.start : layout.start + layout.length
                                    ]
                                    torch.testing.assert_close(
                                        actual, reference, rtol=1e-6, atol=1e-7
                                    )
                finally:
                    bound.close()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_captured_owned_state_matches_adamw_with_and_without_masters():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(root,), nprocs=2, join=True)


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_distributed_optimizer_cuda_graphs_lower_for_save_and_recompute():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(root, True), nprocs=2, join=True)
