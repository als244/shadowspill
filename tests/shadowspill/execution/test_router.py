"""A packed MoE router trains from a saved Plan in a fresh worker."""

import pytest
import torch

from shadowspill import Parallelism, build_step, measure_program
from tests.shadowspill.execution.helpers import run_worker, save_plan


@pytest.mark.cuda
@pytest.mark.parametrize("ranks", [1, 2])
def test_packed_router_training_from_saved_plan(tmp_path, ranks):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    router = pytest.importorskip("mlops.kernels.router")
    weights = pytest.importorskip("mlops.expert_parallel.parameters")

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                weights.BF16ComputeWeight(torch.empty(3, 16, dtype=torch.bfloat16))
            )

        def forward(self, x):
            return router.routed_logits(x, self.weight)

    sample = torch.arange(112, dtype=torch.bfloat16).reshape(7, 16) / 128
    with torch.device("meta"):
        template = Network()
    program = build_step(
        template,
        optimizer=torch.optim.SGD(template.parameters(), lr=0.01, foreach=False),
        objective=lambda m, x: m(x).sum(),
        example_args=(sample,),
        partition="whole",
        master_dtype=torch.float32,
        grad_dtype=torch.float32,
        parallelism=Parallelism.dp(ranks_per_replica=ranks),
    )
    if ranks > 1:
        # One representative GPU prepares a distributed router without NCCL.
        copies = [
            item
            for item in program.bindings["optimizer_bindings"]
            if item["role"] == "compute_copy"
        ]
        assert len(copies) == 1
        assert copies[0]["name"] == "compute.weight._data"
        assert copies[0]["dtype"] == "torch.bfloat16"
        assert tuple(copies[0]["shape"]) == (24,)
        gathers = [
            item
            for item in program.bindings["collective_tasks"].values()
            if item["operation"] == "all_gather"
        ]
        assert gathers and all(item["dtype"] == "torch.bfloat16" for item in gathers)
        measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
        return  # Distinct-device execution is covered by the EP integration case.
    save_plan(program, tmp_path)
    payload = torch.arange(48, dtype=torch.bfloat16).reshape(3, 16) / 64
    initial = {"weight._data": payload.clone(), "master.weight": payload.float()}
    master = torch.nn.Parameter(payload.float())
    optimizer = torch.optim.SGD([master], lr=0.01, foreach=False)
    expected = []
    for _ in range(3):
        loss = (sample.float() @ payload.float().t()).sum()
        master.grad = sample.float().sum(0).expand_as(master).contiguous()
        optimizer.step()
        payload.copy_(master.detach())
        expected.append(
            (
                [(loss, None)],
                {
                    "weight._data": payload.clone(),
                    "master.weight": master.detach().clone(),
                },
            )
        )
    torch.save(
        dict(
            initial=initial,
            examples=[((sample,), {})],
            expected=expected,
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
