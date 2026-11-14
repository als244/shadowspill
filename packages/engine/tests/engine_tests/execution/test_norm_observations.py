"""Bounded diagnostics preserve compute weights, owned gradients and padding."""

from types import SimpleNamespace

import pytest
import torch

from shadowspill.execution.observations.bindings import norm_inputs
from shadowspill.execution.observations.collector import ParameterNorms
from shadowspill.ir import ResourceKind, ResourceSpec


def path(ownership=()):
    resource = ResourceSpec("gpu", ResourceKind.COMPUTE)

    def task(identifier, phase, values):
        return SimpleNamespace(
            task=SimpleNamespace(task_id=identifier, phase=phase, resource=resource),
            entry=SimpleNamespace(
                input_slots=[
                    SimpleNamespace(object_id=v, leaf_index=i)
                    for i, (v, _) in enumerate(values)
                ]
            ),
            description={
                "tensor_positions": list(range(len(values))),
                "tensor_geometry": [
                    {"shape": (count,), "stride": (1,), "dtype": "torch.float32"}
                    for _, count in values
                ],
            },
        )

    n = 3 if ownership else 5
    return SimpleNamespace(
        plan=SimpleNamespace(
            program=SimpleNamespace(
                bindings={
                    "registrations": [{"name": "weight", "object_id": "compute"}],
                    "optimizer_bindings": [
                        {
                            "role": "gradient",
                            "parameter_name": "weight",
                            "object_id": "gradient",
                        }
                    ],
                    "parameter_ownership": ownership,
                }
            )
        ),
        tasks=[
            task("first", "forward", [("compute", 5)]),
            task("recompute", "backward", [("compute", 5)]),
            task("update", "optimizer", [("master", n), ("gradient", n)]),
        ],
    )


def test_observe_weight_once_and_gradient_only_at_its_update():
    names, reads = norm_inputs(path(), 0)
    assert names == ("weight",)
    assert set(reads) == {"first", "update"}
    assert reads["first"][0].argument == 0
    assert reads["update"][0].argument == 1
    assert [r.column for group in reads.values() for r in group] == [0, 1]


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("partitioned", [False, True])
def test_owned_gradient_ranges_exclude_padding_and_duplicate_replicas(
    rank, partitioned
):
    copies = ((0, 2), (1, 3)) if partitioned else ((0, 1, 2, 3),)
    groups = copies if partitioned else ((0, 1), (2, 3))
    ownership = [
        {
            "names": ("weight",),
            "local_shape": (5,),
            "copy_groups": copies,
            "optimizer": {"elements": 5, "groups": groups},
        }
    ]
    _, reads = norm_inputs(path(ownership), rank)
    weight, gradient = reads["first"][0], reads["update"][0]
    assert weight.contribute == (rank < 2 if partitioned else rank == 0)
    assert gradient.elements == (
        3 if (rank < 2 if partitioned else rank % 2 == 0) else 2
    )
    assert gradient.contribute == (True if partitioned else rank < 2)


def test_finish_sums_squares_and_returns_cpu_norms():
    collector = object.__new__(ParameterNorms)
    collector.names = ("weight",)
    collector.sums = torch.tensor([9.0, 16.0])
    collector.worker = SimpleNamespace(
        groups=SimpleNamespace(gather=lambda local: [local, [16.0, 9.0]])
    )
    result = {"weight": {kind: torch.empty(()) for kind in ("param_norm", "grad_norm")}}
    collector.finish(result)
    assert all(value.item() == 5 for value in result["weight"].values())


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_norm_kernel_handles_strides_scalars_and_bounded_multi_pass(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from shadowspill.execution.observations.kernels import (
        BLOCK,
        PARTIALS,
        grid,
        norm_partials,
        norm_sum,
    )

    scratch = torch.empty(PARTIALS, dtype=torch.float32, device="cuda")
    output = torch.empty(1, dtype=torch.float32, device="cuda")
    torch.manual_seed(1)
    for value in (
        torch.tensor(2.0, device="cuda", dtype=dtype),
        torch.randn(81, 67, device="cuda", dtype=dtype).T[::2],
        torch.randn(PARTIALS * BLOCK + 37, device="cuda", dtype=dtype),
    ):
        count = grid(value.numel())
        for _ in range(2):
            before = torch.cuda.memory_allocated()
            norm_partials[(count,)](
                value,
                scratch,
                value.numel(),
                tuple(value.shape),
                value.stride(),
                BLOCK,
                enable_fp_fusion=False,
            )
            norm_sum[(1,)](scratch, output, 0, count, PARTIALS)
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
        torch.testing.assert_close(
            output.sqrt()[0],
            torch.linalg.vector_norm(value, dtype=torch.float32),
            atol=1e-5,
            rtol=2e-6,
        )
