"""Fresh-process canary: a planning-only build gives back what its own scopes
allocated, even what something outside the plan still references.

A kernel library may keep small tensors it made while a task ran -- a cache of
its last few results, keyed by argument identity -- and nothing evicts them
until the library is called again with other arguments. A callable takes such
ranges back when it closes. A build that profiles and lowers without leaving a
callable must take them back too, or they stay in the device pool after its
plan has closed, where a later plan that needs one contiguous slice of the
pool cannot have it.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from canary_phases import phase

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import (
    ObjectiveResult,
    Runtime,
    build_step_programs,
    import_model_state,
)
from shadowspill.runtime.occupancy import live_allocations
from tests.spill_pool import spill_pool

#: What a kernel's metadata cache holds: tensors made while a task ran,
#: referenced from module state after the task is done. One per shape, so a
#: task's repeated warmup calls find it and allocate nothing more, as a cache
#: keyed by its arguments settles.
_KEPT: dict[tuple[torch.Size, torch.device], torch.Tensor] = {}


@torch.library.custom_op("planning_residue_canary::scale", mutates_args=())
def _scale(value: torch.Tensor) -> torch.Tensor:
    key = (value.shape, value.device)
    if key not in _KEPT:
        _KEPT[key] = torch.arange(3, device=value.device, dtype=torch.int64)
    return value * 2


@_scale.register_fake
def _scale_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


def _scale_backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
    del ctx
    return grad * 2


_scale.register_autograd(_scale_backward)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(512, 512, bias=False)
        self.second = nn.Linear(512, 512, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(_scale(torch.relu(self.first(value))))


def _objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor
) -> ObjectiveResult:
    return ObjectiveResult(((model(value) - target) ** 2).mean(), {})


def _build_optimizer(parameters: Iterable[torch.nn.Parameter]) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=0.003, foreach=False)


def _still_attached(tensor: torch.Tensor) -> bool:
    """Whether a tensor still points at memory. A detached storage refuses to
    give its pointer at all, so a later read cannot reach the reclaimed bytes."""

    try:
        return tensor.untyped_storage().data_ptr() != 0
    except RuntimeError:
        return False


def main(arguments: Iterable[str] | None = None) -> int:
    values = tuple(sys.argv[1:] if arguments is None else arguments)
    adapter = Path(values[0]).resolve()
    with tempfile.TemporaryDirectory() as cache:
        torch.manual_seed(7)
        example_inputs = [
            [torch.randn(4, 512), torch.randn(4, 512)],
            [torch.randn(4, 512), torch.randn(4, 512)],
        ]
        phase("runtime")
        runtime = Runtime(
            pools={
                "execution": device(
                    physical_capacity=2 << 30, external_headroom=512 << 20
                ),
                "spill": spill_pool(1 << 30),
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
            library_path=adapter,
        )
        model = import_model_state(
            _Model(), runtime=runtime, pool="spill", release_source=True
        )

        phase("build")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_step_programs(
                model,
                objective=_objective,
                optimizer=_build_optimizer,
                example_inputs=example_inputs,
                runtime=runtime,
                execution="execution",
                spill="spill",
                artifact_store=cache,
                profiling_options=CORRECTNESS_PROFILING,
                verbose=False,
            )

        phase("residue")
        if not _KEPT:
            raise AssertionError("the kernel kept nothing, so nothing was tested")
        left = [
            item
            for item in live_allocations(runtime, "execution")
            if item.unclaimed_scope_workspace
        ]
        if left:
            raise AssertionError(
                f"the closed build left {len(left)} scope allocation(s) in the"
                " device pool: "
                + ", ".join(
                    f"plan {item.origin_plan_id} offset {item.offset}"
                    for item in left[:4]
                )
            )
        attached = [item for item in _KEPT.values() if _still_attached(item)]
        if attached:
            raise AssertionError(
                f"{len(attached)} of the {len(_KEPT)} kept tensors still point at"
                " pool memory the build gave back"
            )
        if not any("reclaimed" in str(item.message) for item in caught):
            raise AssertionError("reclaiming the build's residue must say so")
        phase("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
