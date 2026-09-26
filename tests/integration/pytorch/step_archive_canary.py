"""Fresh-process canary: a build with an export bypass key answers from the
step archive the second time, and one capture serves every ordering."""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from canary_phases import phase

from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import (
    ObjectiveResult,
    Runtime,
    build_step_programs,
    import_model_state,
)
from shadowspill.step import StepDataOrdering
from tests.spill_pool import spill_pool


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(512, 512, bias=False)
        self.second = nn.Linear(512, 512, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(value)))


def _objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor
) -> ObjectiveResult:
    return ObjectiveResult(((model(value) - target) ** 2).mean(), {})


def _build_optimizer(parameters: Iterable[torch.nn.Parameter]) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=0.003, foreach=False)


def _phases(program: object) -> tuple[str, ...]:
    return tuple(name for name, _duration in program.phase_timings_ns)  # type: ignore[attr-defined]


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
                    physical_capacity=2 << 30, provider_headroom=512 << 20
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
        request = dict(
            objective=_objective,
            optimizer=_build_optimizer,
            example_inputs=example_inputs,
            runtime=runtime,
            execution="execution",
            spill="spill",
            artifact_store=cache,
            verbose=False,
        )
        orderings = (StepDataOrdering(2, 1), StepDataOrdering(1, 2))

        phase("build")
        first = build_step_programs(
            model, orderings=orderings, export_bypass_key="canary-1", **request
        )
        if len(first) != 2:
            raise AssertionError("one program per ordering was expected")
        if tuple(item.data_ordering for item in first) != orderings:
            raise AssertionError("programs must come back in the orderings' order")
        if first[0].problem.program.digest == first[1].problem.program.digest:
            raise AssertionError("two orderings lowered to the same program")
        if "objective_export" not in _phases(first[0]):
            raise AssertionError("the first program must carry the shared capture")
        if "objective_export" in _phases(first[1]) or "program_lowering" not in _phases(
            first[1]
        ):
            raise AssertionError("a later ordering carries only its own lowering")
        archived = sorted(Path(cache).glob("v*/build/steps/*/*/step_program.json"))
        if len(archived) != 2:
            raise AssertionError(
                f"two step programs should be archived, found {len(archived)}"
            )

        phase("bypass")
        second = build_step_programs(
            model, orderings=orderings, export_bypass_key="canary-1", **request
        )
        for before, after in zip(first, second, strict=True):
            if after.digest != before.digest:
                raise AssertionError(
                    "a bypassed build must return the archived program"
                )
            if _phases(after) != ("step_lookup", "total"):
                raise AssertionError(f"a hit must only look up, not {_phases(after)}")

        phase("other-key")
        (other,) = build_step_programs(
            model, orderings=orderings[:1], export_bypass_key="canary-2", **request
        )
        if "objective_export" not in _phases(other):
            raise AssertionError("another key must capture again")
        # Profiles are filed under the key too, so the program is measured
        # again and its costs may differ; its structure may not.
        structure = lambda program: (  # noqa: E731
            program.data_ordering,
            tuple(task.task_id for task in program.problem.program.tasks),
        )
        if structure(other) != structure(first[0]):
            raise AssertionError("the same request under another key is the same step")

        phase("no-key")
        (unkeyed,) = build_step_programs(model, orderings=orderings[:1], **request)
        if "objective_export" not in _phases(unkeyed):
            raise AssertionError("without a key every build captures")
        if len(sorted(Path(cache).glob("v*/build/steps/*/*/step_program.json"))) != 3:
            raise AssertionError("a build without a key files nothing")
        phase("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
