"""Fresh-process public planning-failure and rollback canary."""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    AdmissionError,
    CaptureError,
    CompilationError,
    PlanInfeasibleError,
    ProfilingError,
    Runtime,
    export_model_state,
    import_model_state,
    plan_forward,
)


class _DataDependentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scaled = value * self.scale
        if scaled.sum().item() > 0:
            return scaled
        return -scaled


_BROKEN_OP = torch.library.Library(  # type: ignore[no-untyped-call]
    "shadowspill_failure_canary", "DEF"
)
_BROKEN_OP.define(  # type: ignore[no-untyped-call]
    "cpu_only(Tensor value) -> Tensor"
)
_BROKEN_OP.define(  # type: ignore[no-untyped-call]
    "workspace_oom(Tensor value) -> Tensor"
)
_BROKEN_OP.define(  # type: ignore[no-untyped-call]
    "unsupported_compile(Tensor value) -> Tensor"
)


@torch.library.impl(_BROKEN_OP, "cpu_only", "CPU")
def _cpu_only(value: torch.Tensor) -> torch.Tensor:
    return value.sin()


@torch.library.register_fake(  # type: ignore[untyped-decorator]
    "shadowspill_failure_canary::cpu_only"
)
def _cpu_only_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


@torch.library.impl(_BROKEN_OP, "workspace_oom", "CPU")
def _workspace_oom_cpu(value: torch.Tensor) -> torch.Tensor:
    return value + 1


@torch.library.impl(_BROKEN_OP, "workspace_oom", "CUDA")
def _workspace_oom_cuda(value: torch.Tensor) -> torch.Tensor:
    workspace = torch.empty(2 << 30, dtype=torch.uint8, device=value.device)
    del workspace
    # CUDAPluggableAllocator does not turn a null callback result into an
    # immediate Python exception. Do no further CUDA work after the deliberate
    # allocation failure; profiling after_task observes the latched failure.
    return value


@torch.library.register_fake(  # type: ignore[untyped-decorator]
    "shadowspill_failure_canary::workspace_oom"
)
def _workspace_oom_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


@torch.library.impl(_BROKEN_OP, "unsupported_compile", "CPU")
def _unsupported_compile_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.sin()


@torch.library.impl(_BROKEN_OP, "unsupported_compile", "CUDA")
def _unsupported_compile_cuda(value: torch.Tensor) -> torch.Tensor:
    return value.sin()


@torch.library.register_fake(  # type: ignore[untyped-decorator]
    "shadowspill_failure_canary::unsupported_compile"
)
def _unsupported_compile_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


class _MissingCudaImplementation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            torch.ops.shadowspill_failure_canary.cpu_only.default(value * self.scale),
        )


class _UnsupportedCompilation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            torch.ops.shadowspill_failure_canary.unsupported_compile.default(
                value * self.scale
            ),
        )


class _LargeStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2048, 2048, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.projection(value))


class _ProfilingOOM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            torch.ops.shadowspill_failure_canary.workspace_oom.default(
                value * self.scale
            ),
        )


def _imported(runtime: Runtime, model: nn.Module) -> nn.Module:
    return import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )


def _release(runtime: Runtime, model: nn.Module) -> None:
    export_model_state(model, runtime=runtime, release_runtime=True)


def _expect(
    expected: type[BaseException],
    operation: Callable[[], object],
    *,
    text: str,
) -> BaseException:
    try:
        result = operation()
    except expected as error:
        if text not in str(error):
            raise AssertionError(f"planning error omitted {text!r}: {error}") from error
        return error
    except BaseException as error:
        raise AssertionError(
            f"expected {expected.__name__}, received {type(error).__name__}: {error}"
        ) from error
    close = getattr(result, "close", None)
    if callable(close):
        close()
    raise AssertionError(
        f"planning unexpectedly succeeded; expected {expected.__name__}"
    )


def _plan(
    model: nn.Module,
    inputs: list[Any],
    runtime: Runtime,
    cache: str,
    execution_budget: int | None = None,
    spill_budget: int | None = None,
) -> Any:
    return plan_forward(
        model,
        example_inputs=inputs,
        runtime=runtime,
        execution="execution",
        spill="spill",
        planning_cachedir=cache,
        force_fresh=True,
        save_plan=False,
        verbose=False,
        execution_budget=execution_budget,
        spill_budget=spill_budget,
    )


def main() -> int:
    adapter = Path(sys.argv[1]).resolve()
    runtime = Runtime(
        pools={
            "execution": device(
                physical_capacity=2 << 30,
                provider_headroom=512 << 20,
            ),
            "spill": pinned_host(capacity=1 << 30),
        },
        library_path=adapter,
    )
    with tempfile.TemporaryDirectory() as cache:
        inputs = torch.randn(2, 2048)

        invalid = _imported(runtime, _DataDependentModel())
        try:
            error = _expect(
                CaptureError,
                lambda: _plan(invalid, [inputs], runtime, cache),
                text="forward graph capture failed",
            )
            if not any("capture_lowering" in note for note in error.__notes__):
                raise AssertionError("capture failure omitted its planning phase")
        finally:
            _release(runtime, invalid)

        unsupported = _imported(runtime, _UnsupportedCompilation())
        try:
            with inductor_config.patch(implicit_fallbacks=False):
                error = _expect(
                    CompilationError,
                    lambda: _plan(unsupported, [inputs], runtime, cache),
                    text="missing lowering",
                )
            if not isinstance(error, CompilationError):
                raise AssertionError("compiler failure lost its public error type")
            if error.structural_contract is None or not error.operators:
                raise AssertionError("compiler failure omitted structural context")
            if error.__cause__ is None:
                raise AssertionError("compiler failure lost its PyTorch cause")
        finally:
            _release(runtime, unsupported)

        broken = _imported(runtime, _MissingCudaImplementation())
        try:
            error = _expect(
                ProfilingError,
                lambda: _plan(broken, [inputs], runtime, cache),
                text="shadowspill_failure_canary::cpu_only",
            )
            if not any(
                "compiler_manifest" in note or "structural_profiling" in note
                for note in error.__notes__
            ):
                raise AssertionError("compiled-task failure omitted its phase")
            if not any("structural contract" in note for note in error.__notes__) and (
                not isinstance(error, ProfilingError)
                or error.structural_contract is None
            ):
                raise AssertionError("profile failure omitted its structural contract")
            if not isinstance(error.__cause__, RuntimeError):
                raise AssertionError("profile failure lost its provider cause")
        finally:
            _release(runtime, broken)

        profiling_oom = _imported(runtime, _ProfilingOOM())
        try:
            error = _expect(
                ProfilingError,
                lambda: _plan(profiling_oom, [inputs], runtime, cache),
                text="ShadowSpill no-progress OOM",
            )
            if not isinstance(error.__cause__, torch.OutOfMemoryError):
                raise AssertionError("profiling OOM lost its allocator cause")
        finally:
            _release(runtime, profiling_oom)

        constrained = _imported(runtime, _LargeStage())
        try:
            _expect(
                AdmissionError,
                lambda: _plan(
                    constrained,
                    [inputs],
                    runtime,
                    cache,
                    spill_budget=64 << 20,
                ),
                text="spill-pool budget",
            )
            infeasible = _expect(
                PlanInfeasibleError,
                lambda: _plan(
                    constrained,
                    [inputs],
                    runtime,
                    cache,
                    execution_budget=(
                        # Leave two MiB beyond the minimum workspace reserve.
                        # Admission topology is valid, but the stage's
                        # parameter/input floor cannot fit at preflight.
                        (514 << 20) + runtime._installed.fixed_execution_bytes
                    ),
                ),
                text="could not construct a feasible memory schedule",
            )
            if not isinstance(infeasible, PlanInfeasibleError):
                raise AssertionError("infeasible plan lost its public error type")
            if infeasible.kind not in {"required_capacity", "analytic_capacity"}:
                raise AssertionError(
                    f"unexpected infeasibility kind {infeasible.kind!r}"
                )
            notes = tuple(getattr(infeasible, "__notes__", ()))
            if not any("'feasibility_preflight'" in note for note in notes):
                raise AssertionError(
                    "irreducible task capacity was not rejected by preflight"
                )
            if any("'pressurefit_simulation'" in note for note in notes):
                raise AssertionError(
                    "irreducible task capacity incorrectly entered PressureFit"
                )

            planned = _plan(constrained, [inputs], runtime, cache)
            actual = planned([inputs])
            if not isinstance(actual, torch.Tensor) or not torch.isfinite(actual).all():
                raise AssertionError("valid plan after rollback produced bad output")
            planned.close()
        finally:
            _release(runtime, constrained)
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
