"""Failed preparation leaves reusable inputs and permits a later cold execution."""

import gc

import pytest
import torch
from torch import nn
from torch._inductor import config

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    Program,
    apply_profiles,
    build_forward,
    measure_program,
    plan_program,
)
from shadowspill.errors import (
    CaptureError,
    CompilationError,
    PlanInfeasibleError,
    ProfilingError,
)
from shadowspill.runtime.bootstrap import installed_runtime
from tests.shadowspill.execution._oracle import oracle
from tests.shadowspill.execution.helpers import run_worker

from . import _failure_models as models


def check_task_error(error, operator):
    assert error.structural_contract
    assert error.task_kind
    assert any(operator in target for target in error.operators)
    assert error.__cause__ is not None


def build_failures(inputs):
    model = models.DataDependent()
    before = model.scale.detach().clone()
    with pytest.raises(CaptureError, match=r"forward.*capture") as caught:
        build_forward(model, example_args=(inputs,))
    assert caught.value.__cause__ is not None
    torch.testing.assert_close(model.scale, before, rtol=0, atol=0)

    with (
        config.patch(implicit_fallbacks=False),
        pytest.raises(CompilationError, match="missing lowering") as caught,
    ):
        build_forward(models.Opaque(), example_args=(inputs,))
    check_task_error(caught.value, "shadowspill_phase_failure.opaque")

    with pytest.raises(CompilationError, match="kernel finalization") as caught:
        build_forward(models.Opaque(missing_backend=True), example_args=(inputs,))
    check_task_error(caught.value, "shadowspill_phase_failure.cpu_only")
    assert isinstance(caught.value.__cause__, RuntimeError)


def profile_failure(tmp_path, inputs):
    program = build_forward(models.Opaque(), example_args=(inputs,))
    program.save(tmp_path / "program.ssp")
    program = Program.load(tmp_path / "program.ssp")
    options = dict(warmup_seconds=0.001, measure_seconds=0.002)
    models.profiling_failure = True
    try:
        with pytest.raises(ProfilingError, match="injected provider") as caught:
            measure_program(program, **options)
        check_task_error(caught.value, "shadowspill_phase_failure.opaque")
        assert isinstance(caught.value.__cause__, torch.OutOfMemoryError)
    finally:
        models.profiling_failure = False
    assert apply_profiles(program, measure_program(program, **options)).task_profiles


@pytest.mark.cuda
def test_phase_failures_leave_preparation_reusable(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires a representative GPU")
    inputs = torch.randn(2, 64)
    build_failures(inputs)
    profile_failure(tmp_path, inputs)
    gc.collect()  # Release caught tracebacks before testing recovery.
    assert installed_runtime() is None
    assert torch.cuda.memory.get_allocator_backend() == "native"

    model = nn.Linear(64, 64)
    program = build_forward(model, example_args=(inputs,))
    profiled = apply_profiles(
        program, measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    )
    communication = CommunicationInfo(1e10, 1e10)
    for budget in (
        MemoryBudgets(2 << 30, 1),
        MemoryBudgets(profiled.fixed_device_bytes + 1024, 64 << 20),
    ):
        with pytest.raises(PlanInfeasibleError) as caught:
            plan_program(
                profiled, memory_budgets=budget, communication_info=communication
            )
        assert caught.value.kind
        if budget.spill_bytes == 1:
            assert caught.value.diagnostics
        else:
            assert caught.value.kind in {"required_capacity", "analytic_capacity"}

    plan = plan_program(
        profiled,
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=communication,
    )
    plan.save(tmp_path / "plan.ssp")
    torch.save(oracle(model, [((inputs,), {})], "forward"), tmp_path / "oracle.pt")
    run_worker(tmp_path)
