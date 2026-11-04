"""Numerical configuration is validated without changing caller settings."""

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from shadowspill.pytorch import numerics
from shadowspill.pytorch.executable import restore_executable


@pytest.fixture
def changed_settings(monkeypatch):
    contract = numerics.capture_numerical_settings()
    matmul = torch.backends.cuda.matmul
    monkeypatch.setattr(
        matmul,
        "allow_bf16_reduced_precision_reduction",
        not matmul.allow_bf16_reduced_precision_reduction,
    )
    return contract


def _program(contract):
    return SimpleNamespace(
        executables={"task": b"not executable"},
        bindings={"task_metadata": {"task": {"numerical_contracts": (contract,)}}},
    )


def test_capture_is_cuda_free_and_supports_both_precision_apis():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from shadowspill.pytorch import numerics
def forbidden(*args, **kwargs):
    raise AssertionError('numerical configuration initialized CUDA')
torch.cuda._lazy_init = forbidden
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
initial = numerics.capture_numerical_settings()
torch.set_float32_matmul_precision('highest')
assert numerics.capture_numerical_settings() == initial
snapshots = []
for precision in ('highest', 'high', 'medium'):
    torch.set_float32_matmul_precision(precision)
    snapshots.append(numerics.capture_numerical_settings())
    numerics.validate_numerical_settings((snapshots[-1],))
assert len(set(snapshots)) == 3
for precision in ('ieee', 'tf32'):
    torch.backends.cuda.matmul.fp32_precision = precision
    contract = numerics.capture_numerical_settings()
    numerics.validate_numerical_settings((contract,))
assert not torch.cuda.is_initialized()
""",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mismatch_precedes_executable_loading(changed_settings):
    with pytest.raises(RuntimeError, match="allow_bf16_reduced_precision_reduction"):
        restore_executable(b"not executable", numerical_contracts=(changed_settings,))


@pytest.mark.parametrize("phase", ["finalize", "profile", "cached_profile", "run"])
def test_phase_rejects_configuration_changes_before_task_use(
    monkeypatch, changed_settings, phase
):
    program = _program(changed_settings)
    if phase == "finalize":
        from shadowspill.build import finalize

        monkeypatch.setattr(
            finalize,
            "prepare_resources",
            lambda *args: nullcontext(SimpleNamespace(groups=None)),
        )
        invoke = partial(finalize.finalize_tasks, program, torch.device("cpu"))
    elif phase in ("profile", "cached_profile"):
        from shadowspill.profiling.session import MeasurementSession

        if phase == "profile":
            invoke = partial(MeasurementSession, program, torch.device("cpu"), {}, 0, 0)
        else:
            session = MeasurementSession.__new__(MeasurementSession)
            session.program = program
            session.functions = {"task": object()}
            invoke = partial(session.function, "task")
    else:
        from shadowspill.execution import Executor

        executor = Executor.__new__(Executor)
        executor.synchronize = lambda: None
        executor.parameter_norms = False
        executor._independent = frozenset()
        executor._distributed = executor._observations = None
        executor._initialized = True
        executor._named_paths = {}
        # No bridge, controls, inputs or dispatcher may be touched on refusal.
        executor._paths = (SimpleNamespace(numerical_contracts=(changed_settings,)),)
        invoke = executor.run
    with pytest.raises(RuntimeError, match=r"expected .* actual"):
        invoke()


def test_collective_validation_reports_local_failure_to_peers(changed_settings):
    messages = []
    with pytest.raises(RuntimeError, match="allow_bf16_reduced_precision_reduction"):
        numerics.validate_numerical_settings((changed_settings,), check=messages.append)
    assert len(messages) == 1 and "actual" in messages[0]
    messages.clear()
    numerics.validate_numerical_settings((), check=messages.append)
    assert messages == [None]


def test_composition_refuses_conflicting_or_missing_settings(changed_settings):
    current = numerics.capture_numerical_settings()
    with pytest.raises(ValueError, match="conflicting"):
        numerics.compose_numerical_contracts([(changed_settings,), (current,)])
    with pytest.raises(ValueError, match="missing"):
        numerics.compose_numerical_contracts([(current,), ()])
    contracts = numerics.compose_numerical_contracts(
        [(current,), (current, "cuda.sigmoid_backward.scalar_rounding.v1")]
    )
    assert set(contracts) == {current, "cuda.sigmoid_backward.scalar_rounding.v1"}


@pytest.mark.parametrize("damage", ["version", "json", "field", "type"])
def test_malformed_settings_cannot_fall_back_to_legacy(damage):
    contract = numerics.capture_numerical_settings()
    prefix, payload = contract.split(":", 1)
    settings = json.loads(payload)
    if damage == "version":
        contract = prefix.replace("v1", "v999") + ":" + payload
    elif damage == "json":
        contract = prefix + ":invalid"
    else:
        if damage == "field":
            del settings["deterministic_algorithms"]
        else:
            settings["deterministic_algorithms"] = 1
        contract = prefix + ":" + json.dumps(settings)
    with pytest.raises(ValueError, match="Torch numerical settings"):
        numerics.validate_numerical_settings((contract,))


def test_legacy_artifacts_and_unrelated_rounding_contracts_remain_readable():
    numerics.validate_numerical_settings(())
    numerics.validate_numerical_settings(("cuda.sigmoid_backward.scalar_rounding.v1",))


def test_program_contract_selection_includes_only_requested_tasks(changed_settings):
    current = numerics.capture_numerical_settings()
    program = _program(changed_settings)
    program.executables["other"] = b"other"
    program.bindings["task_metadata"]["other"] = {"numerical_contracts": (current,)}
    assert numerics.program_numerical_contracts(program, ["other"]) == (current,)
    assert set(numerics.program_numerical_contracts(program)) == {
        current,
        changed_settings,
    }
