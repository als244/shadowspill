"""The installed optimizer factory preserves the existing captured AdamW ABI."""

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine_tests.pytorch.optimizer.test_optimizer import _install_declared_state
from shadowspill.build.optimizer_parameters import local_optimizer_parameters
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.optimizer.sandbox import copy_optimizer_to_meta
from shadowspill.training.definitions import Registry, adamw
from shadowspill_client import Optimizer


def resolve(**settings):
    registry = Registry()
    adamw.register(registry)
    return registry.resolve_optimizer(
        "mlops/adamw@1", settings=Optimizer("mlops/adamw@1", **settings)
    )


def test_registration_and_configuration_do_not_import_framework_or_provider():
    script = """
import sys
for name in ('torch', 'triton', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.training.definitions import load_registry
from shadowspill_client import Optimizer
registry = load_registry(['shadowspill'])
resolved = registry.resolve_optimizer('mlops/adamw@1', settings=Optimizer(
    'mlops/adamw@1', precision={
        'opt_state_dtype': 'float32', 'master_dtype': 'float32'}))
assert resolved.config['opt_state_dtype'] == 'float32'
assert resolved.master_dtype == 'float32'
assert 'mlops' not in sys.modules
assert registry.resolve_optimizer(resolved.definition.id,
    settings=resolved.settings, source=resolved.source) == resolved
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "parameter"])
def test_factory_preserves_precision_trainable_selection_and_consumed_controls(dtype):
    resolved = resolve(
        precision={"gradient_dtype": dtype, "opt_state_dtype": "float32"},
        defaults={"lr": 0.03, "beta1": 0.8, "beta2": 0.95, "weight_decay": 0.02},
    )
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[0].requires_grad_(False)
    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer = resolved.create(
        model, {name: torch.tensor(value) for name, value in resolved.defaults.items()}
    )
    from mlops.optim import AdamW

    assert type(optimizer) is AdamW
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert list(map(id, group["params"])) == list(map(id, model[1].parameters()))
    assert group["opt_state_dtype"] == torch.float32
    assert group["gradient_dtype"] == (
        "parameter" if dtype == "parameter" else getattr(torch, dtype)
    )
    named = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    _install_declared_state(named, optimizer)
    captured = capture_optimizer(named, optimizer)
    resolved.definition.validate_capture(captured)
    assert captured.update_tasks
    assert all(
        torch.count_nonzero(value) == 0
        for state in optimizer.state.values()
        for value in state.values()
    )
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, initial[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"groups": {"head": {"lr": 0.1}}},
        {"precision": {"master_dtype": "int32"}},
        {"precision": {"gradient_dtype": "float64"}},
        {"precision": {"parameter_rounding": "truncate"}},
        {"config": {"maximize": 1}},
        {"config": {"amsgrad": True}},
        {
            "config": {"opt_state_dtype": "float16"},
            "precision": {"opt_state_dtype": "float32"},
        },
        {"defaults": {"lr": -1}},
        {"defaults": {"beta1": 1}},
        {"defaults": {"beta2": float("nan")}},
        {"defaults": {"eps": -1}},
        {"defaults": {"weight_decay": -1}},
        {"defaults": {"clip_grad_norm": 1}},
    ],
)
def test_unsupported_settings_fail_during_cpu_resolution(settings):
    with pytest.raises((TypeError, ValueError)):
        resolve(**settings)


@pytest.mark.parametrize("master", [None, "bfloat16", "float32"])
def test_bf16_rounding_and_master_storage_preserve_fp32_parameters(master):
    resolved = resolve(
        precision={
            "master_dtype": master,
            "gradient_dtype": "bfloat16",
            "opt_state_dtype": "bfloat16",
            "parameter_rounding": "stochastic",
            "opt_state_rounding": "stochastic",
        }
    )
    model = torch.nn.ParameterDict(
        {
            "weight": torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16)),
            "decay": torch.nn.Parameter(torch.ones(2, dtype=torch.float32)),
        }
    )
    original = resolved.create(
        model, {name: torch.tensor(value) for name, value in resolved.defaults.items()}
    )
    named = dict(model.named_parameters())
    copied, names = copy_optimizer_to_meta(
        original, {id(parameter): name for name, parameter in named.items()}
    )
    optimizer, names, copies = local_optimizer_parameters(
        copied,
        names,
        named,
        set(named),
        None if master is None else getattr(torch, master),
    )
    parameters = {names[id(p)]: p for p in optimizer.param_groups[0]["params"]}
    assert parameters["decay"].dtype == torch.float32
    assert set(copies) == ({"weight"} if master == "float32" else set())
    if master != "float32":
        assert optimizer is copied  # No redundant BF16 or FP32 master allocation.
        assert parameters["weight"].dtype == torch.bfloat16
    group = optimizer.param_groups[0]
    assert group["parameter_rounding"] == group["opt_state_rounding"] == "stochastic"
    _install_declared_state(parameters, optimizer)
    captured = capture_optimizer(parameters, optimizer, compute_copies=copies)
    resolved.definition.validate_capture(captured)
    assert sum(b.role == "compute_copy" for b in captured.bindings) == len(copies)
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.bfloat16


def test_controls_are_nonsticky_and_provider_sources_are_rechecked(
    tmp_path, monkeypatch
):
    resolved = resolve(defaults={"lr": 0, "beta1": 0, "beta2": 0})
    assert resolved.defaults["lr"] == 0
    assert resolve().defaults["lr"] == 0.001
    assert resolve().source == resolved.source
    with pytest.raises(ValueError, match="source changed"):
        registry = Registry()
        adamw.register(registry)
        registry.resolve_optimizer(
            resolved.definition.id,
            source=replace(resolved.source, digest="0" * 64),
        )

    (tmp_path / "__init__.py").write_text("# provider\n")
    provider = tmp_path / "kernel.py"
    provider.write_text("# first version\n")
    monkeypatch.setattr(
        adamw,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(tmp_path / "__init__.py")),
    )
    first = resolve().source
    provider.write_text("# different allocation implementation\n")
    assert resolve().source != first
    monkeypatch.setattr(adamw, "find_spec", lambda _: None)
    with pytest.raises(ValueError, match="installed MLOps provider"):
        resolve()
