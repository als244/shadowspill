"""Saved Quack choices: CPU dispatch and marked CUDA kernel replay checks."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from functools import wraps

import pytest
import torch

from shadowspill.pytorch.tuning import capture_choices, use_choices
from shadowspill.pytorch.tuning import quack as adapter

quack = pytest.importorskip("quack.autotuner")


def selected(value, *, length, tile):
    if value.numel() != length:
        raise ValueError("incorrect length")
    return tile


def fixture(selected_index=0, *, configs=None, function=selected):
    configs = configs or [quack.AutotuneConfig(tile=16), quack.AutotuneConfig(tile=32)]
    tuner = quack.Autotuner(function, key=["length"], configs=configs)
    tensor = torch.ones(4)
    # The actual provider's key, specified independently of the adapter.
    key = (4, (4,), (1,), torch.float32)
    tuner.cache[key] = configs[selected_index]
    return tuner, tensor, key


def captured(tuner, tensor):
    with capture_choices() as choices:
        result = tuner(tensor, length=tensor.numel())
    return json.loads(json.dumps(choices)), result


def forbidden(*args, **kwargs):
    raise AssertionError("saved invocation tried to tune or consult a cache")


def test_replay_ignores_empty_changed_and_disk_caches(monkeypatch):
    tuner, tensor, key = fixture()
    choices, result = captured(tuner, tensor)
    assert result == 16 and len(choices) == 1
    choice = next(iter(choices.values()))
    assert choice["backend"] == "quack" and choice["version"] == 1
    assert choice["config"] == {"tile": 16}
    for name in ("_bench", "prune_configs", "check_disk_cache"):
        monkeypatch.setattr(tuner, name, forbidden)
    replay = use_choices(tuner, choices)
    tuner.cache.clear()
    assert replay(tensor, length=4) == 16
    assert tuner.cache == {}
    tuner.cache[key] = tuner.configs[1]
    assert replay(tensor, length=4) == 16
    assert tuner.best_config is tuner.configs[0]
    assert tuner.cache[key] is tuner.configs[1]
    # Leaving the scope restores ordinary provider selection.
    assert tuner(tensor, length=4) == 32


@pytest.mark.parametrize("change", ["shape", "stride", "dtype", "scalar", "function"])
def test_unseen_specializations_refuse_before_kernel_or_tuning(change, monkeypatch):
    tuner, tensor, _ = fixture()
    choices, _ = captured(tuner, tensor)
    length = 4
    if change == "shape":
        tensor = tensor.reshape(2, 2)
    elif change == "stride":
        tensor = torch.ones(8)[::2]
    elif change == "dtype":
        tensor = tensor.double()
    elif change == "scalar":
        length = 8
    else:
        tuner.fn = forbidden
    monkeypatch.setattr(tuner, "_bench", forbidden)
    with pytest.raises(ValueError, match="no finalized configuration"):
        use_choices(tuner, choices)(tensor, length=length)


@pytest.mark.parametrize(
    "change", ["candidate", "negative", "config", "source", "version"]
)
def test_incompatible_records_refuse(change):
    tuner, tensor, _ = fixture()
    choices, _ = captured(tuner, tensor)
    choice = next(iter(choices.values()))
    if change == "candidate":
        tuner.configs.reverse()
    elif change == "negative":
        choice["candidate"] = -1
    elif change == "config":
        choice["config"]["tile"] = 999
    elif change == "source":
        choice["source"] = "changed"
    else:
        choice["version"] = 999
    with pytest.raises(ValueError, match=r"Quack .* (differ|version)"):
        use_choices(tuner, choices)(tensor, length=4)


def test_single_candidate_and_legacy_missing_record_are_checked():
    tuner, tensor, _ = fixture(configs=[quack.AutotuneConfig(tile=16)])
    choices, result = captured(tuner, tensor)
    assert result == 16 and len(choices) == 1
    assert use_choices(tuner, choices)(tensor, length=4) == 16
    with pytest.raises(ValueError, match="rebuild the Program"):
        use_choices(tuner, {})(tensor, length=4)


def test_nested_scopes_and_exception_cleanup():
    tuner, tensor, key = fixture()
    first, _ = captured(tuner, tensor)
    tuner.cache[key] = tuner.configs[1]
    second, _ = captured(tuner, tensor)
    inner = use_choices(tuner, second)

    def outer():
        assert tuner(tensor, length=4) == 16
        assert inner(tensor, length=4) == 32
        assert tuner(tensor, length=4) == 16
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        use_choices(outer, first)()
    assert tuner(tensor, length=4) == 32
    assert tuner.cache[key] is tuner.configs[1]
    with capture_choices() as outer_choices:
        assert tuner(tensor, length=4) == 32
        with capture_choices() as inner_choices:
            assert tuner(tensor, length=4) == 32
        assert outer_choices == inner_choices == second


def test_conflicting_winners_and_generated_candidates_are_rejected():
    tuner, tensor, key = fixture()
    with capture_choices():
        assert tuner(tensor, length=4) == 16
        tuner.cache[key] = tuner.configs[1]
        with pytest.raises(ValueError, match="conflicting Quack"):
            tuner(tensor, length=4)
    tuner.cache[key] = quack.AutotuneConfig(tile=64)
    with pytest.raises(ValueError, match="generated Quack candidate"):
        captured(tuner, tensor)


def test_closure_values_distinguish_generated_callables():
    def factory(offset):
        def function(value, *, length, tile):
            return tile + offset

        return function

    first, tensor, _ = fixture(function=factory(1))
    second, _, _ = fixture(function=factory(2))
    choices, result = captured(first, tensor)
    assert result == 17
    with pytest.raises(ValueError, match="no finalized configuration"):
        use_choices(second, choices)(tensor, length=4)


def test_gemm_dataclass_configuration_round_trips():
    from quack.gemm_config import GemmConfig

    def function(value, *, length, config):
        return config.tile_m

    tuner, tensor, key = fixture(
        function=function,
        configs=[
            quack.AutotuneConfig(config=GemmConfig(tile_m=64)),
            quack.AutotuneConfig(config=GemmConfig(tile_m=128)),
        ],
    )
    choices, result = captured(tuner, tensor)
    assert result == 64
    tuner.cache[key] = tuner.configs[1]
    assert use_choices(tuner, choices)(tensor, length=4) == 64


def test_generated_epilogue_and_reduction_sink_identities():
    from quack.epilogue.frontend import gemm_epilogue
    from quack.epilogue.ops import ColVecReduce
    from quack.gemm_runtime.autotune import _make_tuned_fn

    @gemm_epilogue(reduces={"total": ColVecReduce("total")})
    def sum_epilogue(acc):
        return {"D": acc, "total": acc}

    @gemm_epilogue(reduces={"total": ColVecReduce("total", combine="max")})
    def max_epilogue(acc):
        return {"D": acc, "total": acc}

    first = _make_tuned_fn(sum_epilogue, ("total",))
    second = _make_tuned_fn(max_epilogue, ("total",))
    identity = adapter._value(first)
    assert json.loads(json.dumps(identity)) == identity
    assert identity == adapter._value(_make_tuned_fn(sum_epilogue, ("total",)))
    assert identity != adapter._value(second)
    assert identity != adapter._value(_make_tuned_fn(sum_epilogue, ("total", "extra")))


def test_unknown_callable_state_and_changed_provider_contract_refuse():
    unknown = object()

    def function(value, *, length, tile):
        return unknown

    tuner, tensor, _ = fixture(function=function)
    with pytest.raises(ValueError, match="needs a provider initializer"):
        captured(tuner, tensor)
    with capture_choices(), pytest.raises(ValueError, match="unsupported Quack"):
        adapter._run(forbidden, False, tuner, (tensor,), {"length": 4})


def test_fresh_process_replay_without_tuning_cache(tmp_path):
    import quack.cache

    from tests.shadowspill.offline._example import program

    tuner, tensor, _ = fixture()
    choices, _ = captured(tuner, tensor)
    artifact = replace(program(), bindings={"triton_choices": {"update_impl": choices}})
    artifact.save(tmp_path / "program.ssp")
    saved = tmp_path / "choices.json"
    saved.write_text(
        json.dumps(
            {
                "choices": choices,
                "sources": list(map(str, quack.cache.EXTRA_SOURCE_DIRS)),
            }
        )
    )
    script = """
import json, sys, torch, quack.cache
from pathlib import Path
from shadowspill import Program
from tests.shadowspill.build.test_quack_tuning import fixture, forbidden
from shadowspill.pytorch.tuning import use_choices
saved = json.load(open(sys.argv[1]))
program = Program.load(Path(sys.argv[1]).with_name('program.ssp'))
choices = program.bindings['triton_choices']['update_impl']
quack.cache.EXTRA_SOURCE_DIRS[:] = saved['sources']
tuner, tensor, key = fixture(1)
tuner._bench = tuner.check_disk_cache = tuner.prune_configs = forbidden
assert use_choices(tuner, choices)(tensor, length=4) == 16
assert tuner.cache[key] is tuner.configs[1]
assert not torch.cuda.is_initialized()
print('PASS: saved Quack choice in fresh CPU process')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(saved)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS: saved Quack choice" in completed.stdout


def test_captured_records_do_not_share_mutable_input_state():
    tuner, tensor, _ = fixture()
    choices, _ = captured(tuner, tensor)
    expected = copy.deepcopy(choices)
    replay = use_choices(tuner, choices)
    next(iter(choices.values()))["config"]["tile"] = 999
    assert replay(tensor, length=4) == 16
    assert captured(tuner, tensor)[0] == expected


def test_registration_after_first_fingerprint_changes_identity(tmp_path, monkeypatch):
    import quack.cache

    tuner, tensor, _ = fixture()
    choices, _ = captured(tuner, tensor)
    (tmp_path / "kernel.py").write_text("KERNEL_VERSION = 2\n")
    monkeypatch.setattr(
        quack.cache, "EXTRA_SOURCE_DIRS", [*quack.cache.EXTRA_SOURCE_DIRS, tmp_path]
    )
    with pytest.raises(ValueError, match="provider source differs"):
        use_choices(tuner, choices)(tensor, length=4)


def test_mixed_triton_and_quack_choices_keep_legacy_triton_keys():
    from shadowspill.pytorch.tuning import triton as triton_adapter
    from tests.shadowspill.build.test_provider_tuning import square

    def triton_dispatch(tuner, *args):
        # CPU dispatch control only. The existing CUDA test covers kernels.
        key = (args[2], "torch.float32", "torch.float32")
        tuner.best_config = tuner.cache.get(key, tuner.configs[0])
        return tuner.best_config.kwargs["BLOCK"]

    tuner, tensor, _ = fixture()

    def invoke():
        return (
            tuner(tensor, length=4),
            triton_adapter._run(triton_dispatch, square, (tensor, tensor, 4), {}),
        )

    with capture_choices() as choices:
        assert invoke() == (16, 64)
    assert len(choices) == 2
    assert sum(key.startswith("quack/v1/") for key in choices) == 1
    assert use_choices(invoke, json.loads(json.dumps(choices)))() == (16, 64)


def test_optional_quack_installation(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name="quack")

    monkeypatch.setattr(adapter, "_installed", False)
    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    adapter.install()
    assert not adapter._installed


def test_unrecognized_call_wrapper_is_not_hidden_by_wrapped_metadata(monkeypatch):
    original = quack.Autotuner.__call__

    @wraps(original)
    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(quack.Autotuner, "__call__", wrapped)
    monkeypatch.setattr(adapter, "_installed", False)
    tuner, tensor, _ = fixture()
    with pytest.raises(ValueError, match="unsupported Quack Autotuner call contract"):
        captured(tuner, tensor)


@pytest.mark.cuda
def test_actual_gemm_and_generated_epilogue_replay_without_tuning(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from quack.epilogue.library import scaled_residual
    from quack.gemm_interface import gemm, gemm_tuned
    from quack.gemm_runtime.autotune import _MOD_TUNERS

    generator = torch.Generator(device="cuda").manual_seed(731)
    left = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda", generator=generator)
    right = torch.randn(
        64, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    residual = torch.randn(
        64, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )

    def invoke(a, b, c):
        product = gemm(a, b, tuned=True)
        destination = torch.empty_like(c)
        scaled_residual.gemm_tuned(
            a, b, destination, c, epi_args={"alpha": 0.5}, b_kn=True
        )
        return product, destination

    with capture_choices() as choices:
        initial = invoke(left, right, residual)
    kernels = {
        value["kernel"] for key, value in choices.items() if key.startswith("quack/v1/")
    }
    assert any(name.endswith(".gemm_tuned") for name in kernels)
    assert any(".mod_scaled_residual_" in name for name in kernels)
    expected = left.float() @ right.float()
    torch.testing.assert_close(initial[0], expected.bfloat16())
    torch.testing.assert_close(
        initial[1], (0.5 * expected + residual.float()).bfloat16()
    )
    for name in ("_bench", "prune_configs", "check_disk_cache"):
        monkeypatch.setattr(quack.Autotuner, name, forbidden)
    tuners = [gemm_tuned, *_MOD_TUNERS.values()]
    previous = [dict(tuner.cache) for tuner in tuners]
    try:
        for tuner in tuners:
            tuner.cache.clear()
        replay = use_choices(invoke, json.loads(json.dumps(choices)))
        actual = replay(left, right, residual)
        for result, wanted in zip(actual, initial, strict=True):
            torch.testing.assert_close(result, wanted, rtol=0, atol=0)
        assert all(not tuner.cache for tuner in tuners)
        with pytest.raises(ValueError, match="no finalized configuration"):
            replay(left[:-1], right, residual[:-1])
    finally:
        for tuner, cache in zip(tuners, previous, strict=True):
            tuner.cache.update(cache)
