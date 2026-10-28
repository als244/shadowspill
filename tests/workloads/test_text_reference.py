"""The reference stays independent of preparation and observes only requested steps."""

import json
import os
import subprocess
import sys

from tests.training._text_recipe import _config, _records
from workloads.recipes.text.run import run_config


def test_reference_uses_shared_bounded_initializer_and_plan_schedule_names(tmp_path):
    import torch

    from shadowspill.artifacts.tensors import TensorChunk
    from shadowspill.training._model import initialize_model
    from tests.training._synthetic import TinyModel, fill_tiny_tensor
    from workloads.recipes.text.reference.initialization import initializer

    options = {"initialize": "@tests.training._synthetic:fill_tiny_tensor"}
    with torch.device("meta"):
        model = TinyModel()
    with initializer(options) as fill:
        initialize_model(model, initialize=fill)
    for name, parameter in model.named_parameters():
        expected = torch.empty_like(parameter).view(-1)
        fill_tiny_tensor(
            TensorChunk(
                name, tuple(parameter.shape), parameter.dtype, 0, parameter.numel()
            ),
            expected,
        )
        torch.testing.assert_close(parameter.view(-1), expected, rtol=0, atol=0)

    path = _config(tmp_path)
    raw = json.loads(path.read_text())
    raw.update(options)
    raw["schedules"]["optimizer_group.0.lr"] = raw["schedules"].pop("lr")
    path.write_text(json.dumps(raw))
    result = run_config(path, ["steps=2", "checkpoint_every=0", "eval_every=0"])
    assert result.step == 2
    assert result.hyperparams["optimizer_group.0.lr"] == 0.01


def test_reference_runs_without_preparation_or_executor(tmp_path):
    path = _config(tmp_path)
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for name in ('shadowspill.build', 'shadowspill.profiling.measure',
                     'shadowspill.execution'):
            if fullname == name or fullname.startswith(name + '.'):
                raise AssertionError('reference imported ' + fullname)
sys.meta_path.insert(0, Block())
from workloads.recipes.text.run import run_config
result = run_config(sys.argv[1], ['steps=2', 'checkpoint_every=0', 'eval_every=0'])
assert result.step == 2
"""
    child = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            script,
            str(path),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_reference_norms_are_off_by_default_and_follow_cadence(tmp_path):
    path = _config(tmp_path)
    defaults = ["steps=4", "checkpoint_every=0", "eval_every=0"]
    unobserved = tmp_path / "unobserved"
    last = run_config(path, [*defaults, f"run_dir={unobserved}"])
    assert last.parameter_metrics == {}
    assert not _records(unobserved, "train/grad_norm/global/l2")
    observed = tmp_path / "observed"
    last = run_config(
        path, [*defaults, f"run_dir={observed}", "parameter_norms_every=2"]
    )
    assert last.parameter_metrics
    assert list(_records(observed, "train/grad_norm/global/l2")) == [2, 4]
    assert {
        step: values["train/loss"]
        for step, values in _records(unobserved, "train/loss").items()
    } == {
        step: values["train/loss"]
        for step, values in _records(observed, "train/loss").items()
    }
