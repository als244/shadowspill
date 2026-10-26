"""The reference stays independent of preparation and observes only requested steps."""

import os
import subprocess
import sys

from tests.training._text_recipe import _config, _records
from workloads.recipes.text.run import run_config


def test_reference_runs_without_legacy_preparation_or_execution(tmp_path):
    path = _config(tmp_path)
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for name in ('shadowspill.training._legacy_trainer',
                     'shadowspill.training.backends', 'shadowspill.execution'):
            if fullname == name or fullname.startswith(name + '.'):
                raise AssertionError('reference imported ' + fullname)
sys.meta_path.insert(0, Block())
from workloads.recipes.text.run import run_config
result = run_config(sys.argv[1], ['steps=2', 'checkpoint_every=0', 'eval_every=0'])
assert result.step == 2
"""
    child = subprocess.run(
        [sys.executable, "-S", "-c", script, str(path)],
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
