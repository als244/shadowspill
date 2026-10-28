"""A bandwidth sweep must need only persisted profiles and CPU planning."""

import json
import os
import subprocess
import sys

import pytest

from benchmarking.bandwidth_frontier.cli import main
from tests.shadowspill.offline._example import profiled


def arguments(tmp_path):
    source = tmp_path / "profiled.ssp"
    profiled().save(source)
    return [
        "--program",
        f"tiny={source}",
        "--budget-gib",
        str(99 / (1 << 30)),
        str(200 / (1 << 30)),
        "--spill-gib",
        str(64 / (1 << 30)),
        "--sim-transfer-bandwidths",
        "0.064/0.064,0.128/0.128",
        "--output-dir",
        str(tmp_path / "frontier"),
        "--no-plots",
    ]


def test_full_sweep_is_cpu_only_and_retains_failures(tmp_path):
    args = arguments(tmp_path)
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch' or any(
            fullname == key or fullname.startswith(key + '.') for key in (
                'shadowspill.build', 'shadowspill.execution', 'shadowspill.runtime',
                'shadowspill.profiling.measure', 'shadowspill.store', 'workloads',
            )
        ):
            raise AssertionError('CPU sweep imported ' + fullname)
sys.meta_path.insert(0, Block())
from benchmarking.bandwidth_frontier.cli import main
main(sys.argv[1:])
"""
    child = subprocess.run(
        [sys.executable, *(["-S"] if sys.flags.no_site else []), "-c", script, *args],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    root = tmp_path / "frontier/raw_data"
    rows = json.loads((root / "frontier.json").read_text())
    assert len(rows) == 4
    assert len([r for r in rows if r["status"] == "PlanInfeasibleError"]) == 2
    winners = [r for r in rows if r["is_winner"]]
    assert len(winners) == 2
    assert winners[1]["seconds"] < winners[0]["seconds"]
    assert all(r["simulation_only"] for r in winners)
    assert len(list(root.glob("calibration-*.ssp"))) == 2
    assert (root / "frontier.csv").is_file()


def test_resume_reuses_saved_plans_and_rejects_changed_inputs(tmp_path, monkeypatch):
    from benchmarking.bandwidth_frontier import run

    args = arguments(tmp_path)
    assert main(args) == 0

    def unexpected(*args, **kwargs):
        raise AssertionError("resume replanned a complete calibration")

    monkeypatch.setattr(run, "plan", unexpected)
    assert main([*args, "--resume"]) == 0
    with pytest.raises(ValueError, match="unchanged inputs"):
        main([*args, "--resume", "--latency-us", "1", "1"])
    changed = profiled()
    from dataclasses import replace

    replace(changed, program=replace(changed.program, name="changed")).save(
        tmp_path / "profiled.ssp"
    )
    with pytest.raises(ValueError, match="unchanged inputs"):
        main([*args, "--resume"])


def test_replot_uses_numeric_evidence_without_program_archives(tmp_path):
    args = arguments(tmp_path)
    assert main(args) == 0
    root = tmp_path / "frontier/raw_data"
    for path in root.glob("*.ssp"):
        path.unlink()
    (tmp_path / "profiled.ssp").unlink()
    assert main(["--replot", str(root), "--output-dir", str(tmp_path / "redrawn")]) == 0
    assert list((tmp_path / "redrawn/figures").glob("*.png"))


@pytest.mark.parametrize("rates", ["nan/1", "inf/1", "0/1", "-1/1", "1", "1/2/3"])
def test_invalid_rates_are_rejected(tmp_path, rates):
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path), "--sim-transfer-bandwidths", rates])
