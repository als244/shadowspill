"""Packed text keeps exact resume and observations across saved phase workers."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from shadowspill.pytorch.guards import capture_input_signature
from shadowspill.training import config
from tests.training._text_recipe import _config, _records
from workloads.recipes.text.experiment import results
from workloads.recipes.text.experiment.inputs import call_inputs, candidates, source
from workloads.recipes.text.run import run_config


def test_packed_normalizer_is_a_runtime_tensor():
    parts = [[torch.ones(1, 8, dtype=torch.int64)] * 3] * 2
    first, second = [
        call_inputs({"candidates": {"batch": parts}, "normalizer": n}, "batch")
        for n in (7, 13)
    ]
    for calls, denominator in ((first, 7), (second, 13)):
        assert len(calls["args"]) == 2
        assert all(
            args[1].item() == pytest.approx(1 / denominator) for args in calls["args"]
        )
        assert all(args[1].shape == () for args in calls["args"])


def test_build_examples_match_actual_packed_layouts(tmp_path):
    options = json.loads(_config(tmp_path).read_text())
    data = config.resolve(options["data"])
    example, _, geometries, _ = candidates(options, data)
    for name, (tokens, count) in geometries.items():
        expected = capture_input_signature(call_inputs(example, name)["args"][0])
        actual = next(
            source(
                data,
                options,
                {
                    "candidate": name,
                    "tokens_per_microbatch": tokens,
                    "microbatches": count,
                },
            )
        )
        for args in actual["args"]:
            expected.validate(args)


@pytest.mark.cuda
@pytest.mark.parametrize("masters", [False, True])
def test_text_phases_preserve_packing_metrics_and_exact_cold_resume(
    tmp_path, monkeypatch, masters
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    path = _config(tmp_path)
    options = json.loads(path.read_text())
    options["engine"] = "shadowspill"
    # Exact resume checks require deterministic reductions. The ordinary CUDA
    # embedding backward uses atomics whose summation order can vary by launch.
    options["model"]["deterministic_embedding"] = True
    if masters:
        options["model"]["dtype"] = "bfloat16"
        options.update(master_dtype="@torch:float32", grad_dtype="@torch:float32")
    else:
        options["optimizer"] = "@mlops.optim:AdamW"
        options["optimizer_args"].update(
            gradient_dtype="@torch:float32", opt_state_dtype="@torch:float32"
        )
    # Keep variable document boundaries/packing, but use learnable text rather
    # than independent random next tokens for the loss-decrease assertion.
    for split in ("train", "val"):
        tokens = np.memmap(
            Path(options["data"]["directory"]) / f"{split}.bin",
            dtype=np.uint32,
            mode="r+",
        )
        tokens[tokens != 0] = 1
        tokens.flush()
        del tokens
    options.update(
        execution_gib=4,
        spill_gib=0.125,
        initialize="@tests.training._synthetic:fill_tiny_tensor",
        parameter_norms_every=1,
        warmup_seconds=0.001,
        measure_seconds=0.002,
        plots=False,
        startup_diagnostics=not masters,
    )
    options["optimizer_args"]["lr"] = {"@call": "torch:tensor", "data": 0.01}
    options["schedules"]["optimizer_group.0.lr"] = options["schedules"].pop("lr")
    path.write_text(json.dumps(options))
    caches = tmp_path / "compiler-cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(caches / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(caches / "triton"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"

    def run(root, overrides=(), phase="all"):
        try:
            return run_config(path, [f"run_dir={root}", *overrides], phase=phase)
        except subprocess.CalledProcessError:
            logs = "\n".join(
                f"{p.name}:\n{p.read_text()}" for p in (root / "logs").glob("*.log")
            )
            pytest.fail(logs)

    final = run(whole)
    assert final.step == 6
    if not masters:
        assert (whole / "startup/timelines/traced.html").is_file()
        startup = json.loads((whole / "startup/summary.json").read_text())
        assert startup["training_step"] == 0
    assert final.parameter_metrics["embed.weight"]["grad_norm"].device.type == "cpu"
    torch.testing.assert_close(
        asdict(final), asdict(results.load(whole / "result.json"))
    )
    expected = _records(whole, "train/loss")
    assert expected[6]["train/loss"] < expected[1]["train/loss"]
    assert list(_records(whole, "eval/loss")) == [3, 6]
    assert list(_records(whole, "train/grad_norm/global/l2")) == list(range(1, 7))
    assert len(json.loads((whole / "search.json").read_text())) == 2

    resumed.mkdir()
    shutil.copytree(whole / "plans", resumed / "plans")
    shutil.copy2(whole / "planning.json", resumed / "planning.json")
    shutil.rmtree(caches)
    run(resumed, ["steps=3"], phase="execute")
    checkpoint = resumed / "checkpoints/step_00000003.ssp"
    run(resumed, [f"resume={checkpoint}"], phase="execute")
    found = _records(resumed, "train/loss")
    for step, row in expected.items():
        for key in (
            "train/loss",
            "hyperparameters/optimizer_group.0.lr",
            "train/grad_norm/global/l2",
            "train/param_rms/embed/weight",
        ):
            assert found[step][key] == row[key]
    assert (
        _records(whole, "eval/loss")[6]["eval/loss"]
        == _records(resumed, "eval/loss")[6]["eval/loss"]
    )
    assert (whole / "packing.jsonl").read_bytes() == (
        resumed / "packing.jsonl"
    ).read_bytes()

    def stored_tensors(root):
        with zipfile.ZipFile(root / "checkpoints/step_00000006.ssp") as archive:
            manifest = json.loads(archive.read("manifest.json"))
            return {v["name"]: v["sha256"] for v in manifest["tensors"]}

    assert stored_tensors(whole) == stored_tensors(resumed)
    child = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "from workloads.recipes.text.experiment.plan import run; run(sys.argv[1])",
            str(whole),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
