"""The Llama application keeps data/setup on a framework-free client."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.train_llama3 import arguments, main
from examples.training.llama3.setup import APPLICATION, Settings, read_resume, start
from examples.training.text_data import TokenBatches
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client.protocol.requests import ArtifactRef


def corpus(path):
    path.mkdir()
    values = np.arange(97, dtype=np.uint32) % 32
    values.tofile(path / "train.bin")
    ((values + 1) % 32).tofile(path / "val.bin")
    (path / "meta.json").write_text(json.dumps({"vocab_size": 32}))
    return path


def test_global_windows_preserve_previous_shift_and_update_normalization(tmp_path):
    data = corpus(tmp_path / "data")
    batches = TokenBatches(data, 4, 3)
    assert len(batches) == 8
    first, following = batches[0], batches[1]
    for index, datum in enumerate(first):
        np.testing.assert_array_equal(
            datum.inputs["tokens"], np.arange(4 * index, 4 * index + 4)
        )
        np.testing.assert_array_equal(
            datum.loss_inputs["target_tokens"], datum.inputs["tokens"] + 1
        )
        np.testing.assert_allclose(datum.loss_inputs["weights"], 1 / 12)
    assert following[0].inputs["tokens"].tolist() == [12, 13, 14, 15]
    assert sum(float(d.loss_inputs["weights"].sum()) for d in first) == pytest.approx(1)
    first[0].inputs["tokens"][0] = 9
    assert batches[0][0].inputs["tokens"][0] == 0  # Caller-owned int64 windows.
    assert TokenBatches(data, 4, 3, split="val")[0][0].inputs["tokens"][0] == 1
    with pytest.raises(IndexError):
        batches[len(batches)]


def test_setup_uses_installed_definitions_and_caps_scheduled_padding(tmp_path):
    settings = Settings(
        str(tmp_path),
        model="llama3-tiny",
        sequence_length=4,
        sequences_per_step=3,
        sequences_per_example=1,
        backward_pieces=2,
    )
    resolved = resolve_training(
        load_registry(["shadowspill", "shadowspill-workloads"]), **settings.request()
    )
    assert resolved.config.optimizer.precision["master_dtype"] == "float32"
    layouts = prepare_layouts(resolved)
    assert {key.operation for key in layouts.families} == {
        "step",
        "forward_backward",
        "forward",
    }
    for family in layouts.families.values():
        assert all(
            sum(
                buffer.fields[("inputs", "tokens")][0]
                for buffer in row.candidate.schedule.buffers
            )
            <= 12
            for row in family
        )
    with pytest.raises(ValueError, match="backward_pieces"):
        replace(settings, backward_pieces=4)


@pytest.mark.parametrize("change", ["data", "settings", "next_update"])
def test_restore_rejects_changed_data_or_progress_before_starting_workers(
    tmp_path, change
):
    settings = Settings(
        str(corpus(tmp_path / "data")),
        model="llama3-tiny",
        sequence_length=4,
        sequences_per_step=3,
        sequences_per_example=1,
    )
    source = TokenBatches(settings.data, 4, 3)
    reference = ArtifactRef("fixture", "a" * 64, 1)
    metadata = dict(
        application=APPLICATION,
        settings=asdict(settings),
        data=source.identity,
        next_update=1,
    )
    metadata[change] = 2 if change == "next_update" else {}
    info = SimpleNamespace(metadata=metadata, completed_updates=1, plans=reference)
    service = SimpleNamespace(
        artifact=lambda _: SimpleNamespace(
            manifest=lambda: SimpleNamespace(result=lambda: SimpleNamespace(info=info))
        )
    )
    record = {
        "application": APPLICATION,
        "settings": asdict(settings),
        "checkpoint": reference.to_dict(),
        "plans": reference.to_dict(),
    }
    path = tmp_path / "resume.json"
    path.write_text(json.dumps(record))
    assert read_resume(path) == (record, settings)
    # This service has no create method: rejecting after startup would fail.
    with pytest.raises(ValueError, match="settings, corpus or progress"):
        start(service, settings, source, tmp_path, resume=record)


def test_application_imports_and_cli_need_no_engine_or_framework():
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('shadowspill', 'shadowspill_workloads', 'torch', 'triton', 'mlops'):
    sys.modules[name] = None
from examples.train_llama3 import arguments
args = arguments(['--run-dir', '/tmp/unused', '--data', '/tmp/unused',
                  '--ssh', 'chicago'])
assert args.ssh == 'chicago'
""",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_resume_cli_does_not_apply_fresh_defaults_or_ignore_explicit_overrides(
    tmp_path,
):
    args = arguments(["--run-dir", str(tmp_path), "--resume", "checkpoint.json"])
    assert "data" not in vars(args) and "steps" not in vars(args)
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "application": APPLICATION,
                "settings": asdict(Settings(str(tmp_path), steps=3)),
            }
        )
    )
    with pytest.raises(ValueError, match="saved training settings"):
        main(
            ["--run-dir", str(tmp_path / "new"), "--resume", str(path), "--steps", "4"]
        )
