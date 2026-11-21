"""Global regression inputs and portable DP setup without a client GPU stack."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.regression_training import main
from examples.training.application.setup import start
from examples.training.regression.data import SyntheticBatches
from examples.training.regression.settings import APPLICATION, Settings, parallelism
from shadowspill.parallelism import Parallelism
from shadowspill.training.components import load_registry
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.setup import TrainingSetup


def test_global_inputs_are_repeatable_distinct_and_mean_normalized():
    source = SyntheticBatches(ranks=2)
    batch = source[0]
    assert len(batch) == 16
    assert not np.array_equal(batch[0].inputs["value"], batch[8].inputs["value"])
    assert not np.array_equal(batch[0].inputs["value"], source[1][0].inputs["value"])
    for first, restored in zip(batch, SyntheticBatches(ranks=2)[0], strict=True):
        np.testing.assert_array_equal(first.inputs["value"], restored.inputs["value"])
        np.testing.assert_array_equal(
            first.loss_inputs["targets"], restored.loss_inputs["targets"]
        )
    assert sum(row.loss_inputs["weights"] * 8 for row in batch) == 1
    batch[0].inputs["value"].fill(99)
    assert not np.array_equal(batch[0].inputs["value"], source[0][0].inputs["value"])


@pytest.mark.parametrize("ranks", [1, 2])
def test_settings_use_generic_dp_and_cover_the_global_scheduled_cap(ranks):
    settings = Settings(ranks=ranks)
    record = parallelism(ranks)
    generic = Parallelism.from_dict(record)
    assert generic == Parallelism.dp(ranks_per_replica=ranks, inputs=generic.inputs)
    setup = resolve_training(
        load_registry(["shadowspill-workloads"]),
        TrainingSetup(**settings.request()),
    )
    layouts = prepare_layouts(setup)
    # Whole updates (the default) prepare step, plus forward for predictions.
    assert {key.operation for key in layouts.families} == {"step", "forward"}
    assert setup.context.parallelism == generic
    for rows in layouts.families.values():
        for row in rows:
            assert (
                sum(buffer.max_datums for buffer in row.candidate.schedule.buffers)
                == 8 * ranks
            )
    with pytest.raises(ValueError, match="backward_pieces"):
        replace(settings, backward_pieces=ranks * 8 + 1)


@pytest.mark.parametrize("change", ["settings", "data", "next_update"])
def test_resume_validates_settings_generator_and_cursor_before_creation(
    tmp_path, change
):
    settings = Settings(ranks=1, steps=3)
    source = SyntheticBatches(ranks=1)
    reference = ArtifactRef("fixture", "a" * 64, 1)
    metadata = dict(
        application=APPLICATION,
        settings=asdict(settings),
        data=source.identity,
        next_update=1,
    )
    metadata[change] = 2 if change == "next_update" else {}
    info = SimpleNamespace(
        to_dict=lambda: {"metadata": metadata}, plans=reference, completed_updates=1
    )
    service = SimpleNamespace(
        artifact=lambda _: SimpleNamespace(
            manifest=lambda: SimpleNamespace(result=lambda: SimpleNamespace(info=info))
        )
    )
    with pytest.raises(ValueError, match="settings, inputs or progress"):
        start(
            service,
            settings,
            tmp_path,
            data=source.identity,
            resume={"checkpoint": reference.to_dict(), "plans": reference.to_dict()},
        )


def test_resume_cli_rejects_an_explicit_change_before_connecting(tmp_path):
    path = tmp_path / "resume.json"
    path.write_text(
        json.dumps(
            {"application": APPLICATION, "settings": asdict(Settings(ranks=1, steps=3))}
        )
    )
    with pytest.raises(ValueError, match="saved training settings"):
        main(
            ["--run-dir", str(tmp_path / "new"), "--resume", str(path), "--ranks", "2"]
        )


def test_regression_cli_and_data_import_without_engine_or_frameworks():
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill', 'shadowspill_workloads'):
    sys.modules[name] = None
from examples.regression_training import arguments
from examples.training.regression.data import SyntheticBatches
assert arguments(['--run-dir', '/unused', '--ranks', '2']).ranks == 2
assert len(SyntheticBatches(ranks=2)[1]) == 16
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr
