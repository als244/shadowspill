"""Logger uses the actual completed update index for scalars and tables."""

from __future__ import annotations

import pytest

from shadowspill.training.logging import Wandb
from shadowspill.training.observations import MetricTable


@pytest.mark.parametrize("first_step", [1, 500])
def test_wandb_uses_completed_updates_for_scalars_tables_and_eval(
    tmp_path, monkeypatch, first_step
):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    with Wandb(
        project="shadowspill-logger-test", run_dir=tmp_path, mode="offline"
    ) as logger:
        for step in range(first_step, first_step + 3):
            logger({"step": step, "train/loss": 1 / step, "train/step_seconds": 0.3})
            logger({"step": step, "packing/trained_tokens": 1024})
            logger.table(
                step,
                "parameters/norms",
                MetricTable(("parameter", "norm"), (("weight", 2.0),)),
            )
            if step == first_step + 1:
                logger({"step": step, "eval/loss": 0.5})
            assert logger.run.step == step


@pytest.mark.parametrize("as_object", [False, True])
def test_rank_gpu_filter_preserves_settings_without_mutating_caller(as_object):
    wandb = pytest.importorskip("wandb")
    from shadowspill.training._wandb_devices import monitored_options

    values = {"x_stats_sampling_interval": 7, "x_stats_gpu_device_ids": [0, 1]}
    supplied = wandb.Settings(**values) if as_object else values
    result = monitored_options({"settings": supplied, "group": "example"}, [4])
    assert result["settings"].x_stats_gpu_device_ids == [4]
    assert result["settings"].x_stats_sampling_interval == 7
    assert result["group"] == "example"
    original = (
        supplied.x_stats_gpu_device_ids
        if as_object
        else supplied["x_stats_gpu_device_ids"]
    )
    assert original == [0, 1]


def test_gpu_mapping_uses_selected_uuid_and_scopes_aggregate_to_node(monkeypatch):
    from types import SimpleNamespace

    import torch

    from shadowspill.training import _wandb_devices as devices

    monkeypatch.setattr(
        devices, "resolve_device", lambda *a, **k: torch.device("cuda:1")
    )
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(uuid="selected", name="example GPU"),
    )
    seen = []

    def index(uuid):
        seen.append(uuid)
        return 5

    monkeypatch.setattr(devices, "monitor_index", index)
    monkeypatch.setattr(devices.socket, "gethostname", lambda: "node-a")
    local = dict(devices.device_record("cuda:1"), rank=0)
    remote = dict(local, rank=1, hostname="node-b", uuid="GPU-remote")
    assert local["device"] == "cuda:1"
    assert local["wandb_gpu_index"] == 5
    assert seen == ["GPU-selected"]
    mapped = devices.aggregate_devices([local, remote])
    assert mapped[0]["aggregate_metric_prefix"] == "gpu.5."
    assert mapped[1]["aggregate_metric_prefix"] is None
    assert mapped[1]["metric_prefix"] == "gpu.5."
    assert seen == ["GPU-selected", "GPU-selected"]
