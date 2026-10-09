"""The CLI selects model trainability and storage without changing generic planning."""

import json
import sys
from dataclasses import replace

import pytest
import torch

from benchmarking.quickstart import cli
from benchmarking.quickstart.options import _parser, _request_record, parse_arguments
from shadowspill.ssd import SSDPool
from workloads.recipes.text import quickstart as text


def test_ssd_selection_reaches_the_request_and_saved_configuration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "resolve_device", lambda _: torch.device("cuda:0"))
    parser = _parser()
    args = parser.parse_args(
        [
            "--factory",
            "tests.shadowspill.pytorch.api.test_06_generic_quickstart:experiment",
            "--ssd-spill",
            str(tmp_path),
            "--spill-gib",
            "24",
            "--ssd-staging-mib",
            "96",
            "--ssd-chunk-mib",
            "4",
            "--ssd-queue-depth",
            "8",
        ]
    )
    request, _ = cli.resolve_request(parser, args)
    assert isinstance(request.spill, SSDPool)
    assert request.spill.capacity == 24 << 30
    assert request.spill.staging_bytes == 96 << 20
    assert request.spill.chunk_bytes == 4 << 20
    assert request.spill.queue_depth == 8
    saved = _request_record(args, tmp_path, None, tmp_path)["request"]
    assert saved["ssd_spill"] == str(tmp_path)
    assert saved["ssd_staging_mib"] == 96
    json.dumps(saved)


def test_remote_and_ssd_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["--remote-spill", "host:1234", "--ssd-spill", str(tmp_path)]
        )


def test_lora_is_configured_before_initialization_with_frozen_base(monkeypatch):
    original = text.throughput_spec

    def small(family, implementation):
        spec = original(family, implementation)
        config = replace(
            spec.model_config,
            n_layers=1,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            vocab_size=256,
        )
        return replace(spec, model_config=config)

    monkeypatch.setattr(text, "throughput_spec", small)
    parser = _parser()
    args = parser.parse_args(
        [
            "mlops_llama3",
            "--lora",
            "--lora-rank",
            "4",
            "--lora-alpha",
            "8",
            "--lora-dtype",
            "bfloat16",
            "--lora-head",
            "lora",
            "--sequence-length",
            "16",
            "--sequences-per-step",
            "2",
            "--sequences-per-microbatch",
            "1",
        ]
    )
    setup, _, _ = text.recipe(parser, args)
    experiment = setup(device=torch.device("cuda:0"))
    with torch.device("meta"):
        model = experiment["model_factory"]()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("lora" in n.lower() for n in names)
    assert any(n.startswith("lm_head.") for n in names)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters() if p.requires_grad)
    assert not model.get_parameter("lm_head.weight").requires_grad
    assert experiment["metadata"]["trainable"]["rank"] == 4
    assert experiment["metadata"]["trainable"]["mode"] == "lora"


def test_factory_does_not_silently_ignore_lora(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["quickstart", "--factory", "x:y", "--lora"])
    with pytest.raises(SystemExit):
        parse_arguments()
