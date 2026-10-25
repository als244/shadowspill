"""The CLI selects model trainability and storage without changing generic planning."""

import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarking.quickstart import cli, runner
from benchmarking.quickstart.options import (
    _parser,
    _request_record,
    graph_pair_variants,
    parse_arguments,
    search_policy,
)
from benchmarking.quickstart.runner import write_cap_verdict
from shadowspill.planner import GenericPlanningOptions, StepDataOrdering
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
            "tests.benchmarking._generic_quickstart:experiment",
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
    original = text.spec

    def small(model, implementation, **changes):
        spec = original(model, implementation, **changes)
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

    monkeypatch.setattr(text, "spec", small)
    parser = _parser()
    args = parser.parse_args(
        [
            "--model",
            "llama3-8b",
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


def factory_taking_lora(
    *, device, lora_rank=32, lora_alpha=32.0, lora_dtype="float32", other=0
):
    return {
        "device": device,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dtype": lora_dtype,
        "other": other,
    }


def factory_without_lora(*, device):
    return {"device": device}


def test_lora_flags_reach_a_factory_that_takes_them(monkeypatch):
    monkeypatch.setattr(cli, "resolve_device", lambda _: torch.device("cuda:0"))
    parser = _parser()
    args = parser.parse_args(
        [
            "--factory",
            f"{__name__}:factory_taking_lora",
            "--factory-args",
            '{"other": 7}',
            "--lora-rank",
            "8",
            "--lora-alpha",
            "8",
            "--lora-dtype",
            "bfloat16",
        ]
    )
    _, setup = cli.resolve_request(parser, args)
    built = setup(device="cpu")
    assert built["lora_rank"] == 8 and built["lora_alpha"] == 8.0
    assert built["lora_dtype"] == "bfloat16" and built["other"] == 7
    saved = _request_record(args, Path("store"), None, Path("plans"))["request"]
    assert saved["lora_rank"] == 8 and saved["factory_args"] == {"other": 7}


def test_lora_flags_are_refused_by_a_factory_that_does_not_take_them(monkeypatch):
    monkeypatch.setattr(cli, "resolve_device", lambda _: torch.device("cuda:0"))
    parser = _parser()
    args = parser.parse_args(
        ["--factory", f"{__name__}:factory_without_lora", "--lora-rank", "8"]
    )
    with pytest.raises(SystemExit):
        cli.resolve_request(parser, args)
    args = parser.parse_args(
        [
            "--factory",
            f"{__name__}:factory_taking_lora",
            "--lora-rank",
            "8",
            "--factory-args",
            '{"lora_rank": 4}',
        ]
    )
    with pytest.raises(SystemExit):
        cli.resolve_request(parser, args)


def test_preset_lora_settings_require_lora(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["quickstart", "--model", "llama3-8b", "--lora-rank", "8"]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_preset_lora_defaults_are_configure_loras_own():
    parser = _parser()
    args = parser.parse_args(
        [
            "--model",
            "llama3-8b",
            "--lora",
            "--sequence-length",
            "16",
            "--sequences-per-step",
            "2",
            "--sequences-per-microbatch",
            "1",
        ]
    )
    setup, _, _ = text.recipe(parser, args)
    trainable = setup(device=torch.device("cuda:0"))["metadata"]["trainable"]
    assert (trainable["rank"], trainable["alpha"], trainable["factor_dtype"]) == (
        32,
        32.0,
        "bfloat16",
    )


def test_small_object_threshold_reaches_the_search_and_the_record(monkeypatch):
    parser = _parser()
    policy = search_policy(parser.parse_args(["--factory", "x:y"]))
    assert (
        policy.generic.minimum_object_bytes_evict_eligible
        == GenericPlanningOptions().minimum_object_bytes_evict_eligible
    )
    args = parser.parse_args(["--factory", "x:y", "--minimum-evict-kib", "128"])
    assert search_policy(args).generic.minimum_object_bytes_evict_eligible == 128 << 10
    saved = _request_record(args, Path("store"), None, Path("plans"))["request"]
    assert saved["minimum_evict_kib"] == 128 and saved["max_evict_gib"] is None
    monkeypatch.setattr(
        sys, "argv", ["quickstart", "--factory", "x:y", "--minimum-evict-kib", "-1"]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


@pytest.mark.parametrize("resolution", ["0", "1", "0,1", "quarters"])
def test_pinned_resolution_keeps_both_variants_for_accounting(resolution):
    parser = _parser()
    args = parser.parse_args(["--factory", "x:y", "--resolution-options", resolution])
    assert graph_pair_variants(args) == ("save", "recompute")
    # Profiling both variants does not broaden the requested search.
    policy = search_policy(args)
    assert policy.algorithm.options.resolution_options == tuple(
        Fraction(value) for value in args.resolution_options
    )


def test_write_cap_declines_only_a_plan_above_it(monkeypatch):
    assert write_cap_verdict(None, 1 << 50) is None
    assert write_cap_verdict(1.0, 1 << 30) is None
    verdict = write_cap_verdict(1.0, (1 << 30) + 1)
    assert verdict is not None and verdict.endswith("above the 1.00 GiB cap")
    monkeypatch.setattr(
        sys, "argv", ["quickstart", "--factory", "x:y", "--max-evict-gib", "0"]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_write_cap_skips_a_run_budget_before_it_is_planned(tmp_path):
    arguments = _parser().parse_args(["--factory", "x:y", "--max-evict-gib", "1"])
    tour = object.__new__(runner.Tour)
    tour.arguments = arguments
    tour.request = runner.Request(
        label="t",
        search_budgets=[1 << 30],
        run_budgets=[1 << 30],
        physical_capacity=1 << 30,
        spill_budget=1 << 30,
        device=torch.device("cpu"),
    )
    tour.budgets = runner.Budgets([1 << 30], [1 << 30], {1 << 30: 1 << 30})
    tour.candidates = {"1": ((torch.zeros(1),),)}
    tour.paths = runner.RunPaths(tmp_path, tmp_path, None, tmp_path)
    tour.units_per_step, tour.unit_label, tour.distributed = 1.0, "updates", None
    notes = []
    tour.plan_log = SimpleNamespace(note=notes.append, write=lambda text: len(text))
    point = SimpleNamespace(
        candidate="1",
        ordering=StepDataOrdering.depth_first(1),
        summary=SimpleNamespace(transfer_bytes_evicted=2 << 30),
    )
    tour.report = SimpleNamespace(winner=lambda budget, spill: point, winner_plans={})
    tour.run_one_budget = lambda *args, **kwargs: pytest.fail("the budget was planned")
    assert tour.run() == []
    assert notes and "above the 1.00 GiB cap" in notes[0]


def test_an_image_needs_the_ssd_pool_it_sits_behind(tmp_path, monkeypatch):
    (tmp_path / "frozen.image").write_bytes(b"\0" * 4096)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quickstart",
            "--factory",
            "x:y",
            "--ssd-image",
            str(tmp_path / "frozen.image"),
            "--spill-gib",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        parse_arguments()
