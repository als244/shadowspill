"""A run records its request whole, and --reproduce repeats it from the record."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import pytest
import torch

from benchmarking.quickstart import (
    Precision,
    _dtype_name,
    _reproduced_arguments,
    _request_record,
)
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch import StepSearchReport


class _Parser:
    def error(self, message: str) -> None:
        raise SystemExit(message)


def _arguments(**overrides: object) -> argparse.Namespace:
    named: dict[str, object] = dict(
        model="mlops_llama3",
        sequence_length=1024,
        sequences_per_step=64,
        search_budget_gib=[6.0, 8.0],
        run_budget_gib=[8.0],
        resolution_options=("0", "1/2", "1"),
        orderings="factors",
        transfer_bandwidths=None,
        steps=5,
        seed=0,
        artifact_store=None,
        build_store=None,
        plan_store=None,
        build_store_mode="contribute",
        plan_store_mode="contribute",
        deterministic=True,
        incumbents=True,
        master_dtype="float32",
        grad_dtype="float32",
        opt_state_dtype="parameter",
        parameter_rounding=None,
        opt_state_rounding="stochastic",
        round_accumulation_once=False,
        export_bypass_key="rev-1",
        reproduce=None,
        output_dir=Path("somewhere"),
        force_overwrite=True,
        plots=True,
    )
    named.update(overrides)
    return argparse.Namespace(**named)


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "seq1024" / "seqsperstep64"
    run.mkdir(parents=True)
    record = _request_record(_arguments(), tmp_path / "store", None, tmp_path / "plans")
    (run / "request.json").write_text(json.dumps(record))
    bandwidths = TransferBandwidths(
        fetch_bytes_per_second=25_500_000_000,
        evict_bytes_per_second=26_000_000_000,
    )
    StepSearchReport(
        total_sequences_per_step=64,
        sequence_length=1024,
        budgets=((8 << 30, 1 << 30),),
        geometries=(),
        points=(),
        skipped=(),
        transfer_bandwidths=bandwidths,
    ).save(run / "search.json")
    return run


def test_the_record_holds_the_request_and_not_where_it_was_written(
    tmp_path: Path,
) -> None:
    record = _request_record(_arguments(), tmp_path / "store", None, tmp_path / "plans")
    request = record["request"]
    assert isinstance(request, dict)
    assert request["model"] == "mlops_llama3"
    assert request["export_bypass_key"] == "rev-1"
    assert request["resolution_options"] == ["0", "1/2", "1"]
    assert request["artifact_store"] == str(tmp_path / "store")
    assert request["plan_store"] == str(tmp_path / "plans")
    assert request["master_dtype"] == "float32"
    assert request["grad_dtype"] == "float32"
    assert request["opt_state_dtype"] == "parameter"
    assert request["parameter_rounding"] is None
    assert request["opt_state_rounding"] == "stochastic"
    assert request["round_accumulation_once"] is False
    for name in (
        "output_dir",
        "force_overwrite",
        "plots",
        "reproduce",
        "timelines",
        "resolution_plans",
    ):
        assert name not in request


def test_reproduce_reads_the_request_and_pins_the_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    monkeypatch.setattr(sys, "argv", ["quickstart", "--reproduce", str(run)])
    given = argparse.Namespace(model=None, reproduce=run, output_dir=None, plots=False)
    arguments = _reproduced_arguments(_Parser(), given)  # type: ignore[arg-type]
    assert arguments.model == "mlops_llama3"
    assert arguments.plan_store_mode == "require"
    assert arguments.artifact_store == tmp_path / "store"
    assert arguments.build_store is None
    assert arguments.resolution_options == ("0", "1/2", "1")
    assert arguments.transfer_bandwidths.fetch_bytes_per_second == 25_500_000_000
    assert arguments.export_bypass_key == "rev-1"
    assert arguments.master_dtype == "float32"
    assert arguments.opt_state_rounding == "stochastic"
    assert arguments.round_accumulation_once is False


def test_reproduce_refuses_a_flag_that_would_contradict_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["quickstart", "--reproduce", str(run), "--steps", "3"]
    )
    given = argparse.Namespace(model=None, reproduce=run, output_dir=None, plots=False)
    with pytest.raises(SystemExit, match="--steps"):
        _reproduced_arguments(_Parser(), given)  # type: ignore[arg-type]
    monkeypatch.setattr(
        sys, "argv", ["quickstart", "mlops_qwen35", "--reproduce", str(run)]
    )
    given = argparse.Namespace(
        model="mlops_qwen35", reproduce=run, output_dir=None, plots=False
    )
    with pytest.raises(SystemExit, match="mlops_qwen35"):
        _reproduced_arguments(_Parser(), given)  # type: ignore[arg-type]


def _adamw(parameters: object, **arguments: object) -> dict[str, object]:
    return {"parameters": parameters, **arguments}


def test_precision_reaches_planning_and_the_optimizer_as_the_harness_names_it() -> None:
    """The flags carry the training config's names into the same two places:
    plan_step's keywords, and the optimizer's own arguments."""

    nothing = Precision.from_arguments(
        _arguments(
            master_dtype=None,
            grad_dtype=None,
            opt_state_dtype=None,
            parameter_rounding=None,
            opt_state_rounding=None,
            round_accumulation_once=False,
        )
    )
    assert nothing.plan_arguments() == {
        "master_dtype": None,
        "grad_dtype": None,
        "round_accumulation_once": False,
    }
    # Naming nothing hands planning the optimizer as it was, so a request
    # without the flags plans exactly as before.
    assert nothing.optimizer(_adamw) is _adamw

    precise = Precision.from_arguments(_arguments(round_accumulation_once=True))
    assert precise.plan_arguments() == {
        "master_dtype": torch.float32,
        "grad_dtype": torch.float32,
        "round_accumulation_once": True,
    }
    built = precise.optimizer(_adamw)
    assert isinstance(built, functools.partial)
    assert built(["p"]) == {
        "parameters": ["p"],
        # A gradient dtype is also what the optimizer reads gradients at,
        # else it would round them to its default on the way in.
        "gradient_dtype": torch.float32,
        "opt_state_dtype": "parameter",
        "opt_state_rounding": "stochastic",
    }
    assert (
        Precision.from_arguments(
            _arguments(opt_state_dtype="float32")
        ).optimizer_arguments()["opt_state_dtype"]
        is torch.float32
    )
    labels = [label for label, _value, _meaning in precise.lines()]
    assert labels == [
        "master dtype",
        "gradient dtype",
        "optimizer state",
        "weight rounding",
        "state rounding",
    ]


def test_a_dtype_flag_takes_a_torch_name_or_none() -> None:
    assert _dtype_name("float32") == "float32"
    assert _dtype_name("torch.bfloat16") == "bfloat16"
    assert _dtype_name("none") is None
    with pytest.raises(argparse.ArgumentTypeError, match="float64"):
        _dtype_name("float64")
