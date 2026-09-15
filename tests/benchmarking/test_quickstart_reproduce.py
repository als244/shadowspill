"""A run records its request whole, and --reproduce repeats it from the record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from benchmarking.quickstart import _reproduced_arguments, _request_record
from shadowspill.planner.program_inputs import TransferBandwidths


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
    (run / "search.json").write_text(
        json.dumps({"transfer_bandwidths": bandwidths.to_dict(), "geometries": []})
    )
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
    for name in ("output_dir", "force_overwrite", "plots", "reproduce"):
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
