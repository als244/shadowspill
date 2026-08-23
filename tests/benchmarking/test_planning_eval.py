from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarking.planning_eval.cli import _find_resume_baseline
from benchmarking.planning_eval.config import (
    BandwidthScale,
    FrontierConfig,
    FrontierGrid,
    TransferBandwidthBaseline,
    load_frontier_config,
)
from benchmarking.planning_eval.controller import _recover_interrupted_points
from benchmarking.planning_eval.matrix import (
    FrontierPointRequest,
    expand_frontier_points,
    expand_grid_axes,
)
from benchmarking.planning_eval.process import execute_case_worker
from benchmarking.planning_eval.provenance import (
    RepositoryProvenance,
)
from benchmarking.planning_eval.source import (
    CorpusProgramCase,
    corpus_manifest_digest,
    discover_program_cases,
)
from benchmarking.planning_eval.storage import (
    BaselinePaths,
    atomic_json,
    begin_point_attempt,
    finish_point_attempt,
    initialize_point,
    point_attempt_count,
    point_complete,
    recover_running_attempt,
)
from benchmarking.planning_eval.summary import write_frontier_summary
from benchmarking.program_collection.corpus import (
    ProgramCaseIdentity,
    save_step_program,
)
from tests.benchmarking._fixtures import _fixture

_REPOSITORY = Path(__file__).resolve().parents[2]
#: An older revision of this repository, used to prove that resume does not
#: care which one a baseline started on.
_REPOSITORY_HEAD = "487de0b355365d7ce84911630c96216e5bc9794b"
_CONFIG = (
    _REPOSITORY
    / "benchmarking"
    / "planning_eval"
    / "configs"
    / "full_pressurefit_frontier_v1.json"
)


def test_full_frontier_has_2520_points_and_three_global_bandwidths() -> None:
    config = load_frontier_config(_CONFIG)
    assert config.point_timeout_seconds == 300
    assert config.max_point_attempts == 1
    assert len(expand_grid_axes(config.grids)) == 15
    assert config.expected_programs * config.expected_points_per_program == 2520
    program = _fixture().recurrent
    points = expand_frontier_points(
        program,
        config.grids,
        transfer_baseline=config.transfer_bandwidths,
    )
    assert len(points) == 15
    assert {
        (
            item.transfer_bandwidths.fetch_bytes_per_second,
            item.transfer_bandwidths.evict_bytes_per_second,
        )
        for item in points
    } == {
        (12_721_870_088, 12_611_359_693),
        (25_443_740_177, 25_222_719_387),
        (50_887_480_354, 50_445_438_774),
    }
    assert FrontierPointRequest.from_value(points[0].to_dict()) == points[0]


def test_config_rejects_duplicate_points(tmp_path: Path) -> None:
    value = json.loads(_CONFIG.read_text())
    value["grids"].append(value["grids"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(value))
    try:
        load_frontier_config(path)
    except ValueError as error:
        assert "duplicate point" in str(error)
    else:
        raise AssertionError("duplicate frontier point was accepted")


def test_corpus_discovery_and_point_crash_recovery(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    saved = save_step_program(
        corpus_root,
        identity=ProgramCaseIdentity("llama3", "mlops", 1024, 1024, 1),
        program=_fixture(),
    )
    cases = discover_program_cases(corpus_root, expected_count=1)
    assert len(cases) == 1
    assert corpus_manifest_digest(cases) == corpus_manifest_digest(cases)
    relocated_cases = (
        CorpusProgramCase(
            tmp_path / "relocated" / cases[0].directory.name,
            cases[0].identity,
            cases[0].program_digest,
            cases[0].artifact_digest,
        ),
    )
    assert corpus_manifest_digest(relocated_cases) == corpus_manifest_digest(cases)
    config = FrontierConfig(
        name="test-frontier",
        expected_programs=1,
        expected_points_per_program=1,
        program_role="recurrent",
        point_timeout_seconds=10,
        max_point_attempts=2,
        max_worker_restarts_per_program=4,
        pressurefit_cache_mode="cold",
        transfer_bandwidths=TransferBandwidthBaseline(100, 80, "test"),
        grids=(
            FrontierGrid(
                "main",
                (96,),
                (1024,),
                (BandwidthScale(1, 1),),
            ),
        ),
    )
    provenance = RepositoryProvenance(
        _REPOSITORY,
        "a" * 40,
        "",
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    paths = BaselinePaths.initialize(
        tmp_path / "frontiers",
        baseline_id="test-baseline",
        config=config,
        provenance=provenance,
        corpus_root=corpus_root,
        corpus_digest=corpus_manifest_digest(cases),
        cases=cases,
    )
    request = expand_frontier_points(
        _fixture().recurrent,
        config.grids,
        transfer_baseline=config.transfer_bandwidths,
    )[0]
    case = CorpusProgramCase(
        saved.directory,
        saved.identity,
        saved.program_digest,
        saved.artifact_digest,
    )
    directory = initialize_point(paths, case, request)
    first = begin_point_attempt(directory, request, revision="0" * 40)
    assert first == 1
    assert not recover_running_attempt(
        directory,
        request,
        case=case,
        max_attempts=2,
        elapsed_seconds=1.0,
        error={"type": "CompiledCrash", "message": "signal 11"},
    )
    second = begin_point_attempt(directory, request, revision="0" * 40)
    finish_point_attempt(
        directory,
        request,
        attempt=second,
        status_name="error",
        elapsed_seconds=1.0,
        evidence={"error": {"type": "CompiledCrash"}},
        error={"type": "CompiledCrash"},
        final=True,
    )
    assert point_complete(directory, request)


def test_resume_preserves_but_does_not_charge_an_interrupted_attempt(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    saved = save_step_program(
        corpus_root,
        identity=ProgramCaseIdentity("llama3", "mlops", 1024, 1024, 1),
        program=_fixture(),
    )
    case = CorpusProgramCase(
        saved.directory,
        saved.identity,
        saved.program_digest,
        saved.artifact_digest,
    )
    config = FrontierConfig(
        name="interrupted-frontier",
        expected_programs=1,
        expected_points_per_program=1,
        program_role="recurrent",
        point_timeout_seconds=300,
        max_point_attempts=1,
        max_worker_restarts_per_program=1,
        pressurefit_cache_mode="cold",
        transfer_bandwidths=TransferBandwidthBaseline(100, 80, "test"),
        grids=(
            FrontierGrid(
                "main",
                (96,),
                (1024,),
                (BandwidthScale(1, 1),),
            ),
        ),
    )
    provenance = RepositoryProvenance(
        _REPOSITORY,
        "a" * 40,
        "",
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    paths = BaselinePaths.initialize(
        tmp_path / "frontiers",
        baseline_id="interrupted-baseline",
        config=config,
        provenance=provenance,
        corpus_root=corpus_root,
        corpus_digest=corpus_manifest_digest((case,)),
        cases=(case,),
    )
    request = expand_frontier_points(
        _fixture().recurrent,
        config.grids,
        transfer_baseline=config.transfer_bandwidths,
    )[0]
    directory = initialize_point(paths, case, request)
    assert begin_point_attempt(directory, request, revision="0" * 40) == 1
    assert point_attempt_count(directory, request) == 1

    atomic_json(
        paths.case_directory(case) / "case.json",
        {"points": [request.to_dict()]},
    )
    assert _recover_interrupted_points(paths, (case,)) == 1
    assert point_attempt_count(directory, request) == 0
    assert begin_point_attempt(directory, request, revision="0" * 40) == 2


def test_timeout_recovery_writes_summarizable_canonical_evidence(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    saved = save_step_program(
        corpus_root,
        identity=ProgramCaseIdentity("qwen35", "mlops", 4096, 1024, 8),
        program=_fixture(),
    )
    case = CorpusProgramCase(
        saved.directory,
        saved.identity,
        saved.program_digest,
        saved.artifact_digest,
    )
    config = FrontierConfig(
        name="timeout-frontier",
        expected_programs=1,
        expected_points_per_program=1,
        program_role="recurrent",
        point_timeout_seconds=300,
        max_point_attempts=1,
        max_worker_restarts_per_program=2,
        pressurefit_cache_mode="cold",
        transfer_bandwidths=TransferBandwidthBaseline(100, 80, "test"),
        grids=(
            FrontierGrid(
                "main",
                (16 << 30,),
                (112 << 30,),
                (BandwidthScale(1, 2),),
            ),
        ),
    )
    provenance = RepositoryProvenance(
        _REPOSITORY,
        "a" * 40,
        "",
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    paths = BaselinePaths.initialize(
        tmp_path / "frontiers",
        baseline_id="timeout-baseline",
        config=config,
        provenance=provenance,
        corpus_root=corpus_root,
        corpus_digest=corpus_manifest_digest((case,)),
        cases=(case,),
    )
    atomic_json(
        paths.case_directory(case) / "case.json",
        {"case": case.to_dict()},
    )
    request = expand_frontier_points(
        _fixture().recurrent,
        config.grids,
        transfer_baseline=config.transfer_bandwidths,
    )[0]
    directory = initialize_point(paths, case, request)
    begin_point_attempt(directory, request, revision="0" * 40)
    assert recover_running_attempt(
        directory,
        request,
        case=case,
        max_attempts=1,
        elapsed_seconds=300.25,
        error={"type": "FrontierPointTimeout", "message": "timed out"},
    )

    point = json.loads((directory / "point.json").read_text())
    assert point["case"]["case_id"] == case.case_id
    assert point["request"] == request.to_dict()
    assert point["timing"]["attempt_elapsed_seconds"] == 300.25
    summary = write_frontier_summary(
        paths,
        expected_programs=1,
        expected_points_per_program=1,
    )
    assert summary["status_counts"] == {"error": 1}
    assert summary["observed_transfer_bandwidth_combinations"] == [
        {"fetch_bytes_per_second": 50, "evict_bytes_per_second": 40}
    ]

    # Incomplete historical point shapes are rejected rather than rebuilt from
    # sidecars. Only the current complete point record is a source of truth.
    (directory / "point.json").write_text(
        json.dumps(
            {
                "schema": "shadowspill.pressurefit_frontier_point/v1",
                "request_digest": request.digest,
                "point_id": request.point_id,
                "status": "error",
                "error": {
                    "type": "FrontierPointTimeout",
                    "message": "timed out",
                },
            }
        )
    )
    with pytest.raises(ValueError, match="has no embedded request"):
        write_frontier_summary(
            paths,
            expected_programs=1,
            expected_points_per_program=1,
        )


def test_worker_output_is_streamed_to_worker_main_log_and_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    worker_log = tmp_path / "worker.log"
    collection_log = tmp_path / "collection.log"
    outcome = execute_case_worker(
        [
            sys.executable,
            "-c",
            (
                "print('POINT START point=one utc=2026-08-16T00:00:00+00:00', "
                "flush=True); print(flush=True); "
                "print('POINT SUCCESS point=one utc=2026-08-16T00:00:01+00:00', "
                "flush=True)"
            ),
        ],
        repository_root=tmp_path,
        case_run_directory=tmp_path / "case",
        log_path=worker_log,
        collection_log_path=collection_log,
        point_timeout_seconds=10,
        console_prefix="[1/1] example",
    )
    assert outcome.return_code == 0
    assert "POINT START point=one" in worker_log.read_text()
    assert "POINT SUCCESS point=one" in worker_log.read_text()
    main_log = collection_log.read_text()
    assert "[1/1] example | POINT START point=one" in main_log
    assert "[1/1] example | POINT SUCCESS point=one" in main_log
    assert main_log.splitlines()[1] == ""
    captured = capsys.readouterr()
    assert "[1/1] example | POINT START point=one" in captured.out
    assert "[1/1] example | POINT SUCCESS point=one" in captured.out
    assert "\n\n[1/1] example | POINT SUCCESS point=one" in captured.out


def _clean_provenance_at_head() -> RepositoryProvenance:
    """This repository at HEAD, as if nothing were modified."""

    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    empty = hashlib.sha256(b"").hexdigest()
    return RepositoryProvenance(_REPOSITORY, head, "", "", empty)


def test_resume_continues_across_a_planner_affecting_revision(
    tmp_path: Path,
) -> None:
    """Resume is about finishing a run, not about proving the source matched."""

    output = tmp_path / "results"
    prior = output / "frontier__old__clean__cfg-config"
    prior.mkdir(parents=True)
    atomic_json(
        prior / "manifest.json",
        {
            "config_digest": "config",
            "corpus": {"manifest_digest": "corpus"},
            "repository": {"head": _REPOSITORY_HEAD, "dirty": False},
        },
    )
    atomic_json(prior / "summary.json", {"pending_points": 10})
    provenance = _clean_provenance_at_head()

    path, relationship = _find_resume_baseline(
        output,
        baseline_id="frontier__new__clean__cfg-config",
        config_digest="config",
        corpus_digest="corpus",
        provenance=provenance,
        enabled=True,
    )

    assert path == prior
    assert relationship is not None
    assert relationship["recorded_head"] == _REPOSITORY_HEAD
    assert relationship["resume_head"] == provenance.head
    # Planner sources changed between the two, and resume says so rather than
    # refusing over it.
    assert relationship["spans_revisions"] is True
    assert relationship["classification"] != "exact_source"


def test_resume_still_refuses_a_different_corpus(tmp_path: Path) -> None:
    """What is being measured has to match, even though the source need not."""

    output = tmp_path / "results"
    prior = output / "frontier__old__clean__cfg-config"
    prior.mkdir(parents=True)
    atomic_json(
        prior / "manifest.json",
        {
            "config_digest": "config",
            "corpus": {"manifest_digest": "a-different-corpus"},
            "repository": {"head": _REPOSITORY_HEAD, "dirty": False},
        },
    )
    atomic_json(prior / "summary.json", {"pending_points": 10})

    path, relationship = _find_resume_baseline(
        output,
        baseline_id="frontier__new__clean__cfg-config",
        config_digest="config",
        corpus_digest="corpus",
        provenance=_clean_provenance_at_head(),
        enabled=True,
    )

    assert path is None
    assert relationship is None


def test_resume_discovers_one_compatible_incomplete_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results"
    prior = output / "frontier__old__clean__cfg-config"
    prior.mkdir(parents=True)
    atomic_json(
        prior / "manifest.json",
        {
            "config_digest": "config",
            "corpus": {"manifest_digest": "corpus"},
            "repository": {"head": "a" * 40, "dirty": False},
        },
    )
    atomic_json(prior / "summary.json", {"pending_points": 10})
    expected = {
        "recorded_head": "a" * 40,
        "resume_head": "b" * 40,
        "changed_files": ["benchmarking/planning_eval/summary.py"],
        "classification": "harness_only",
    }
    monkeypatch.setattr(
        "benchmarking.planning_eval.cli.resume_provenance_relationship",
        lambda _current, _recorded: expected,
    )
    provenance = RepositoryProvenance(
        _REPOSITORY,
        "b" * 40,
        "",
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )

    path, compatibility = _find_resume_baseline(
        output,
        baseline_id="frontier__new__clean__cfg-config",
        config_digest="config",
        corpus_digest="corpus",
        provenance=provenance,
        enabled=True,
    )

    assert path == prior
    assert compatibility == expected
