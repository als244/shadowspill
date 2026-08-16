from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from qualification.planner.corpus import (
    ProgramCaseIdentity,
    save_step_program,
)
from qualification.planner.frontier_collection.config import (
    BandwidthScale,
    FrontierConfig,
    FrontierGrid,
    TransferBandwidthBaseline,
    load_frontier_config,
)
from qualification.planner.frontier_collection.matrix import (
    FrontierPointRequest,
    expand_frontier_points,
    expand_grid_axes,
)
from qualification.planner.frontier_collection.process import execute_case_worker
from qualification.planner.frontier_collection.provenance import (
    RepositoryProvenance,
)
from qualification.planner.frontier_collection.source import (
    CorpusProgramCase,
    corpus_manifest_digest,
    discover_program_cases,
)
from qualification.planner.frontier_collection.storage import (
    BaselinePaths,
    begin_point_attempt,
    finish_point_attempt,
    initialize_point,
    point_complete,
    recover_running_attempt,
)
from qualification.planner.smoke_program_artifacts import _fixture

_REPOSITORY = Path(__file__).resolve().parents[2]
_CONFIG = (
    _REPOSITORY
    / "qualification"
    / "planner"
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
    first = begin_point_attempt(directory, request)
    assert first == 1
    assert not recover_running_attempt(
        directory,
        request,
        max_attempts=2,
        error={"type": "NativeCrash", "message": "signal 11"},
    )
    second = begin_point_attempt(directory, request)
    finish_point_attempt(
        directory,
        request,
        attempt=second,
        status_name="error",
        elapsed_seconds=1.0,
        evidence={"error": {"type": "NativeCrash"}},
        error={"type": "NativeCrash"},
        final=True,
    )
    assert point_complete(directory, request)


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
                "flush=True); "
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
    captured = capsys.readouterr()
    assert "[1/1] example | POINT START point=one" in captured.out
    assert "[1/1] example | POINT SUCCESS point=one" in captured.out
