from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarking.program_collection.config import (
    CollectionConfig,
    GeometryAxes,
    ModelSpec,
    PlanningSpec,
    RuntimeSpec,
    load_collection_config,
)
from benchmarking.program_collection.controller import (
    ControllerOptions,
    run_collection,
)
from benchmarking.program_collection.corpus import (
    ProgramCaseIdentity,
    save_step_program,
)
from benchmarking.program_collection.matrix import (
    ProgramRequest,
    expand_program_requests,
)
from benchmarking.program_collection.state import (
    CollectionPaths,
    begin_attempt,
    completed_artifact,
    finish_attempt,
    load_case_status,
)
from tests.benchmarking._fixtures import _fixture

_REPOSITORY = Path(__file__).resolve().parents[2]
_FULL_CONFIG = (
    _REPOSITORY
    / "benchmarking"
    / "program_collection"
    / "configs"
    / "full_model_program_corpus_v1.json"
)


def _config() -> CollectionConfig:
    return CollectionConfig(
        name="test-collection",
        seed=17,
        expected_programs=2,
        case_timeout_seconds=10,
        max_attempts=2,
        geometry=GeometryAxes((1024,), (1024,), (1, 2)),
        models=(ModelSpec("mlops-llama", "llama3", "mlops"),),
        runtime=RuntimeSpec(4096, 8192, 4096, 8192),
        planning=PlanningSpec("stage_interleaved", 1, 2, True, False, False, None),
    )


def test_full_collection_config_expands_to_168_unique_programs() -> None:
    config = load_collection_config(_FULL_CONFIG)
    requests = expand_program_requests(config)
    assert len(requests) == 168
    assert len({request.case_id for request in requests}) == 168
    assert {
        model.name: sum(request.model == model for request in requests)
        for model in config.models
    } == {
        "mlops-llama3-8b": 56,
        "mlops-qwen35-9b": 56,
        "mlops-olmoe-7b": 56,
    }
    assert tuple(request.model.name for request in requests[:3]) == (
        "mlops-llama3-8b",
        "mlops-qwen35-9b",
        "mlops-olmoe-7b",
    )


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    value = json.loads(_FULL_CONFIG.read_text())
    value["unexpected"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unknown keys: unexpected"):
        load_collection_config(path)


def test_completed_artifact_is_validated_before_resume(tmp_path: Path) -> None:
    config = _config()
    request = expand_program_requests(config)[0]
    paths = CollectionPaths.initialize(tmp_path, config)
    program = _fixture()
    saved = save_step_program(
        tmp_path,
        identity=ProgramCaseIdentity("llama3", "mlops", 1024, 1024, 1),
        program=program,
        metadata={"request_digest": request.digest},
    )
    attempt, _log, _result = begin_attempt(paths, request, command=("fixture",))
    artifact = {
        "directory": str(saved.directory),
        "program_path": str(saved.program_path),
        "program_digest": saved.program_digest,
        "artifact_digest": saved.artifact_digest,
    }
    finish_attempt(
        paths,
        request,
        attempt=attempt,
        status_name="succeeded",
        elapsed_seconds=1.0,
        return_code=0,
        error=None,
        artifact=artifact,
    )
    completed = completed_artifact(paths, request)
    assert completed is not None
    assert completed["directory"] == str(saved.directory.relative_to(tmp_path))
    assert completed["program_path"] == str(saved.program_path.relative_to(tmp_path))
    assert completed["program_digest"] == saved.program_digest


def test_completed_artifact_survives_dataset_relocation(tmp_path: Path) -> None:
    config = _config()
    request = expand_program_requests(config)[0]
    original = tmp_path / "original"
    paths = CollectionPaths.initialize(original, config)
    saved = save_step_program(
        original,
        identity=ProgramCaseIdentity("llama3", "mlops", 1024, 1024, 1),
        program=_fixture(),
        metadata={"request_digest": request.digest},
    )
    attempt, _log, result_path = begin_attempt(paths, request, command=("fixture",))
    result_path.write_text(
        json.dumps(
            {
                "passed": True,
                "artifact": {
                    "directory": str(saved.directory),
                    "program_path": str(saved.program_path),
                    "program_digest": saved.program_digest,
                    "artifact_digest": saved.artifact_digest,
                },
            }
        )
    )
    finish_attempt(
        paths,
        request,
        attempt=attempt,
        status_name="succeeded",
        elapsed_seconds=1.0,
        return_code=0,
        error=None,
        artifact={
            "directory": str(saved.directory),
            "program_path": str(saved.program_path),
            "program_digest": saved.program_digest,
            "artifact_digest": saved.artifact_digest,
        },
    )
    validation_attempt, _log, _result = begin_attempt(
        paths, request, command=("controller-validation",)
    )
    finish_attempt(
        paths,
        request,
        attempt=validation_attempt,
        status_name="failed",
        elapsed_seconds=0.0,
        return_code=None,
        error={"type": "ValueError", "message": "old absolute path is absent"},
        artifact=None,
    )

    relocated = tmp_path / "relocated"
    original.rename(relocated)
    relocated_paths = CollectionPaths.initialize(relocated, config)
    completed = completed_artifact(relocated_paths, request)

    assert completed is not None
    assert completed["directory"].startswith("cases/")
    assert load_case_status(relocated_paths, request)["status"] == "succeeded"


def test_controller_records_each_failure_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    requests = expand_program_requests(config)

    def failing_command(
        _paths: CollectionPaths,
        request: ProgramRequest,
        _result_path: Path,
        _options: ControllerOptions,
    ) -> list[str]:
        return [
            sys.executable,
            "-c",
            f"raise RuntimeError('forced failure for {request.case_id}')",
        ]

    monkeypatch.setattr(
        "benchmarking.program_collection.controller._worker_command",
        failing_command,
    )
    output = tmp_path / "corpus"
    summary = run_collection(
        config,
        requests,
        requests,
        output_root=output,
        options=ControllerOptions(
            planning_cache=tmp_path / "cache",
            resume=False,
            timeout_seconds=10,
            max_attempts=1,
            quiet_plan=False,
            force_fresh=False,
        ),
    )
    assert summary["counts"] == {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 2,
        "interrupted": 0,
    }
    paths = CollectionPaths.initialize(output, config)
    for request in requests:
        status = load_case_status(paths, request)
        assert status["status"] == "failed"
        assert len(status["attempts"]) == 1
        assert status["attempts"][0]["error"]["type"] == "WorkerProcessFailure"
        assert (output / status["attempts"][0]["log_path"]).read_text()


def test_controller_records_controller_exception_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    requests = expand_program_requests(config)
    visited: list[str] = []

    def broken_run(*_args: object, **kwargs: object) -> object:
        prefix = str(kwargs["prefix"])
        visited.append(prefix)
        raise OSError(f"cannot launch {prefix}")

    monkeypatch.setattr(
        "benchmarking.program_collection.controller._run_one",
        broken_run,
    )
    output = tmp_path / "corpus"
    summary = run_collection(
        config,
        requests,
        requests,
        output_root=output,
        options=ControllerOptions(
            planning_cache=tmp_path / "cache",
            resume=False,
            timeout_seconds=10,
            max_attempts=1,
            quiet_plan=False,
            force_fresh=False,
        ),
    )
    assert len(visited) == 2
    assert summary["counts"] == {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 2,
        "interrupted": 0,
    }
    paths = CollectionPaths.initialize(output, config)
    for request in requests:
        status = load_case_status(paths, request)
        assert status["attempts"][0]["error"]["type"] == "OSError"
