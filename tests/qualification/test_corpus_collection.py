from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from qualification.planner.corpus import ProgramCaseIdentity, save_step_program
from qualification.planner.corpus_collection.config import (
    CollectionConfig,
    GeometryAxes,
    ModelSpec,
    PlanningSpec,
    RuntimeSpec,
    load_collection_config,
)
from qualification.planner.corpus_collection.controller import (
    ControllerOptions,
    run_collection,
)
from qualification.planner.corpus_collection.matrix import (
    ProgramRequest,
    expand_program_requests,
)
from qualification.planner.corpus_collection.state import (
    CollectionPaths,
    begin_attempt,
    completed_artifact,
    finish_attempt,
    load_case_status,
)
from qualification.planner.smoke_program_artifacts import _fixture

_REPOSITORY = Path(__file__).resolve().parents[2]
_FULL_CONFIG = (
    _REPOSITORY
    / "qualification"
    / "planner"
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
    assert completed_artifact(paths, request) == artifact


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
        "qualification.planner.corpus_collection.controller._worker_command",
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
        assert Path(status["attempts"][0]["log_path"]).read_text()


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
        "qualification.planner.corpus_collection.controller._run_one",
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
