"""Candidate logs describe resolved geometry and preserve recoverable outcomes."""

from types import SimpleNamespace

import pytest
import torch

from engine_tests.training.test_preparation_templates import inventory
from shadowspill.diagnostics.progress import Progress
from shadowspill.training.preparation.failures import PreparationFailure
from shadowspill.training.preparation.job_artifacts import attempt
from shadowspill.training.preparation.progress import (
    candidate_title,
    describe_candidate,
)


def test_candidate_descriptions_distinguish_training_and_explicit_forward():
    prepared = inventory()
    for index, template in enumerate(prepared.templates.values(), 1):
        lines = []
        with Progress(lines.append).activate():
            describe_candidate(prepared, template)
        text = "\n".join(lines)
        title = candidate_title(template, index, len(prepared.templates))
        assert f"Candidate {index}/6" in title
        assert ("Forward" in title) == (template.kind == "forward")
        assert "Microbatch size: " in text
        assert "Sequence/item slots: " in text and "inputs.pixels" in text
        assert "Execution chunking: Disabled" in text
        assert "Retention: Cost-input (default)" in text
        assert ("paired by default" in text) == (template.kind != "forward")


def test_attempt_returns_rejection_and_keeps_the_failure_record(tmp_path):
    key = "a" * 64
    workers = SimpleNamespace(count=1)

    def exhausted():
        raise torch.cuda.OutOfMemoryError("test head workspace")

    assert not attempt(tmp_path, key, "build", exhausted, workers)
    path = tmp_path / "failures" / f"{key}.ssp"
    failure = PreparationFailure.load(path)
    assert "test head workspace" in failure.reason
    assert attempt(tmp_path, key, "build", lambda: None, workers)
    assert not path.exists()

    def invalid():
        raise ValueError("bad capture")

    with pytest.raises(ValueError, match="bad capture"):
        attempt(tmp_path, key, "build", invalid, workers)
