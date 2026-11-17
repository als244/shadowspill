"""Candidate capacity failures remain diagnostic; contract failures stop preparation."""

import pytest


def test_candidate_exhaustion_is_recorded_but_contract_errors_stop(tmp_path):
    from types import SimpleNamespace

    import torch

    from shadowspill.training.preparation.candidates import (
        attempt_candidate,
        available,
        failures,
        program_paths,
    )

    options = {
        "candidates": [{"path": "large"}, {"path": "small"}],
        "eval_every": 1,
    }

    def exhausted():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    attempt_candidate(
        tmp_path,
        options["candidates"][0],
        "profile",
        exhausted,
        SimpleNamespace(count=1),
    )
    assert failures(tmp_path)["large"]["phase"] == "profile"
    assert available(tmp_path, options) == ({"path": "small"},)
    assert program_paths(options, root=tmp_path) == (
        "small/training.ssp",
        "small/evaluation.ssp",
    )
    with pytest.raises(torch.cuda.OutOfMemoryError):
        attempt_candidate(
            tmp_path,
            {"path": "collective"},
            "build",
            exhausted,
            SimpleNamespace(count=2),
        )

    def invalid():
        raise ValueError("invalid execution-axis contract")

    with pytest.raises(ValueError, match="execution-axis"):
        attempt_candidate(
            tmp_path, {"path": "invalid"}, "build", invalid, SimpleNamespace(count=1)
        )
    assert set(failures(tmp_path)) == {"large"}
