"""Opt-in actual Sonic kernels through saved phases and independent training."""

import pytest

from workload_tests.providers.prepared_sonic.run import prerequisites, run_case


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize(
    "kind,variant",
    [
        ("forward_backward", "recompute"),
        ("step", "save"),
    ],
)
def test_prepared_sonic_saved_phases_and_resume(tmp_path, kind, variant):
    environment = prerequisites()
    if environment["reason"] is not None:
        pytest.skip(environment["reason"])
    run_case(tmp_path / "case", dict(batch=2, variant=variant, kind=kind))
