"""A missing template removes all of its uses while keeping full call coverage."""

import pytest

from engine_tests.training.test_preparation_templates import inventory
from shadowspill.errors import PlanInfeasibleError
from shadowspill.training.preparation.failures import (
    PreparationFailure,
    prune_failed_templates,
)


def test_masked_failure_prunes_every_dependent_operation_without_lowering_capacity(
    tmp_path,
):
    prepared = inventory()
    key = next(
        key
        for key, t in prepared.templates.items()
        if t.output_only and t.binding.inputs.layout.buffer.max_datums == 4
    )
    failure = PreparationFailure(key, "profile", "device exhausted")
    path = tmp_path / "failure.ssp"
    failure.save(path)
    assert PreparationFailure.load(path) == failure
    actual, rejected = prune_failed_templates(prepared, {key: failure})
    assert actual.layouts.setup is prepared.layouts.setup
    assert len(actual.layouts.coverage) == len(prepared.layouts.coverage) == 3
    assert {row.operation.operation for row in rejected} == {"step", "forward_backward"}
    assert all(row.stage == "profile" and not row.search_exhausted for row in rejected)
    assert key not in actual.templates
    for (operation, batch), proof in actual.layouts.coverage.items():
        assert proof.config == prepared.layouts.setup.config.batches[batch]
        assert proof.schedule in [
            row.layout.candidate.schedule for row in actual.families[operation]
        ]


def test_lost_universal_witness_keeps_candidates_but_lost_capability_fails():
    prepared = inventory()
    failed = {
        key: PreparationFailure(key, "build", "device exhausted")
        for key, template in prepared.templates.items()
        if template.output_only
        and template.binding.inputs.layout.buffer.max_datums >= 2
    }
    actual, _ = prune_failed_templates(prepared, failed)
    assert actual.layouts.setup is prepared.layouts.setup
    assert len(actual.layouts.coverage) < len(prepared.layouts.coverage)
    assert all(actual.families.values())
    failed.update(
        {
            key: PreparationFailure(key, "build", "device exhausted")
            for key, template in prepared.templates.items()
            if template.output_only
        }
    )
    with pytest.raises(PlanInfeasibleError, match="removed every candidate") as caught:
        prune_failed_templates(prepared, failed)
    assert caught.value.kind == "prepared_capture_coverage"
    assert caught.value.diagnostics
    unknown = "f" * 64
    with pytest.raises(ValueError, match="differ from the capture inventory"):
        prune_failed_templates(
            prepared, {unknown: PreparationFailure(unknown, "build", "failure")}
        )
