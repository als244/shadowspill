from __future__ import annotations

from qualification.performance.cases import manifests
from qualification.performance.matrix import (
    _active_planning_phases,
    _termination_signal,
)


def test_full_model_manifests_preserve_retained_geometries() -> None:
    rows = {
        (item.family, item.implementation): (
            item.tokens_per_microbatch,
            item.accumulation_count,
            item.tokens_per_step,
        )
        for item in manifests()
    }
    assert rows == {
        ("llama3", "mlops"): (8_192, 8, 65_536),
        ("qwen35", "mlops"): (16_384, 4, 65_536),
        ("olmoe", "mlops"): (32_768, 2, 65_536),
        ("llama3", "pytorch"): (8_192, 8, 65_536),
        ("qwen35", "pytorch"): (16_384, 4, 65_536),
    }


def test_only_mlops_cells_have_throughput_authorities() -> None:
    for item in manifests():
        expected = item.implementation == "mlops"
        assert (item.regression_tokens_per_second is not None) == expected
        assert (item.remote_regression_tokens_per_second is not None) == expected
        assert (item.predecessor_tokens_per_second is not None) == expected


def test_remote_floor_is_below_the_local_floor() -> None:
    """A peer's pool is reached over a link many times slower than pinned host
    memory, so a remote floor above the local one would be a mistake in the
    table rather than a measurement."""

    for item in manifests():
        if item.regression_tokens_per_second is None:
            continue
        assert item.remote_regression_tokens_per_second is not None
        assert (
            item.remote_regression_tokens_per_second < item.regression_tokens_per_second
        )


def test_predecessor_parity_is_not_silently_declared_reached() -> None:
    """The parity target is the predecessor's number, not ours.

    Re-basing it onto a current measurement would read as parity reached while
    the gap is open, so this pins the two authorities apart until a deliberate
    change moves them together.
    """

    for item in manifests():
        if item.regression_tokens_per_second is None:
            continue
        assert item.predecessor_tokens_per_second is not None
        assert item.regression_tokens_per_second < item.predecessor_tokens_per_second


def test_full_model_launcher_recovers_killed_planning_phase() -> None:
    log = "\n".join(
        (
            "[shadowspill.plan +   0.001s] capture_lowering: started",
            "[shadowspill.plan +   1.001s]   objective_export: started",
            "[shadowspill.plan +   2.001s]   objective_export: finished in 1.000s",
            "[shadowspill.plan +   2.002s] capture_lowering: finished in 2.001s",
            "[shadowspill.plan +   2.003s] optimizer_capture: started",
        )
    )
    assert _active_planning_phases(log) == ("optimizer_capture",)
    assert _termination_signal(-9) == "SIGKILL"


def test_performance_gate_preserves_the_three_default_mlops_workloads() -> None:
    """Hardware-aware judging does not change the default workload matrix."""

    from qualification.performance.matrix import default_cells

    assert [item.identity for item in default_cells()] == [
        "mlops_llama3",
        "mlops_qwen35",
        "mlops_olmoe",
    ]
    for item in default_cells():
        assert item.regression_tokens_per_second is not None


def test_a_smaller_spill_pool_reaches_the_cell_that_runs_it() -> None:
    from dataclasses import replace
    from pathlib import Path

    from qualification.performance.cases import manifest_for
    from qualification.performance.matrix import _cell_command, _parser

    manifest = manifest_for("llama3", "mlops")
    arguments = _parser().parse_args([])

    def command(item: object) -> list[str]:
        return _cell_command(item, arguments, Path("out"), Path("out/cell.json"), {})  # type: ignore[arg-type]

    assert "--spill-budget-gib" not in command(manifest)
    smaller = command(replace(manifest, spill_budget_bytes=80 << 30))
    assert smaller[smaller.index("--spill-budget-gib") + 1] == "80"


def test_a_mixture_objective_reports_the_heads_share_as_its_metric() -> None:
    import pytest
    import torch

    from workloads.full_model import (
        BALANCING_COEFFICIENT,
        HEAD_LOSS_METRIC,
        _with_balancing,
    )

    head = torch.tensor(20.0, requires_grad=True)
    balancing = torch.tensor(4.0, requires_grad=True)
    result = _with_balancing(head, balancing, 10.0)
    # the objective: both shares, the balancing term weighted
    assert float(result[0]) == pytest.approx(2.0 + BALANCING_COEFFICIENT * 0.4)
    assert result[0].requires_grad
    # the metric: the head's share alone, not differentiated
    metric = result[1][HEAD_LOSS_METRIC]
    assert float(metric) == pytest.approx(2.0)
    assert not metric.requires_grad
