from __future__ import annotations

from tools.qualification.performance_matrix import (
    _active_planning_phases,
    _termination_signal,
)
from workloads.full_model import manifests


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
        assert (item.predecessor_tokens_per_second is not None) == expected


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


def test_performance_gate_defaults_to_the_cells_it_can_judge() -> None:
    """A cell with no throughput authority cannot pass or fail the gate.

    Running one anyway only spends wall time, so the default is the set that
    carries a floor to compare against. `--cells` still reaches the others.
    """

    from tools.qualification.performance_matrix import default_cells

    assert [item.identity for item in default_cells()] == [
        "mlops_llama3",
        "mlops_qwen35",
        "mlops_olmoe",
    ]
    for item in default_cells():
        assert item.regression_tokens_per_second is not None
