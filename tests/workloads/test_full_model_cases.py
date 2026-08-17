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


def test_only_mlops_cells_have_historical_throughput_authorities() -> None:
    for item in manifests():
        assert (item.historical_tokens_per_second is not None) == (
            item.implementation == "mlops"
        )


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
