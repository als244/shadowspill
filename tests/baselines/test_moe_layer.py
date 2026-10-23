import json

import pytest

from baselines.moe_layer.config import Shape, summarize
from baselines.moe_layer.plot import collect
from baselines.moe_layer.sweep import selected_result


def test_full_counts_forward_input_and_weight_products():
    shape = Shape(width=4, hidden=6, experts=8, top_k=2)
    # Three projections per selected expert and one dense router.
    forward = 2 * 10 * (2 * 3 * 4 * 6 + 4 * 8)
    assert shape.flops(10, "full") == {
        "forward": forward,
        "backward": 2 * forward,
        "total": 3 * forward,
    }


def test_lora_omits_base_weight_gradients_and_includes_adapter_training():
    shape = Shape(width=4, hidden=6, experts=8, top_k=2, lora_rank=2)
    base_forward = 2 * 10 * (2 * 3 * 4 * 6 + 4 * 8)
    adapter_forward = 2 * 10 * 2 * 2 * (2 * 4 + 3 * 6)
    actual = shape.flops(10, "lora")
    assert actual["forward"] == base_forward + adapter_forward
    assert actual["total"] == 2 * base_forward + 3 * adapter_forward
    assert actual["total"] == actual["forward"] + actual["backward"]


def test_ep_uses_the_maximum_rank_before_aggregating_iterations():
    actual = summarize([[1, 2, 9], [9, 2, 1]])
    assert actual["median_seconds"] == 9
    assert actual["per_iteration_max_rank_seconds"] == [9, 2, 9]
    with pytest.raises(ValueError):
        summarize([[1, 2], [1]])


def test_later_failed_attempt_does_not_reuse_older_success(tmp_path):
    case = tmp_path / "cases/sonicmoe/bf16/full/tokens-4096"
    for index, status in enumerate(("completed", "failed"), 1):
        attempt = case / f"attempt-{index:03d}"
        attempt.mkdir(parents=True)
        (attempt / "outcome.json").write_text(json.dumps({"status": status}))
    assert selected_result(case)["status"] == "failed"
    rows = collect(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert "tflops_per_gpu" not in rows[0]


def test_phase_plot_uses_measured_combined_interval(tmp_path):
    from baselines.moe_layer.plot import phase_timings

    # Forward and backward maxima occur on different ranks. The direct total
    # is 10 ms, not the 18 ms sum of the separate maxima.
    for rank, forward, backward in ((0, 0.009, 0.001), (1, 0.001, 0.009)):
        directory = tmp_path / f"rank-{rank:05d}"
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "forward_gpu_seconds": forward,
                            "backward_gpu_seconds": backward,
                            "gpu_seconds": forward + backward,
                        }
                    ]
                }
            )
        )
    result = {
        "request": {
            "flops_per_rank": {
                "forward": 9e9,
                "backward": 18e9,
                "total": 27e9,
            }
        }
    }
    phases = phase_timings(result, tmp_path)
    assert phases["total"]["median_seconds"] == pytest.approx(0.010)
    assert phases["forward"]["median_seconds"] == pytest.approx(0.009)
    assert phases["backward"]["median_seconds"] == pytest.approx(0.009)
    assert phases["total"]["effective_tflops_per_gpu"] == pytest.approx(2.7)


def test_bars_and_curves_use_same_phase_data_without_zero_for_failure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from baselines.moe_layer.plot import draw_panel

    rows = [
        dict(
            backend="sonicmoe",
            precision="bf16",
            training="full",
            tokens_per_rank=2048,
            status="completed",
            forward_milliseconds=2.0,
            forward_tflops_per_gpu=25.0,
        ),
        dict(
            backend="deepep",
            precision="bf16",
            training="full",
            tokens_per_rank=2048,
            status="out_of_memory",
        ),
    ]
    figure, (bars, curves) = plt.subplots(1, 2)
    common = dict(
        precision="bf16",
        training="full",
        tokens=[2048],
        backends=["sonicmoe", "deepep"],
        phase="forward",
    )
    draw_panel(bars, rows, metric="runtime", **common)
    draw_panel(curves, rows, metric="throughput", **common)
    assert bars.patches[0].get_height() == 2.0
    assert curves.lines[0].get_ydata()[0] == 25.0
    import math

    assert math.isnan(bars.patches[1].get_height())
    assert math.isnan(curves.lines[1].get_ydata()[0])
    plt.close(figure)


def test_memory_table_preserves_rank_and_phase_peaks(tmp_path):
    from baselines.moe_layer.plot import collect_memory

    attempt = tmp_path / "case"
    attempt.mkdir()
    memory = {
        str(rank): {
            phase: {
                "peak_allocated_bytes": (rank + offset) * 2**30,
                "peak_reserved_bytes": (rank + offset + 1) * 2**30,
                "sampled_device_peak_bytes": None,
                "device_peak_is_sampled_lower_bound": True,
            }
            for phase, offset in (("forward", 2), ("backward", 5))
        }
        for rank in (0, 1)
    }
    (attempt / "result.json").write_text(json.dumps({"memory_by_rank": memory}))
    rows = collect_memory(
        tmp_path,
        [
            dict(
                backend="sonicmoe",
                precision="bf16",
                training="full",
                tokens_per_rank=2048,
                status="completed",
                path="case",
            )
        ],
    )
    by_rank_phase = {(row["rank"], row["phase"]): row for row in rows}
    assert by_rank_phase[0, "forward"]["peak_allocated_gib"] == 2
    assert by_rank_phase[0, "backward"]["peak_allocated_gib"] == 5
    assert by_rank_phase[1, "backward"]["peak_allocated_gib"] == 6
    assert by_rank_phase[1, "backward"]["sampled_device_peak_gib"] is None
