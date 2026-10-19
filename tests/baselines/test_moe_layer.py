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
