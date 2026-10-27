"""ZeRO-3 precision checks must not gather whole expert weights or states."""

from types import SimpleNamespace

import pytest
import torch

from baselines.engines.deepspeed.train import precision_evidence


class NoGather:
    def __getattr__(self, name):
        raise AssertionError(f"precision validation attempted {name}")


def case():
    master = torch.zeros(7, dtype=torch.float32)
    state = {
        master: {
            "exp_avg": torch.zeros_like(master),
            "exp_avg_sq": torch.zeros_like(master),
        }
    }
    optimizer = SimpleNamespace(
        fp32_partitioned_groups_flat=[master],
        grad_partitions_flat_buffer=torch.zeros_like(master),
        optimizer=SimpleNamespace(state=state),
        swap_optimizer=False,
    )
    access = SimpleNamespace(engine=SimpleNamespace(optimizer=optimizer))
    precision = SimpleNamespace(
        stepped_dtype="float32",
        model_dtype="bfloat16",
        grad_dtype="float32",
        resolved=lambda: {"opt_state_dtype": "float32"},
    )
    return master, optimizer, access, precision


def test_zero3_evidence_reads_only_owned_shards():
    _, _, access, precision = case()
    result = precision_evidence(NoGather(), access, precision)
    assert result["parameter_count_scope"] == "local_partitions_including_padding"
    assert result["parameters"] == result["gradient_partition_elements"] == 7
    assert result["moment_elements"] == 14


@pytest.mark.parametrize("field", ["master", "gradient", "moment", "missing_moment"])
def test_zero3_evidence_rejects_invalid_storage(field):
    master, optimizer, access, precision = case()
    if field == "master":
        optimizer.fp32_partitioned_groups_flat = [master.bfloat16()]
    elif field == "gradient":
        optimizer.grad_partitions_flat_buffer = master.bfloat16()
    elif field == "moment":
        optimizer.optimizer.state[master]["exp_avg"] = master.bfloat16()
    else:
        del optimizer.optimizer.state[master]["exp_avg"]
    with pytest.raises(ValueError):
        precision_evidence(NoGather(), access, precision)
