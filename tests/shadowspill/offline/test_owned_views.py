"""The shared plan resolves tensor views by logical group order, not worker ID."""

from dataclasses import replace

import pytest

from shadowspill import Parallelism, Program
from shadowspill.communication.binding.ranks import RankAssignment
from shadowspill.ir import AliasGroupSpec, DeviceSpec, ObjectSpec
from shadowspill.parallelism.views import owned_view_offsets


def program():
    return Program(
        devices=(DeviceSpec("device", "worker", "cuda", 0),),
        alias_groups=(AliasGroupSpec("weight", "device", 72),),
        objects=(
            ObjectSpec("model_weight", "weight", 0, 70),
            ObjectSpec("owned", "weight", 0, 36),
        ),
        tasks=(),
        task_definitions=(),
        parallelism=Parallelism.dp(ranks_per_replica=2, num_replicas=2),
        bindings={
            "owned_parameter_views": {
                "owned": {
                    "groups": ((1, 0), (3, 2)),
                    "elements": 18,
                    "dtype": "torch.bfloat16",
                    "parameter_name": "weight",
                }
            }
        },
    )


def test_owned_views_follow_logical_group_order_after_worker_reassignment(tmp_path):
    source = program()
    source.save(tmp_path / "program.ssp")
    source = Program.load(tmp_path / "program.ssp")
    assignment = RankAssignment.resolve(4, {0: 2, 1: 1, 2: 0, 3: 3})
    offsets = [
        owned_view_offsets(source, logical)["owned"]
        for logical in assignment.logical_by_worker
    ]
    assert offsets == [18, 0, 18, 0]
    assert source.objects[1].offset_bytes == 0  # Shared artifact was not mutated.


@pytest.mark.parametrize("rank", [-1, 4, True])
def test_invalid_logical_rank_is_rejected(rank):
    with pytest.raises(ValueError, match="outside"):
        owned_view_offsets(program(), rank)


def test_owned_view_requires_matching_full_buffer_capacity():
    source = program()
    source = replace(source, alias_groups=(AliasGroupSpec("weight", "device", 80),))
    with pytest.raises(ValueError, match="full-buffer geometry"):
        owned_view_offsets(source, 0)
