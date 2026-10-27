"""Integer statistics may be synthetic only after proving every downstream use."""

from dataclasses import replace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.partition.artifacts import StageRecord, StageValueSource
from shadowspill.pytorch.partition.split import SplitExportGraph
from shadowspill.pytorch.partition.statistics import numeric_outputs


def chain(*functions):
    value = torch.ones(4, dtype=torch.int64)
    producer = make_fx(lambda x: (x.clone(),))(value)
    records = [
        StageRecord(
            "source",
            producer,
            (value,),
            (StageValueSource(root_input_index=0),),
            (value,),
        )
    ]
    for index, function in enumerate(functions):
        graph = make_fx(function)(value)
        output = graph(value)
        records.append(
            StageRecord(
                f"consumer{index}",
                graph,
                (value,),
                (
                    StageValueSource(
                        producer_stage_index=index, producer_output_index=0
                    ),
                ),
                output,
            )
        )
        value = output[0]
    return SplitExportGraph(producer, tuple(records), {}, {})


def test_numeric_proof_follows_forwarding_and_pure_arithmetic():
    split = chain(lambda x: (x.clone(),), lambda x: (x.float().sum() / 4,))
    assert numeric_outputs(split) == {(0, 0), (1, 0)}


@pytest.mark.parametrize(
    "consumer",
    [
        lambda x: (torch.arange(4)[x],),
        lambda x: (torch.unique(x),),
        lambda x: (x.add_(1),),
        lambda x: (x.float().sin(),),
    ],
)
def test_indices_data_dependent_shapes_and_mutation_require_real_values(consumer):
    assert numeric_outputs(chain(lambda x: (x.clone(),), consumer)) == set()


def test_one_unsafe_branch_prevents_numeric_classification():
    split = chain(lambda x: (x.float().sum() + torch.arange(4)[x].sum(),))
    assert numeric_outputs(split) == set()


def test_long_forwarding_chains_do_not_use_python_recursion():
    split = chain(lambda x: (x.clone(),), lambda x: (x.float().sum(),))
    records = [split.stages[0]]
    for index in range(1500):
        records.append(
            replace(
                split.stages[1] if index < 1499 else split.stages[2],
                input_sources=(
                    StageValueSource(
                        producer_stage_index=index,
                        producer_output_index=0,
                    ),
                ),
            )
        )
    actual = numeric_outputs(replace(split, stages=tuple(records)))
    assert actual == {(index, 0) for index in range(1500)}
