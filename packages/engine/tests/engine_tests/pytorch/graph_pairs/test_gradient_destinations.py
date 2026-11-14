"""Base AOT caches are reusable across versioned gradient destination rules."""

from dataclasses import replace

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.capture._gradient_destination_ops import projection
from engine_tests.pytorch.capture.test_gradient_destinations import (
    CALLS,
    RULE,
    reference,
    values,
)
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.pytorch.graph_pairs import GraphPairStore
from shadowspill.pytorch.graph_pairs.serialization import CachedGraphArtifact
from shadowspill.pytorch.partition.artifacts import (
    Stage,
    StageExample,
    StageValueSource,
)
from shadowspill.task.inputs import TaskInputRole


def test_store_snapshots_rules_reuses_base_cache_and_versions_derived_graphs(tmp_path):
    x, a, b, tangent = values(dtype=torch.float64)
    inputs = tuple(v.requires_grad_() for v in (x, a, b))
    graph = make_fx(lambda *args: (projection(*args),))(*inputs)
    stage = Stage(
        "projection",
        "projection",
        graph,
        tuple(StageValueSource(root_input_index=i) for i in range(3)),
        tuple(
            TaskInputProvenance(role, source=name)
            for role, name in (
                (TaskInputRole.ACTIVATION, "x"),
                (TaskInputRole.PARAMETER, "a"),
                (TaskInputRole.PARAMETER, "b"),
            )
        ),
        (),
        (0,),
    )
    example = StageExample(stage, inputs, graph(*inputs))
    rules = [RULE]
    writer = GraphPairStore(tmp_path, gradient_destination_rules=rules)
    rules.clear()
    versions = []
    for version, store in (
        (1, writer),
        (
            2,
            GraphPairStore(
                tmp_path, gradient_destination_rules=(replace(RULE, version=2),)
            ),
        ),
    ):
        pairs = store.resolve(
            example, (0,), specialize_unit_tangents=False, accumulating=True
        )
        assert (store.misses, store.hits) == ((1, 0) if version == 1 else (0, 1))
        derived = pairs.options(accumulates=True)
        versions.append(
            tuple(item.pair.backward.compatibility_digest for item in derived)
        )
        for variant in derived:
            pair = variant.pair
            assert pair.backward.operator_targets.count(RULE.replacement) == 1
            restored = CachedGraphArtifact.capture(pair.backward).restore()
            assert restored.compatibility_digest == pair.backward.compatibility_digest
            expected = reference((*inputs, tangent))
            with torch.no_grad():
                saved = pair.forward.graph_module(*inputs)[-pair.saved_value_count :]
                priors = (torch.ones_like(a), torch.ones_like(b))
                CALLS.clear()
                result = restored.graph_module(*saved, tangent, *priors)
                torch.testing.assert_close(
                    result,
                    (expected[0], expected[1] + 1, expected[2] + 1),
                    rtol=1e-11,
                    atol=1e-11,
                )
                assert tuple(v.data_ptr() for v in result[1:]) == tuple(
                    v.data_ptr() for v in priors
                )
                assert CALLS == {"calls": 1}
    assert all(a != b for a, b in zip(*versions, strict=True))
