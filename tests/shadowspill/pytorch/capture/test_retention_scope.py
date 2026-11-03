"""Source-scoped cuts, independent VJPs and structural cache isolation."""

import json

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.objective import capture_training
from shadowspill.pytorch.capture.retention import _mark_retained, _resolve_scopes
from shadowspill.pytorch.capture.source import traceable_source_graph
from shadowspill.pytorch.graph_pairs import GraphPairStore, partition_training_capture
from tests.shadowspill.pytorch.graph_pairs.test_graph_pairs import _RepeatedNetwork


def _projection(x, weight, left, right):
    return (x @ weight + (x @ left) @ right).sigmoid()


def _projection_graph(inputs, path):
    graph = make_fx(_projection)(*inputs)
    for node in graph.graph.nodes:
        if node.op == "call_function":
            node.meta["nn_module_stack"] = {"source": (path, "Projection")}
    return graph


@pytest.mark.parametrize("default", ["cost", "recompute"])
def test_scoped_cut_applies_only_to_selected_source_and_preserves_vjp(default):
    torch.manual_seed(822)
    inputs = tuple(
        torch.randn(shape, dtype=torch.float64, requires_grad=index != 1)
        for index, shape in enumerate(((17, 5), (5, 12), (5, 2), (2, 12)))
    )
    tangent = torch.randn(17, 12, dtype=torch.float64)
    expected = _projection(*inputs)
    gradients = torch.autograd.grad(
        expected, [value for value in inputs if value.requires_grad], tangent
    )
    # Small projections are memory-bound under cost, like the low-rank
    # contribution in the real layer. Forbidding the sum as a cut forces its
    # two operands to be retained. A scoped rule must not affect another module.
    action = "recompute" if default == "cost" else "retain"
    target = "aten.add.Tensor" if default == "cost" else "aten.sigmoid.default"
    policy = RetentionPolicy(
        default=default,
        rules=(RetentionRule(target, action, module_pattern="blocks.*.local"),),
    )
    saved_products = []
    digests = []
    for path in ("blocks.0.projection", "blocks.0.local", "blocks.1.local"):
        graph = _projection_graph(inputs, path)
        pair = capture_graph_pair(
            graph,
            inputs,
            original_output=expected,
            retention=policy,
            memory_budget=1.0 if default == "cost" else 0.0,
        )
        with torch.no_grad():
            forward = pair.forward.graph_module(*inputs)
            actual = pair.backward.graph_module(
                *forward[-pair.saved_value_count :], tangent
            )
        torch.testing.assert_close(forward[0], expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            [value for value in actual if isinstance(value, torch.Tensor)],
            list(gradients),
            rtol=1e-11,
            atol=1e-11,
        )
        output = next(
            node
            for node in pair.forward.graph_module.graph.nodes
            if node.op == "output"
        )
        saved = output.args[0][-pair.saved_value_count :]
        saved_products.append(
            sum(str(node.target) == "aten.mm.default" for node in saved)
        )
        digests.append(policy.digest(graph))
        if default == "recompute":
            assert ("aten.sigmoid.default" in pair.retention.regenerated_operators) == (
                path == "blocks.0.projection"
            )
    assert digests[0] != digests[1] == digests[2]
    if default == "cost":
        assert saved_products == [0, 2, 2]


@pytest.mark.parametrize("disk_restore", [False, True])
def test_scoped_policy_keys_structurally_equal_occurrences_before_reuse(
    tmp_path, disk_restore
):
    torch.manual_seed(51)
    model = _RepeatedNetwork()
    inputs = (torch.randn(7, 8), torch.randn(7, 8))
    captured = capture_training(
        model, lambda current, x, y: (current(x) - y).square().mean(), inputs
    )
    policy = GraphPairPolicy(
        recompute=RetentionPolicy(
            default="recompute",
            rules=(
                RetentionRule(
                    "aten.add.Tensor", "retain", module_pattern="model.blocks.1"
                ),
            ),
        )
    )
    store = GraphPairStore(tmp_path)
    first = partition_training_capture(
        captured, graph_pair_store=store, graph_pair_policy=policy
    )
    # The two middle blocks normally share a structural key. Only block 1
    # retains the sum, so reading block 2 from that entry would be incorrect.
    assert store.misses == 4 and store.hits == 0
    if disk_restore:
        store = GraphPairStore(tmp_path)
    selected = partition_training_capture(
        captured, graph_pair_store=store, graph_pair_policy=policy
    )
    assert store.hits == 4
    assert store.misses == (0 if disk_restore else 4)
    for old, new in zip(first.stages, selected.stages, strict=True):
        for variant in ("save", "recompute"):
            assert (
                old.graph_pairs.variant(variant).pair.forward.compatibility_digest
                == new.graph_pairs.variant(variant).pair.forward.compatibility_digest
            )
    left, right = (stage.graph_pairs for stage in selected.stages[1:3])
    assert (
        left.variant("save").pair.forward.compatibility_digest
        == right.variant("save").pair.forward.compatibility_digest
    )
    assert (
        left.variant("recompute").pair.forward.compatibility_digest
        != right.variant("recompute").pair.forward.compatibility_digest
    )
    manifests = [
        json.loads(path.read_text()) for path in tmp_path.rglob("manifest.json")
    ]
    indices = [
        entry["retention"]["scoped"]["recompute"]["rules"][0]["source_indices"]
        for entry in manifests
    ]
    assert sum(bool(value) for value in indices) == 1


def test_scope_conflicts_and_missing_provenance_fail_closed():
    graph = make_fx(lambda x: x.sin().cos())(torch.randn(7))
    rule = RetentionRule("aten.sin.default", "retain", module_pattern="block")
    policy = RetentionPolicy(rules=(rule,))
    with pytest.raises(CaptureError, match="source module paths"):
        policy.digest(graph)
    for node in graph.graph.nodes:
        if node.op == "call_function":
            node.meta["nn_module_stack"] = {"source": ("block", "Block")}
    conflict = RetentionPolicy(
        rules=(rule, RetentionRule("aten.sin.default", "recompute"))
    )
    with pytest.raises(CaptureError, match="rules conflict"):
        conflict.digest(graph)
    scopes = _resolve_scopes(graph, policy)
    with pytest.raises(CaptureError, match="source-operation provenance"):
        _mark_retained(graph, policy, scopes=scopes)
    tagged = traceable_source_graph(graph)
    _mark_retained(tagged, policy, scopes=scopes)
    node = next(node for node in tagged.graph.nodes if str(node.target) == rule.target)
    assert node.meta["recompute"].name == "MUST_SAVE"


def test_scoped_rule_identity_is_order_independent_and_matches_ancestor_paths():
    graph = make_fx(lambda x: x.sin().cos())(torch.randn(7))
    for node in graph.graph.nodes:
        if node.op == "call_function":
            node.meta["nn_module_stack"] = {
                "parent": ("blocks.12.local", "Block"),
                "child": ("blocks.12.local.inner", "Leaf"),
            }
    rules = (
        RetentionRule("aten.sin.default", "retain", module_pattern="blocks.*.local"),
        RetentionRule(
            "aten.cos.default", "recompute", module_pattern="blocks.*.local.inner"
        ),
    )
    policy = RetentionPolicy(rules=rules)
    assert policy.digest(graph) == RetentionPolicy(rules=rules[::-1]).digest(graph)
    assert policy.capture_identity(graph)["scoped"]["rules"][1]["source_indices"] == [
        0,
        1,
    ]
    with pytest.raises(ValueError, match="at most once"):
        RetentionPolicy(rules=(rules[0], rules[0]))
    for invalid in ("", "  ", 4):
        with pytest.raises(ValueError, match="module_pattern"):
            RetentionRule("aten.sin.default", "retain", module_pattern=invalid)


def test_scoped_auxiliary_results_are_retained_with_their_producer():
    from tests.shadowspill.pytorch.capture.test_retention import _sincos

    x = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    graph = make_fx(lambda x, w: _sincos(x @ w)[0])(x, weight)
    for node in graph.graph.nodes:
        if node.op == "call_function":
            node.meta["nn_module_stack"] = {"source": ("model.sequence", "Sequence")}
    pair = capture_graph_pair(
        graph,
        (x, weight),
        original_output=_sincos(x @ weight)[0],
        memory_budget=0,
        retention=RetentionPolicy(
            default="recompute",
            rules=(
                RetentionRule(
                    "shadowspill_retention_test.sincos.default",
                    "retain",
                    module_pattern="model.sequence",
                ),
            ),
        ),
    )
    assert (
        "shadowspill_retention_test.sincos.default"
        not in pair.retention.regenerated_operators
    )
    assert "aten.mm.default" not in pair.retention.regenerated_operators
    values = pair.forward.graph_module(x, weight)
    tangent = torch.randn_like(values[0])
    actual = pair.backward.graph_module(*values[-pair.saved_value_count :], tangent)
    expected = torch.autograd.grad(_sincos(x @ weight)[0], (x, weight), tangent)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("attributed", [False, True])
def test_scoped_rules_follow_decomposed_source_operations(attributed):
    from torch.fx import Graph, GraphModule

    x = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(12, 5, dtype=torch.float64, requires_grad=True)
    graph = Graph()
    value = graph.placeholder("x")
    parameter = graph.placeholder("weight")
    linear = graph.call_function(torch.ops.aten.linear.default, (value, parameter))
    result = graph.call_function(torch.ops.aten.sin.default, (linear,))
    graph.output(result)
    if attributed:
        for node in (linear, result):
            node.meta["nn_module_stack"] = {
                "source": ("model.projection", "Projection")
            }
    module = GraphModule({}, graph)
    if not attributed:
        with pytest.raises(CaptureError, match="source module paths"):
            capture_graph_pair(
                module,
                (x, weight),
                original_output=module(x, weight),
                memory_budget=0,
                retention=RetentionPolicy(
                    default="recompute",
                    rules=(
                        RetentionRule(
                            "aten.mm.default",
                            "retain",
                            module_pattern="model.projection",
                        ),
                    ),
                ),
            )
        return
    pair = capture_graph_pair(
        module,
        (x, weight),
        original_output=module(x, weight),
        memory_budget=0,
        retention=RetentionPolicy(
            default="recompute",
            rules=(
                RetentionRule(
                    "aten.mm.default", "retain", module_pattern="model.projection"
                ),
            ),
        ),
    )
    assert "aten.mm.default" not in pair.retention.regenerated_operators
    values = pair.forward.graph_module(x, weight)
    tangent = torch.randn_like(values[0])
    actual = pair.backward.graph_module(*values[-pair.saved_value_count :], tangent)
    expected = torch.autograd.grad(module(x, weight), (x, weight), tangent)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_scoped_rules_cannot_replay_random_state():
    x = torch.ones(7, requires_grad=True)
    module = make_fx(lambda x: x * torch.rand_like(x))(x)
    for node in module.graph.nodes:
        if node.op == "call_function":
            node.meta["nn_module_stack"] = {"source": ("model.random", "Random")}
    policy = RetentionPolicy(
        rules=(
            RetentionRule(
                "aten.rand_like.default", "recompute", module_pattern="model.random"
            ),
        )
    )
    with pytest.raises(CaptureError, match="RNG/effect state"):
        capture_graph_pair(module, (x,), original_output=module(x), retention=policy)


def test_scoped_override_cannot_hide_another_occurrences_changed_cost(monkeypatch):
    from torch.utils.flop_counter import flop_registry

    from tests.shadowspill.pytorch.capture.test_retention import _pass_over

    graph = make_fx(
        lambda x, w: _pass_over(_pass_over(x, w), w), tracing_mode="fake"
    )(torch.randn(7, 5), torch.randn(5))
    nodes = [node for node in graph.graph.nodes if node.op == "call_function"]
    assert len(nodes) == 2
    for node, path in zip(nodes, ("model.first", "model.selected"), strict=True):
        node.meta["nn_module_stack"] = {"source": (path, "Block")}
    policy = RetentionPolicy(
        rules=(
            RetentionRule(
                "shadowspill_retention_test.pass_over.default",
                "retain",
                module_pattern="model.selected",
            ),
        )
    )
    before, classes = policy.digest(graph), policy.operator_classes(graph)
    monkeypatch.setitem(
        flop_registry,
        torch.ops.shadowspill_retention_test.pass_over,
        lambda *args, **kwargs: 10**12,
    )
    assert classes == policy.operator_classes(graph)
    assert before != policy.digest(graph)
