"""Keep PyTorch's ordering tokens inside each compiled task.

Tokens order effectful operators; they are compiler bookkeeping, not model
state. Export lifts them into its signature even though its input-flattening
helper does not supply them. AOT reconstructs the chain from the operators'
effect registrations after Export's standard token-removal pass.

PyTorch 2.13's zero-budget partition can also save a forward token for replay
in backward. Join that replay to backward's own chain before AOT internalizes
tokens. This preserves operation order without retaining a forward token as
an activation. It does not change which operations the partitioner replays.
"""

from __future__ import annotations

import operator

import torch
from torch.export._remove_effect_tokens_pass import _remove_effect_tokens
from torch.export.graph_signature import InputKind

from shadowspill.errors import CaptureError


def normalize_export_effects(exported: torch.export.ExportedProgram) -> None:
    """Remove signature tokens using Export's own unlift preparation pass."""

    if any(
        spec.kind is InputKind.TOKEN for spec in exported.graph_signature.input_specs
    ):
        _remove_effect_tokens(exported)


def join_recomputed_effects(
    forward: torch.fx.GraphModule,
    backward: torch.fx.GraphModule,
    *,
    num_fwd_outputs: int,
) -> None:
    """Thread replay and backward operations through one local ordered chain."""

    effects = [
        node
        for node in backward.graph.nodes
        if node.target is torch.ops.higher_order.with_effects
    ]
    placeholders = list(backward.graph.find_nodes(op="placeholder"))
    # Pinned PyTorch also uses this prefix to identify discovered backward
    # tokens in _aot_autograd.graph_capture; tokens have no distinct dtype.
    backward_tokens = [
        node for node in placeholders if node.name.startswith("tangents_token")
    ]
    roots = {node.args[0] for node in effects if node.args[0].op == "placeholder"}
    saved_tokens = roots.difference(backward_tokens)
    if not saved_tokens:
        return
    if len(backward_tokens) > 1:
        raise CaptureError("recomputed effects require one PyTorch ordered token chain")

    if backward_tokens:
        token = backward_tokens[0]
    else:
        # The derivative may contain only ordinary ATen operations. In that
        # case AOT expects no backward token arguments: make a local chain.
        with backward.graph.inserting_before(effects[0]):
            token = backward.graph.call_function(
                torch.ops.prims._make_token.default, ()
            )
        token.meta["val"] = effects[0].args[0].meta["val"]

    for node in effects:
        node.args = (token, *node.args[1:])
        token_users = [
            user
            for user in node.users
            if user.target is operator.getitem and user.args[1] == 0
        ]
        if token_users:
            token = token_users[0]
        else:
            with backward.graph.inserting_after(node):
                token = backward.graph.call_function(operator.getitem, (node, 0))
            token.meta["val"] = node.meta["val"][0]

    output = next(iter(backward.graph.find_nodes(op="output")))
    values = list(output.args[0])
    if backward_tokens:
        positions = [
            index
            for index, value in enumerate(values)
            if isinstance(value, torch.fx.Node)
            and value.target is operator.getitem
            and value.args[0] in effects
            and value.args[1] == 0
        ]
        if len(positions) != 1:
            raise CaptureError("expected one backward ordering-token output")
        values[positions[0]] = token
        output.args = (tuple(values),)
    else:
        with backward.graph.inserting_before(output):
            backward.graph.call_function(
                torch.ops.prims._sink_tokens.default, ([token],)
            )

    forward_output = next(iter(forward.graph.find_nodes(op="output")))
    saved = list(forward_output.args[0])
    for old in saved_tokens:
        if old.users:
            raise CaptureError("recomputed ordering token still has data consumers")
        # Saved SymInts and tensors have different positional orders across
        # AOT's forward/backward signatures. Their FX names preserve identity.
        positions = [
            index
            for index, value in enumerate(saved)
            if index >= num_fwd_outputs
            and isinstance(value, torch.fx.Node)
            and value.name == old.name
        ]
        if len(positions) != 1:
            raise CaptureError(f"expected one saved ordering token for {old.name}")
        del saved[positions[0]]
        backward.graph.erase_node(old)
    forward_output.args = (tuple(saved),)
    for module in (forward, backward):
        module.graph.lint()
        module.recompile()
