"""Accumulate parameter gradients inside the compiled backward task."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any, cast

import torch
from torch._guards import detect_fake_mode
from torch._prims_common import get_computation_dtype

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.capture.storage import ExplicitMutation
from shadowspill.pytorch.operations import ACCUMULATE_MATMUL, ADDING_INTO
from shadowspill.task.inputs import TaskInputRole

from ..torch_deprecations import copy_graph_module
from .nodes import (
    _ADDS_INTO,
    _MOVING_ONLY,
    _WRITING_AT_DTYPE,
    _record_value,
    _undo_move,
)


def accumulate_gradient_outputs(
    backward: GraphArtifact,
    leaf_indices: Sequence[int],
    *,
    fuse_narrow_gradient_accumulation: bool = True,
) -> GraphArtifact:
    """Return a backward that adds its gradients onto ones it is given.

    Every microbatch after the first contributes to a gradient that already
    exists. Left alone, the graph hands back a fresh gradient and something
    outside it has to do the addition, which puts real device work between
    tasks where no plan accounts for it.

    So the graph takes the running gradient as an argument and returns the sum.
    This adds onto outputs rather than rewriting the operations that produce
    them, so it holds for any backward: whatever computed the gradient, its
    result is what gets added to. The addition is in place, so the running
    gradient keeps its storage. It is declared as a mutation of that argument,
    which is how the runtime knows the returned gradient is the argument rather
    than a new one.

    The compiler folds the add into the kernel it generates for whatever
    produced the contribution -- a reduction, say -- so nothing is written
    twice. A matrix multiply is a library call that writes its result before
    anything can read it, so a gradient one computes, moved at most by views,
    is added by the multiply itself instead (:func:`accumulate_matmul_`):
    ``C = A @ B + C`` into the running gradient, where the device has a kernel
    for it at these dtypes. The multiply adds gradients narrower than its
    accumulator, such as bf16, only with ``fuse_narrow_gradient_accumulation``. This
    avoids a product buffer and separate addition, but the BLAS kernel
    controls rounding: even one kernel may round the product before adding
    the prior gradient. The option permits fusion, not a rounding guarantee.
    """

    if not leaf_indices:
        return backward
    graph_module = copy_graph_module(backward.graph_module)
    graph = graph_module.graph
    output_node = next(node for node in graph.nodes if node.op == "output")
    outputs = list(output_node.args[0])
    for leaf in leaf_indices:
        if leaf < 0 or leaf >= len(outputs) or outputs[leaf] is None:
            raise CaptureError(f"backward has no gradient output at leaf {leaf}")

    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(backward.example_arguments):
        raise CaptureError("backward placeholder count changed before accumulation")
    # The arguments this adds have to belong to the same fake mode as the ones
    # already there, and the caller need not be inside that mode.
    fake_mode = detect_fake_mode(backward.example_arguments)
    anchor = placeholders[-1]
    priors: list[torch.Tensor] = []
    mutations: list[ExplicitMutation] = []
    replaced: list[torch.fx.Node] = []
    for offset, leaf in enumerate(leaf_indices):
        produced = outputs[leaf]
        value = produced.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise CaptureError(f"backward gradient at leaf {leaf} has no geometry")
        with graph.inserting_after(anchor):
            prior = graph.placeholder(f"shadowspill_prior_grad_{leaf}")
        prior.meta = dict(produced.meta)
        with fake_mode if fake_mode is not None else nullcontext():
            extent = value.untyped_storage().nbytes() // value.element_size()
            if extent > value.numel():
                owner = value.new_zeros((extent,))
                prior_value = owner.as_strided(value.shape, value.stride())
            else:
                prior_value = torch.zeros_like(value)
            _record_value(prior, prior_value)
        anchor = prior
        adding = _add_in_multiply(
            graph,
            produced,
            prior,
            outputs,
            fake_mode,
            fuse_narrow_gradient_accumulation,
        )
        if adding is not None:
            outputs[leaf] = prior
            replaced.extend(adding)
        else:
            with graph.inserting_before(output_node):
                total = graph.call_function(
                    torch.ops.aten.add_.Tensor, args=(prior, produced)
                )
            total.meta = dict(produced.meta)
            outputs[leaf] = total
        priors.append(prior.meta["val"])
        mutations.append(
            ExplicitMutation(
                input_position=len(backward.example_arguments) + offset,
                output_leaf_index=leaf,
                target=f"shadowspill_prior_grad_{leaf}",
            )
        )
    output_node.args = (tuple(outputs),)
    for node in replaced:
        graph.erase_node(node)
    graph.lint()
    graph_module.recompile()
    return GraphArtifact.capture(
        kind="backward",
        graph_module=graph_module,
        example_inputs=(*backward.example_arguments, *priors),
        explicit_mutations=tuple(mutations),
        input_provenance=(
            *backward.input_provenance,
            *(TaskInputProvenance(role=TaskInputRole.GRADIENT) for _ in leaf_indices),
        ),
    )


def _add_in_multiply(
    graph: torch.fx.Graph,
    produced: torch.fx.Node,
    prior: torch.fx.Node,
    outputs: Sequence[object],
    fake_mode: Any,
    fuse_narrow_gradient_accumulation: bool,
) -> list[torch.fx.Node] | None:
    """Have the matrix multiply that computes ``produced`` add it into ``prior``.

    It does when ``produced`` is a multiply's result, moved at most, that
    nothing else reads; when every move can be undone on ``prior`` as a view,
    so the multiply's result lands where ``produced`` would have been added;
    and when PyTorch has an in-place kernel for it on the device that accepts
    these dtypes -- the operator's own check. A running gradient narrower than
    the dtype the multiply sums at is added so only with
    ``fuse_narrow_gradient_accumulation``. Returns the nodes it replaced, for the caller
    to erase once the output no longer reads them, or ``None``.
    """

    chain = [produced]
    while chain[-1].target in _MOVING_ONLY:
        source = chain[-1].args[0]
        if not isinstance(source, torch.fx.Node):
            return None
        chain.append(source)
    multiply = chain[-1]
    if (
        multiply.target not in _ADDS_INTO
        or outputs.count(produced) != 1
        or any(len(node.users) != 1 for node in chain)
    ):
        return None
    running = cast(torch.Tensor, prior.meta["val"])
    adding = ADDING_INTO[cast(torch.Tensor, multiply.meta["val"]).dim()]
    schema = adding._schema
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        f"{schema.name}.{schema.overload_name}",
        torch._C._dispatch_key_for_device(running.device.type),
    ):
        return None
    left, right = (cast(torch.fx.Node, item) for item in multiply.args[:2])
    operands = cast(torch.Tensor, left.meta["val"]).dtype
    if not fuse_narrow_gradient_accumulation and running.dtype != get_computation_dtype(
        operands
    ):
        return None
    undoing = [undo for undo in map(_undo_move, chain[:-1]) if undo is not None]
    # Which dtypes the multiply writes at is the operator's own check, made by
    # its overload that takes one: the in-place overloads' fake forms are
    # decompositions that misread it.
    writing = cast(
        Callable[..., torch.Tensor],
        _WRITING_AT_DTYPE.get(multiply.target, multiply.target),
    )
    with fake_mode if fake_mode is not None else nullcontext():
        try:
            views = [running]
            for target, arguments in undoing:
                views.append(target(views[-1], *arguments))
            writing(left.meta["val"], right.meta["val"], running.dtype)
        except RuntimeError:  # a move no view undoes, or a dtype it cannot write
            return None
    with graph.inserting_before(multiply):
        view = prior
        for (target, arguments), value in zip(undoing, views[1:], strict=True):
            view = graph.call_function(target, args=(view, *arguments))
            _record_value(view, value)
        graph.call_function(ACCUMULATE_MATMUL, args=(view, left, right))
    return chain
