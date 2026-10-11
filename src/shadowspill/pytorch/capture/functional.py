"""Functionalize logical tensor operations before their representation is lowered.

Export's post-autograd decomposition unwraps subclass parameters. Doing that
before differentiating loses the logical floating-point gradient of an integer
or scaled representation. Here functionalization preserves those parameters,
and turns its final input copies into ordinary Export mutation outputs.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch.export.graph_signature import (
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta

from shadowspill.errors import CaptureError


def functionalize_logical_export(
    exported: torch.export.ExportedProgram,
    inputs: Sequence[Any],
    functional_copy: Callable[..., torch.Tensor],
) -> torch.export.ExportedProgram:
    graph = exported.graph_module
    if not any(
        node.op == "call_function"
        and getattr(getattr(node.target, "_schema", None), "is_mutable", False)
        for node in graph.graph.nodes
    ):
        return exported

    def execute(*args: Any) -> Any:
        with preserve_node_meta():
            return torch.fx.Interpreter(graph).run(*args)

    functional = make_fx(
        torch.func.functionalize(execute),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        decomposition_table={torch.ops.aten.copy.default: functional_copy},
    )(*inputs)
    placeholders = [n for n in functional.graph.nodes if n.op == "placeholder"]
    specs = exported.graph_signature.input_specs
    for node, spec in zip(placeholders, specs, strict=True):
        node.name = node.target = spec.arg.name
        if isinstance(spec.arg, ConstantArgument):
            # make_fx specializes Python scalars and omits their placeholder
            # values. Export still requires that metadata for its full input
            # signature, including unused/nested controls.
            node.meta["val"] = spec.arg.value

    # make_fx also leaves static/None fields of tuple-returning operators
    # without metadata (for example, optional attention results). Reuse the
    # already inferred parent value; do not retrace or invent tensor geometry.
    for node in functional.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and "val" not in node.meta
        ):
            source, index = node.args
            if isinstance(source, torch.fx.Node) and isinstance(
                values := source.meta.get("val"), (tuple, list, dict)
            ):
                node.meta["val"] = values[index]

    specs_by_input = dict(zip(placeholders, specs, strict=True))
    kinds = {
        InputKind.PARAMETER: OutputKind.PARAMETER_MUTATION,
        InputKind.BUFFER: OutputKind.BUFFER_MUTATION,
        InputKind.USER_INPUT: OutputKind.USER_INPUT_MUTATION,
    }
    mutations = []
    replacements = []
    for node in tuple(functional.graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not torch.ops.aten.copy_.default
            or node.args[0] not in specs_by_input
        ):
            continue
        destination, source = node.args[:2]
        assert isinstance(destination, torch.fx.Node)
        assert isinstance(source, torch.fx.Node)
        spec = specs_by_input[destination]
        if spec.kind not in kinds:
            raise CaptureError("a captured operation mutates constant tensor state")
        target = spec.arg.name if spec.kind is InputKind.USER_INPUT else spec.target
        mutations.append(
            OutputSpec(kinds[spec.kind], TensorArgument(source.name), target)
        )
        replacements.append(source)
        node.replace_all_uses_with(source)
        functional.graph.erase_node(node)

    output = next(n for n in functional.graph.nodes if n.op == "output")
    results = output.args[0]
    assert isinstance(results, (tuple, list))
    outputs = [
        replace(spec, arg=TensorArgument(value.name))
        if isinstance(value, torch.fx.Node) and isinstance(spec.arg, TensorArgument)
        else spec
        for spec, value in zip(
            exported.graph_signature.output_specs, results, strict=True
        )
    ]
    output.args = (tuple(replacements) + tuple(results),)
    functional.recompile()
    return exported._update(
        functional, ExportGraphSignature(specs, [*mutations, *outputs])
    )
