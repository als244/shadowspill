"""Select a model call without changing its registered tensor namespace."""

import copy
from collections.abc import Callable
from types import MethodType
from typing import Any

from torch import nn

from .aot import ExportCapture


def select_forward(model: nn.Module, forward_fn: Callable[..., Any]) -> nn.Module:
    """Change only the capture copy; callbacks can still call its original forward.

    The shallow view shares the fake model's registered tensors and children.
    Neither the caller's model nor its tensor names change. The callback is used
    during export only and is not a dependency of the saved Program.
    """
    view = copy.copy(model)

    def call(_model: nn.Module, *args: Any, **kwargs: Any) -> Any:
        return forward_fn(view, *args, **kwargs)

    model.forward = MethodType(call, model)
    return view


def restore_module_paths(
    capture: ExportCapture, model: nn.Module, view: nn.Module
) -> None:
    """Resolve closure-derived module paths back to the registered names.

    Strict Export can name a shared child through the callback closure. Its
    stack key still identifies that child, so partitioning can retain the source
    model's module names. Tensor names and unregistered modules are unchanged.
    """
    paths = {str(id(module)): path for path, module in model.named_modules()}
    paths[str(id(view))] = ""
    for node in capture.exported_program.graph.nodes:
        stack = node.meta.get("nn_module_stack")
        if isinstance(stack, dict):
            node.meta["nn_module_stack"] = {
                key: (paths.get(key, path), module_type)
                for key, (path, module_type) in stack.items()
            }
