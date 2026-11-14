"""The runner-facing forward/backward interface, independent of kernel packages."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class ModuleImplementation:
    """Adapt an autograd module and its resources to the benchmark contract."""

    def __init__(
        self,
        module: "torch.nn.Module",
        *,
        cleanup: tuple[Callable, ...] = (),
        metadata: dict[str, Any] | None = None,
    ):
        self.module = module
        self.cleanup = cleanup
        self.metadata = metadata or {}

    def prepare_case(self, *, tokens: int) -> None:
        prepare = getattr(self.module, "prepare_case", None)
        if prepare is not None:
            prepare(tokens=tokens)

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        return self.module(inputs)

    def backward(self, output: "torch.Tensor", gradient: "torch.Tensor") -> None:
        output.backward(gradient)

    def zero_grad(self) -> None:
        self.module.zero_grad(set_to_none=True)
        for parameter in self.module.parameters():
            main_grad = getattr(parameter, "main_grad", None)
            if main_grad is not None:
                main_grad.zero_()
            if hasattr(parameter, "grad_added_to_main_grad"):
                parameter.grad_added_to_main_grad = False

    def named_parameters(self):
        return self.module.named_parameters()

    def named_gradients(self):
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                gradient = getattr(parameter, "main_grad", None)
                yield name, parameter.grad if gradient is None else gradient

    def close(self) -> None:
        close = getattr(self.module, "close", None)
        if close is not None:
            close()
        for callback in reversed(self.cleanup):
            callback()
