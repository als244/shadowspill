"""Locate where a model's step stops being bitwise reproducible.

A step is only bitwise reproducible if every kernel under it is. When a
qualification cell fails a bitwise replay, the useful question is not whether
the step is reproducible but which stage of it is not, and this narrows that
to a module rather than leaving it at "somewhere in the backward".

The probe runs the same fixed input through the same model twice and compares
bitwise at three widening levels: the objective, every module's forward
output, and every module's incoming gradient. Modules are reported in
execution order, so the first entry is where reproducibility is lost and
everything after it is downstream consequence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class Divergence:
    """One tensor that differed between two runs of the same computation."""

    name: str
    maximum_absolute: float
    numel: int


@dataclass(slots=True)
class ProbeResult:
    """What two identical runs agreed and disagreed about."""

    objective_reproducible: bool
    objective_values: tuple[float, float]
    forward_divergences: list[Divergence] = field(default_factory=list)
    backward_divergences: list[Divergence] = field(default_factory=list)
    gradient_divergences: list[Divergence] = field(default_factory=list)
    modules_observed: int = 0
    gradients_observed: int = 0

    @property
    def reproducible(self) -> bool:
        return not (
            self.forward_divergences
            or self.backward_divergences
            or self.gradient_divergences
        )

    @property
    def first_forward(self) -> Divergence | None:
        return self.forward_divergences[0] if self.forward_divergences else None

    @property
    def first_backward(self) -> Divergence | None:
        return self.backward_divergences[0] if self.backward_divergences else None


def _compare(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    order: Sequence[str],
) -> list[Divergence]:
    """Return the entries that differ, in the order they were produced."""

    found: list[Divergence] = []
    for name in order:
        first, second = left.get(name), right.get(name)
        if first is None or second is None or torch.equal(first, second):
            continue
        difference = (first.float() - second.float()).abs().max()
        found.append(Divergence(name, float(difference), int(first.numel())))
    return found


@contextmanager
def _recording(
    model: nn.Module,
    forward_into: dict[str, torch.Tensor],
    backward_into: dict[str, torch.Tensor],
    order: list[str],
) -> Iterator[None]:
    """Capture every module's forward output and incoming gradient."""

    handles: list[Any] = []

    def forward_hook(name: str) -> Callable[..., None]:
        def record(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                if name not in order:
                    order.append(name)
                forward_into[name] = tensor.detach().clone()

        return record

    def backward_hook(name: str) -> Callable[..., None]:
        def record(_module: nn.Module, grad_input: Any, _grad_output: Any) -> None:
            tensor = next(
                (item for item in grad_input if isinstance(item, torch.Tensor)),
                None,
            )
            if tensor is not None:
                backward_into[name] = tensor.detach().clone()

        return record

    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(forward_hook(name)))
        handles.append(module.register_full_backward_hook(backward_hook(name)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def probe_step(
    model: nn.Module,
    objective: Callable[[], torch.Tensor],
    *,
    capture_modules: bool = True,
) -> ProbeResult:
    """Run one step twice and report where the two runs stopped agreeing.

    ``objective`` must run the whole forward and return the scalar to
    differentiate, reading whatever inputs the caller fixed. It is called
    twice with nothing changed in between, so anything that differs is the
    computation's own doing.
    """

    runs: list[dict[str, Any]] = []
    order: list[str] = []
    for _ in range(2):
        for parameter in model.parameters():
            parameter.grad = None
        forward: dict[str, torch.Tensor] = {}
        backward: dict[str, torch.Tensor] = {}
        if capture_modules:
            with _recording(model, forward, backward, order):
                loss = objective()
                loss.backward()  # type: ignore[no-untyped-call]
        else:
            loss = objective()
            loss.backward()  # type: ignore[no-untyped-call]
        runs.append(
            {
                "loss": loss.detach().clone(),
                "forward": forward,
                "backward": backward,
                "gradients": {
                    name: value.grad.detach().clone()
                    for name, value in model.named_parameters()
                    if value.grad is not None
                },
            }
        )
    first, second = runs
    gradient_order = [name for name, _ in model.named_parameters()]
    return ProbeResult(
        objective_reproducible=bool(torch.equal(first["loss"], second["loss"])),
        objective_values=(float(first["loss"]), float(second["loss"])),
        forward_divergences=_compare(first["forward"], second["forward"], order),
        backward_divergences=_compare(
            first["backward"], second["backward"], list(reversed(order))
        ),
        gradient_divergences=_compare(
            first["gradients"], second["gradients"], gradient_order
        ),
        modules_observed=len(order),
        gradients_observed=len(first["gradients"]),
    )


__all__ = ["Divergence", "ProbeResult", "probe_step"]
