"""Precision controls shared by phase benchmark factories."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, cast

import torch


@dataclass(frozen=True)
class Precision:
    """The dtypes and roundings a step trains under, named as the training
    harness names them.

    ``model_dtype`` controls model construction. ``master_dtype``,
    ``grad_dtype`` and ``fuse_narrow_gradient_accumulation`` are
    ``build_step``'s. The rest are the optimizer's own, given to it when it is
    built: the dtype it keeps its state at and how it rounds what it stores
    and the weights it steps. A gradient dtype reaches two more places on its
    own, because a step that names one and sets neither silently rounds: the
    mlops kernels are asked for weight gradients at it, and the optimizer
    reads gradients at it. ``None`` everywhere is the library's and the
    optimizer's defaults, which is what a request that names nothing gets.
    """

    model_dtype: str = "bfloat16"
    master_dtype: str | None = None
    grad_dtype: str | None = None
    opt_state_dtype: str | None = None
    parameter_rounding: str | None = None
    opt_state_rounding: str | None = None
    fuse_narrow_gradient_accumulation: bool = True

    @staticmethod
    def _dtype(name: str | None) -> torch.dtype | None:
        return None if name is None else cast(torch.dtype, getattr(torch, name))

    @property
    def master(self) -> torch.dtype | None:
        """``build_step``'s ``master_dtype``."""

        return self._dtype(self.master_dtype)

    @property
    def gradients(self) -> torch.dtype | None:
        """``build_step``'s ``grad_dtype``."""

        return self._dtype(self.grad_dtype)

    def plan_arguments(self) -> dict[str, object]:
        """Builder controls for stored gradients, masters and accumulation fusion."""

        return {
            "master_dtype": self.master,
            "grad_dtype": self.gradients,
            "fuse_narrow_gradient_accumulation": self.fuse_narrow_gradient_accumulation,
        }

    def optimizer_arguments(self) -> dict[str, object]:
        """What the optimizer is built with, beyond its defaults."""

        given: dict[str, object] = {}
        if self.grad_dtype is not None:
            given["gradient_dtype"] = self._dtype(self.grad_dtype)
        if self.opt_state_dtype is not None:
            given["opt_state_dtype"] = (
                "parameter"
                if self.opt_state_dtype == "parameter"
                else self._dtype(self.opt_state_dtype)
            )
        if self.parameter_rounding is not None:
            given["parameter_rounding"] = self.parameter_rounding
        if self.opt_state_rounding is not None:
            given["opt_state_rounding"] = self.opt_state_rounding
        return given

    def optimizer(self, base: Any) -> Any:
        """``base`` built with these settings, or ``base`` itself when it
        names none, so a request that names nothing plans exactly as before."""

        given = self.optimizer_arguments()
        return functools.partial(base, **given) if given else base

    def apply(self) -> None:
        """Ask the mlops kernels for weight gradients at the gradient dtype,
        so what the step sums comes back unrounded."""

        if self.grad_dtype is not None:
            from mlops.dispatch import set_weight_gradient_dtype

            set_weight_gradient_dtype(self._dtype(self.grad_dtype))
