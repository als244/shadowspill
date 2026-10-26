"""Precision controls shared by text benchmark factories and recipes."""

from __future__ import annotations

import argparse
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

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> Precision:
        return cls(
            model_dtype=arguments.model_dtype,
            master_dtype=arguments.master_dtype,
            grad_dtype=arguments.grad_dtype,
            opt_state_dtype=arguments.opt_state_dtype,
            parameter_rounding=arguments.parameter_rounding,
            opt_state_rounding=arguments.opt_state_rounding,
            fuse_narrow_gradient_accumulation=bool(
                arguments.fuse_narrow_gradient_accumulation
            ),
        )

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

    def lines(self) -> tuple[tuple[str, str, str], ...]:
        """The banner's rows: a label, the value, and what it means."""

        gradients = self.grad_dtype or "the weights'"
        return (
            ("model dtype", self.model_dtype, "the model weights and activations"),
            (
                "master dtype",
                self.master_dtype or "none",
                "the optimizer steps the weights themselves"
                if self.master_dtype is None
                else "a master copy of every weight trained at another dtype",
            ),
            (
                "grad dtype",
                self.grad_dtype or "weights'",
                f"gradients summed over the microbatches at {gradients} dtype;"
                + (
                    " a multiply adds a narrower gradient in as it writes it"
                    if self.fuse_narrow_gradient_accumulation
                    else " a narrower gradient is written by its multiply, then added"
                ),
            ),
            (
                "opt state dtype",
                self.opt_state_dtype or "default",
                "the moments at the optimizer's default dtype"
                if self.opt_state_dtype is None
                else "the moments at that dtype ('parameter': what it steps)",
            ),
            (
                "parameter rounding",
                self.parameter_rounding or "default",
                "how the optimizer rounds the weights it steps; its default"
                " is to nearest",
            ),
            (
                "opt state rounding",
                self.opt_state_rounding or "default",
                "ShadowSpill defaults to stochastic for BF16 AdamW moments;"
                " nearest otherwise",
            ),
        )
