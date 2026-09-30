"""Explicit storage and training precision for qualification workloads."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

FLOAT_DTYPES = ("float16", "bfloat16", "float32")
DTYPE_FIELDS = ("model_dtype", "master_dtype", "grad_dtype", "opt_state_dtype")


@dataclass(frozen=True, slots=True)
class TrainingDtypes:
    """Model weights, optional masters, accumulated gradients, and moments.

    ``parameter`` accumulates gradients at model-weight dtype. Optimizers read
    the accumulated gradient at the dtype of the parameter they update (the
    master when present), matching the public ``plan_step`` contract.
    """

    model_dtype: str = "bfloat16"
    master_dtype: str = "none"
    grad_dtype: str = "parameter"
    opt_state_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in DTYPE_FIELDS:
            choices = FLOAT_DTYPES
            if name == "master_dtype":
                choices = (*choices, "none")
            elif name == "grad_dtype":
                choices = (*choices, "parameter")
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")

    def as_dict(self) -> dict[str, str]:
        """Resolved dtypes, so an artifact never hides an inherited choice."""
        return {
            "model_dtype": self.model_dtype,
            "master_dtype": self.master_dtype,
            "grad_dtype": (
                self.model_dtype if self.grad_dtype == "parameter" else self.grad_dtype
            ),
            "opt_state_dtype": self.opt_state_dtype,
        }

    def description(self) -> str:
        values = self.as_dict()
        return ", ".join(
            f"{label}={values[name]}"
            for label, name in zip(
                ("model", "masters", "gradients", "optimizer state"),
                DTYPE_FIELDS,
                strict=True,
            )
        )

    def plan_arguments(self) -> dict[str, torch.dtype | None]:
        return {
            "master_dtype": (
                None
                if self.master_dtype == "none"
                else getattr(torch, self.master_dtype)
            ),
            "grad_dtype": (
                None
                if self.grad_dtype == "parameter"
                else getattr(torch, self.grad_dtype)
            ),
        }

    def optimizer(self) -> Any:
        import mlops

        return partial(
            mlops.optim.AdamW,
            opt_state_dtype=getattr(torch, self.opt_state_dtype),
            gradient_dtype=getattr(
                torch,
                self.model_dtype if self.master_dtype == "none" else self.master_dtype,
            ),
        )
