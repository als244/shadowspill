"""The precision contract every engine exposes, as the quickstart names it.

``model_dtype`` is what the weights and activations compute at;
``master_dtype`` what a trained weight's master copy is kept at, or none to
step the weights themselves; ``grad_dtype`` what gradients are summed at
over a step (under PyTorch's autograd each microbatch's is created at its
weight's own dtype first); ``opt_state_dtype`` what the optimizer keeps its
state at, or ``parameter`` for the dtype of what it steps; the two roundings
how the optimizer rounds the weights it steps and the state it stores;
``fuse_narrow_gradient_accumulation`` whether, with gradients narrower than
a multiply's float32 accumulator, a later microbatch's weight gradient is
added into the running gradient by the multiply as it writes it instead of
written at the narrow dtype and added after (``PRECISION.md``); it asks
nothing of float32 gradients.

The baselines default to BF16 compute and gradients with FP32 masters and
moments, the multiplies adding into the BF16 gradients themselves; an engine
that cannot sum gradients at the model dtype under its masters, or cannot
fuse that accumulation, defaults to its own and says so. An engine states what it can
honor as ``Capabilities``; a request outside them is refused with the
reason, never substituted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace

DTYPES = ("bfloat16", "float16", "float32")
ROUNDINGS = ("nearest", "stochastic")

#: The fields of a request, in the order the flags take them.
FIELDS = (
    "model_dtype",
    "master_dtype",
    "grad_dtype",
    "opt_state_dtype",
    "parameter_rounding",
    "opt_state_rounding",
    "fuse_narrow_gradient_accumulation",
)


class Unsupported(ValueError):
    """An engine cannot honor the requested precision; the message says why."""


@dataclass(frozen=True)
class Precision:
    model_dtype: str = "bfloat16"
    master_dtype: str | None = "float32"
    grad_dtype: str | None = None
    opt_state_dtype: str | None = "float32"
    parameter_rounding: str | None = None
    opt_state_rounding: str | None = None
    fuse_narrow_gradient_accumulation: bool = True

    def __post_init__(self) -> None:
        if self.model_dtype not in DTYPES:
            raise ValueError(f"model_dtype must be one of {DTYPES}")
        for name in ("master_dtype", "grad_dtype"):
            value = getattr(self, name)
            if value is not None and value not in DTYPES:
                raise ValueError(f"{name} must be one of {DTYPES} or none")
        if self.opt_state_dtype is not None and self.opt_state_dtype not in (
            *DTYPES,
            "parameter",
        ):
            raise ValueError(
                f"opt_state_dtype must be one of {DTYPES}, parameter or none"
            )
        for name in ("parameter_rounding", "opt_state_rounding"):
            value = getattr(self, name)
            if value is not None and value not in ROUNDINGS:
                raise ValueError(f"{name} must be one of {ROUNDINGS} or none")
        if not isinstance(self.fuse_narrow_gradient_accumulation, bool):
            raise ValueError("fuse_narrow_gradient_accumulation must be a boolean")

    @property
    def narrow_gradients(self) -> bool:
        """Whether gradients are summed narrower than the float32 a multiply
        of bfloat16 or float16 operands sums its products at."""
        return (self.grad_dtype or self.model_dtype) != "float32"

    @property
    def stepped_dtype(self) -> str:
        """What the optimizer steps: the masters when kept, else the weights."""
        return self.master_dtype or self.model_dtype

    def as_dict(self) -> dict[str, object]:
        """The request as made, ``None`` where a system default applies."""
        return asdict(self)

    def resolved(self) -> dict[str, object]:
        """What the step computes once every default is settled."""
        return {
            "model_dtype": self.model_dtype,
            "master_dtype": self.master_dtype or "none",
            "grad_dtype": self.grad_dtype or self.model_dtype,
            "opt_state_dtype": (
                self.stepped_dtype
                if self.opt_state_dtype == "parameter"
                else self.opt_state_dtype or "default"
            ),
            "parameter_rounding": self.parameter_rounding or "default",
            "opt_state_rounding": self.opt_state_rounding or "default",
            "fuse_narrow_gradient_accumulation": self.fuse_narrow_gradient_accumulation,
        }

    def describe(self) -> str:
        values = self.resolved()
        return ", ".join(f"{name}={values[name]}" for name in FIELDS)

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> Precision:
        return cls(**{name: getattr(arguments, name) for name in FIELDS})

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Precision:
        unknown = set(value) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown precision fields: {sorted(unknown)}")
        return cls(**value)  # type: ignore[arg-type]

    def arguments(self) -> list[str]:
        """The flags that reproduce this request through ``add_arguments``."""
        result = []
        for name in FIELDS:
            value = getattr(self, name)
            flag = "--" + name.replace("_", "-")
            if name == "fuse_narrow_gradient_accumulation":
                result.append(
                    flag if value else "--no-fuse-narrow-gradient-accumulation"
                )
            else:
                result.extend([flag, "none" if value is None else str(value)])
        return result


@dataclass(frozen=True)
class Capabilities:
    """What one engine can honor; ``None`` in a tuple means the system
    default is among the choices."""

    engine: str
    model_dtypes: tuple[str, ...] = DTYPES
    master_dtypes: tuple[str | None, ...] = (None, *DTYPES)
    grad_dtypes: tuple[str | None, ...] = (None, *DTYPES)
    opt_state_dtypes: tuple[str | None, ...] = (None, "parameter", *DTYPES)
    parameter_roundings: tuple[str | None, ...] = (None, *ROUNDINGS)
    opt_state_roundings: tuple[str | None, ...] = (None, *ROUNDINGS)
    fuse_narrow_gradient_accumulation: bool = True

    def check(self, precision: Precision) -> None:
        """Refuse a request this engine cannot run as asked."""

        for field, allowed in (
            ("model_dtype", self.model_dtypes),
            ("master_dtype", self.master_dtypes),
            ("grad_dtype", self.grad_dtypes),
            ("opt_state_dtype", self.opt_state_dtypes),
            ("parameter_rounding", self.parameter_roundings),
            ("opt_state_rounding", self.opt_state_roundings),
        ):
            value = getattr(precision, field)
            if value not in allowed:
                choices = ", ".join(
                    "none" if item is None else item for item in allowed
                )
                raise Unsupported(
                    f"{self.engine} cannot honor {field}={value or 'none'};"
                    f" it offers {choices}"
                )
        if (
            precision.fuse_narrow_gradient_accumulation
            and precision.narrow_gradients
            and not self.fuse_narrow_gradient_accumulation
        ):
            gradients = precision.grad_dtype or precision.model_dtype
            raise Unsupported(
                f"{self.engine} cannot have a multiply add a weight gradient"
                f" into a {gradients} running gradient as it writes it;"
                " --no-fuse-narrow-gradient-accumulation adds it after"
            )


def _dtype_name(value: str) -> str | None:
    if value == "none":
        return None
    if value.startswith("torch."):
        value = value[len("torch.") :]
    if value not in DTYPES:
        raise argparse.ArgumentTypeError(f"expected one of {DTYPES} or none")
    return value


def add_arguments(
    parser: argparse.ArgumentParser, defaults: Precision | None = None
) -> None:
    """The quickstart's precision flags, with the baselines' defaults."""

    defaults = defaults or Precision()
    parser.add_argument(
        "--model-dtype",
        choices=DTYPES,
        default=defaults.model_dtype,
        help=f"model weights and activations (default: {defaults.model_dtype})",
    )
    parser.add_argument(
        "--master-dtype",
        type=_dtype_name,
        default=defaults.master_dtype,
        metavar="DTYPE",
        help="keep a master copy of every trained weight at this dtype and step"
        " it in the weight's place; none steps the weights themselves"
        f" (default: {defaults.master_dtype or 'none'})",
    )
    parser.add_argument(
        "--grad-dtype",
        type=_dtype_name,
        default=defaults.grad_dtype,
        metavar="DTYPE",
        help="the dtype gradients are summed at over a step's microbatches;"
        " none for the weights' own"
        f" (default: {defaults.grad_dtype or 'none'})",
    )
    parser.add_argument(
        "--opt-state-dtype",
        type=lambda value: None if value == "none" else value,
        choices=(None, "parameter", *DTYPES),
        default=defaults.opt_state_dtype,
        metavar="DTYPE",
        help="the dtype the optimizer keeps its state at, parameter for the dtype"
        " of what it steps, none for its own default"
        f" (default: {defaults.opt_state_dtype or 'none'})",
    )
    parser.add_argument(
        "--parameter-rounding",
        type=lambda value: None if value == "none" else value,
        choices=(None, *ROUNDINGS),
        default=defaults.parameter_rounding,
        metavar="ROUNDING",
        help="how the optimizer rounds the weights it steps: nearest or stochastic;"
        " none for its own default",
    )
    parser.add_argument(
        "--opt-state-rounding",
        type=lambda value: None if value == "none" else value,
        choices=(None, *ROUNDINGS),
        default=defaults.opt_state_rounding,
        metavar="ROUNDING",
        help="how the optimizer rounds the state it stores: nearest or stochastic;"
        " none for its own default",
    )
    parser.add_argument(
        "--fuse-narrow-gradient-accumulation",
        action=argparse.BooleanOptionalAction,
        default=defaults.fuse_narrow_gradient_accumulation,
        help="with gradients narrower than float32, let a later microbatch's"
        " multiply add its weight gradient into the running gradient as it writes"
        " it instead of writing it at the narrow dtype and adding it after; no"
        " effect on float32 gradients"
        f" (default: {'on' if defaults.fuse_narrow_gradient_accumulation else 'off'})",
    )


def with_changes(precision: Precision, **changes: object) -> Precision:
    unknown = set(changes) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown precision fields: {sorted(unknown)}")
    return replace(precision, **changes)  # type: ignore[arg-type]


__all__ = [
    "DTYPES",
    "FIELDS",
    "ROUNDINGS",
    "Capabilities",
    "Precision",
    "Unsupported",
    "add_arguments",
    "with_changes",
]
