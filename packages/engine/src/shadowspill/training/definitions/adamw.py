"""Installed settings for the existing MLOps AdamW, with no second updater."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from hashlib import sha256
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shadowspill_client import Optimizer
from shadowspill_client._records import Value
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.definitions import DefinitionSource

from .optimizer import OptimizerDefinition

if TYPE_CHECKING:
    from .registry import Registry


@dataclass(frozen=True)
class AdamWConfig:
    gradient_dtype: str = "bfloat16"
    opt_state_dtype: str = "bfloat16"
    parameter_rounding: str = "nearest"
    opt_state_rounding: str = "nearest"
    maximize: bool = False

    def __post_init__(self) -> None:
        for name in ("gradient_dtype", "opt_state_dtype"):
            if getattr(self, name) not in (
                "parameter",
                "float32",
                "float16",
                "bfloat16",
            ):
                raise ValueError(f"unsupported AdamW {name}")
        for name in ("parameter_rounding", "opt_state_rounding"):
            if getattr(self, name) not in ("nearest", "stochastic"):
                raise ValueError(f"unsupported AdamW {name}")
        if type(self.maximize) is not bool:
            raise TypeError("AdamW maximize must be boolean")


def configure(settings: Optimizer) -> Mapping[str, Value]:
    if settings.groups:
        raise ValueError("mlops/adamw@1 currently requires one parameter group")
    precision = {field.name for field in fields(AdamWConfig)} - {"maximize"}
    if unknown := settings.precision.keys() - precision:
        raise ValueError(f"unsupported AdamW precision: {sorted(unknown)}")
    configured = dict(settings.config)
    for name, value in settings.precision.items():
        if name in configured and configured[name] != value:
            raise ValueError(f"conflicting AdamW precision: {name}")
        configured[name] = value
    return configured


def create(model: Any, config: AdamWConfig, controls: Mapping[str, Any]) -> Any:
    import torch
    from mlops.optim import AdamW

    def dtype(name: str) -> Any:
        return name if name == "parameter" else getattr(torch, name)

    return AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=controls["lr"],
        betas=(controls["beta1"], controls["beta2"]),
        eps=controls["eps"],
        weight_decay=controls["weight_decay"],
        gradient_dtype=dtype(config.gradient_dtype),
        opt_state_dtype=dtype(config.opt_state_dtype),
        parameter_rounding=config.parameter_rounding,
        opt_state_rounding=config.opt_state_rounding,
        maximize=config.maximize,
    )


def register(registry: Registry) -> None:
    """Opt in with the engine's installed ``shadowspill.definitions`` hook.

    Locate and hash the optional provider without importing its framework code.
    Its Python closure includes dtype/rounding helpers and the opaque kernel's
    allocation behavior, so editing that dependency invalidates saved settings.
    External definition packages do not need to load this optional integration.
    """
    spec = find_spec("mlops")
    if spec is None or spec.origin is None:
        raise ValueError("mlops/adamw@1 requires the installed MLOps provider")
    root = Path(spec.origin).resolve().parent
    digest = sha256(Path(__file__).read_bytes())
    digest.update(version("mlops").encode())
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    controls = ControlSchema(
        {
            "lr": ScalarControl(0.001, minimum=0),
            "beta1": ScalarControl(0.9, minimum=0, maximum=1, maximum_inclusive=False),
            "beta2": ScalarControl(
                0.999, minimum=0, maximum=1, maximum_inclusive=False
            ),
            "eps": ScalarControl(1e-8, minimum=0),
            "weight_decay": ScalarControl(0.01, minimum=0),
        }
    )
    registry.add_optimizer(
        OptimizerDefinition(
            "mlops/adamw@1",
            create,
            controls,
            {
                name: ("optimizer_group.0." + binding,)
                for name, binding in (
                    ("lr", "lr"),
                    ("beta1", "betas.0"),
                    ("beta2", "betas.1"),
                    ("eps", "eps"),
                    ("weight_decay", "weight_decay"),
                )
            },
            config=AdamWConfig,
            configure=configure,
            # These exact provider forms allocate only fixed-geometry outputs;
            # the out/in-place kernels use preallocated tensors and no scratch.
            # Scalar values (including zero LR) cannot change that geometry.
            allocation_invariants={
                f"torch.ops.mlops.{name}.default": "mlops/adamw-fixed-allocation@1"
                for name in (
                    "adamw",
                    "adamw_out",
                    "adamw_",
                    "master_adamw",
                    "master_adamw_out",
                    "master_adamw_",
                )
            },
        ),
        source=DefinitionSource(
            "shadowspill", version("shadowspill"), digest.hexdigest()
        ),
    )
