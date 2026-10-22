"""What every engine's settings share: the catalog model and its mode, the
geometry of an update, the precision contract, the data and the weights,
and the run; the validation of all of that; and the forms a case takes, a
mapping, a command line and a description for its row.

An engine subclasses ``Settings`` with its own fields, its ``CAPABILITIES``
of the precision contract and its ``engine_validation``; the sweep and the
trainer see one interface.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from baselines.common import geometry as geometry_flags
from baselines.common import models as model_flags
from baselines.common import precision as precision_flags
from baselines.common.geometry import Geometry, recompute_layers
from baselines.common.models import ModelRequest
from baselines.common.precision import Capabilities, Precision

#: The fields a row records as the model, the geometry and the precision;
#: every other field is the configuration.
MODEL_FIELDS = ("model", "implementation", "mode", "lora", "model_config")
GEOMETRY_FIELDS = ("sequence_length", "tokens_per_step", "tokens_per_microbatch")
PRECISION_FIELDS = precision_flags.FIELDS

HELP = {
    "data": "a corpus directory prepare_corpus wrote; synthetic tokens otherwise",
    "checkpoint": "a local Hugging Face checkpoint of the model, read through"
    " the family's checkpoint.py; initialized weights otherwise",
    "seed": "synthetic data and initialization seed",
    "recompute": "fraction of whole transformer blocks checkpointed, from 0 to 1;"
    " none where the system decides for itself",
    "allocator_cap_gib": "allocator limit; excludes NCCL and external allocations",
    "cpu_threads": "intra-op threads per rank, including CPU optimizer work",
    "verify_reference": "tiny model: compare updates to a reference copy compiled"
    " whole",
    "nsys_capture": "capture measured steps; not a throughput result",
}


@dataclasses.dataclass(frozen=True)
class Settings:
    """The fields every engine takes; an engine adds its own after them."""

    #: What the engine honors of the precision contract.
    CAPABILITIES: Capabilities = dataclasses.field(
        default=Capabilities("engine"), init=False, repr=False, compare=False
    )

    # The model and its training mode.
    model: str | None = None
    implementation: str = "mlops"
    mode: str = "full"
    lora: dict = dataclasses.field(default_factory=dict)
    model_config: dict = dataclasses.field(default_factory=dict)
    # The geometry of one update.
    sequence_length: int = geometry_flags.SEQUENCE_LENGTH
    tokens_per_step: int | None = None
    tokens_per_microbatch: int = 4096
    # The precision contract.
    model_dtype: str = "bfloat16"
    master_dtype: str | None = "float32"
    grad_dtype: str | None = None
    opt_state_dtype: str | None = "float32"
    parameter_rounding: str | None = None
    opt_state_rounding: str | None = None
    fuse_narrow_gradient_accumulation: bool = True
    # The data and the weights.
    data: str | None = None
    checkpoint: str | None = None
    seed: int = 0
    # Recomputation and the allocator.
    recompute: float = 1.0
    allocator_cap_gib: float | None = None
    # The run.
    warmup_steps: int = 3
    steps: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    cpu_threads: int = 1
    verify_reference: bool = False
    nsys_capture: bool = False

    @classmethod
    def field_names(cls) -> set[str]:
        return {field.name for field in dataclasses.fields(cls) if field.init}

    @classmethod
    def coerce(cls, value: dict) -> dict:
        """An engine's own normalization of a mapping's values."""
        return value

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - cls.field_names()
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
        value = dict(value)
        for name in ("lora", "model_config"):
            if name in value:
                if isinstance(value[name], str):
                    value[name] = json.loads(value[name])
                if not isinstance(value[name], dict):
                    raise ValueError(f"{name} must be a mapping")
        return cls(**cls.coerce(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.init
        }

    def request(self) -> ModelRequest:
        if self.model is None:
            raise ValueError("a case names its catalog model")
        return ModelRequest(
            self.model, self.implementation, self.mode, self.lora, self.model_config
        )

    def geometry(self, world_size) -> Geometry:
        if self.tokens_per_step is None:
            raise ValueError("a case names its tokens_per_step")
        return Geometry(
            self.sequence_length,
            self.tokens_per_step,
            self.tokens_per_microbatch,
            world_size,
        )

    def precision(self) -> Precision:
        return Precision(**{name: getattr(self, name) for name in PRECISION_FIELDS})

    def layers(self, request: ModelRequest) -> int:
        return int(request.entry.config(**self.model_config).n_layers)

    def validate(self, world_size):
        """Refuse what cannot run as asked; returns the geometry."""

        defaults = type(self)().as_dict()
        for key, value in self.as_dict().items():
            default = defaults[key]
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            if type(default) is int and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            if type(default) is float and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
        request = self.request()
        geometry = self.geometry(world_size)
        precision = self.precision()
        self.CAPABILITIES.check(precision)
        if self.recompute is not None:
            recompute_layers(self.layers(request), self.recompute)
        if self.data is not None and not (Path(self.data) / "meta.json").is_file():
            raise ValueError(f"{self.data} is not a corpus directory")
        if self.checkpoint is not None:
            if not (Path(self.checkpoint) / "config.json").is_file():
                raise ValueError(f"{self.checkpoint} is not a checkpoint directory")
            request.checkpoint(self.checkpoint).close()
            if self.verify_reference:
                raise ValueError("the reference check starts from an initialization")
        if min(self.steps, self.warmup_steps, self.cpu_threads) < 1:
            raise ValueError(
                "warmup, measured steps, and CPU thread counts must be positive"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning rate must be positive and weight decay nonnegative"
            )
        if self.allocator_cap_gib is not None and (
            type(self.allocator_cap_gib) not in (int, float)
            or not math.isfinite(self.allocator_cap_gib)
            or self.allocator_cap_gib <= 0
        ):
            raise ValueError("allocator_cap_gib must be a positive finite number")
        if self.verify_reference and request.entry.size != "tiny":
            raise ValueError("the reference check materializes a tiny model")
        self.engine_validation(request, geometry, precision)
        return geometry

    def engine_validation(self, request, geometry, precision) -> None:
        """An engine's own rules, after the shared ones."""

    def resolve(self, world_size) -> dict[str, Any]:
        """What an engine's trainer needs of the catalog beyond what every
        case resolves (``baselines.common.launch``), for a trainer whose
        interpreter cannot import it."""
        return {}

    def environment(self) -> dict[str, str | None]:
        """The variables a library reads only from the process environment,
        as this case sets them (``None``: unset), from the case's own
        settings: the launcher sets them for the trainer, and the trainer
        refuses to run in another environment."""
        return {}

    def describe(self, world_size):
        """The model, geometry, precision and configuration a row records."""
        configuration = {
            name: value
            for name, value in self.as_dict().items()
            if name not in MODEL_FIELDS + GEOMETRY_FIELDS + PRECISION_FIELDS
        }
        return {
            "model": self.request().as_dict(),
            "geometry": self.geometry(world_size).as_dict(),
            "precision": self.precision().as_dict(),
            "configuration": configuration,
        }

    def arguments(self):
        """The trainer's command line for these settings."""
        result = []
        for name, value in self.as_dict().items():
            flag = "--" + name.replace("_", "-")
            if name in ("lora", "model_config"):
                result.extend([flag, json.dumps(value, sort_keys=True)])
            elif name == "fuse_narrow_gradient_accumulation":
                result.append(
                    flag if value else "--no-fuse-narrow-gradient-accumulation"
                )
            elif name in PRECISION_FIELDS:
                result.extend([flag, "none" if value is None else str(value)])
            elif isinstance(value, bool):
                result.append(f"--{'' if value else 'no-'}{flag[2:]}")
            elif value is not None:
                result.extend([flag, str(value)])
        return result


def add_arguments(
    parser: argparse.ArgumentParser,
    settings_type: type[Settings],
    *,
    help: dict[str, str] | None = None,
    choices: dict[str, tuple] | None = None,
    types: dict[str, Any] | None = None,
    booleans: frozenset[str] = frozenset(),
    implementations: tuple[str, ...] | None = None,
) -> None:
    """The trainer's flags for an engine's settings: the common model,
    geometry and precision flags, then every other field from its default,
    with the help, choices and types the engine gives; ``booleans`` are the
    fields that are a yes or a no once set and unset (none) by default;
    ``implementations`` replace the catalog's for an engine that trains
    another system's."""

    model_flags.add_arguments(parser, implementations)
    geometry_flags.add_arguments(
        parser, microbatch_required=settings_type().tokens_per_microbatch is not None
    )
    precision_flags.add_arguments(parser, settings_type().precision())
    help = {**HELP, **(help or {})}
    choices = choices or {}
    types = types or {}
    for name, value in settings_type().as_dict().items():
        if name in MODEL_FIELDS + GEOMETRY_FIELDS + PRECISION_FIELDS:
            continue
        options: dict[str, Any] = {"default": value, "help": help.get(name)}
        if name in types:
            options["type"] = types[name]
        elif isinstance(value, bool) or name in booleans:
            options["action"] = argparse.BooleanOptionalAction
        elif name == "allocator_cap_gib":
            options["type"] = float
        elif value is None:
            options["type"] = str
        else:
            options["type"] = type(value)
        if name in choices:
            options["choices"] = choices[name]
        parser.add_argument(f"--{name.replace('_', '-')}", **options)


def from_namespace(settings_type: type[Settings], arguments) -> Settings:
    """The settings a parsed trainer command line describes."""
    names = settings_type.field_names()
    return settings_type.from_dict(
        {key: value for key, value in vars(arguments).items() if key in names}
    )


__all__ = [
    "GEOMETRY_FIELDS",
    "HELP",
    "MODEL_FIELDS",
    "PRECISION_FIELDS",
    "Settings",
    "add_arguments",
    "from_namespace",
]
