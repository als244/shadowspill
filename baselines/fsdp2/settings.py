"""Serializable FSDP2 case settings, shared by the trainer and the sweep."""

from __future__ import annotations

import argparse
import dataclasses
import math

from baselines.common.geometry import Geometry, accumulation_steps, recompute_layers
from baselines.common.precision import Precision

#: The sources whose digests identify this harness in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/fsdp2",
    "workloads/models",
)

#: What the harness builds, by its tiny switch.
MODELS = {False: "qwen3moe-30b-a3b", True: "qwen3moe-tiny"}

#: The settings a row records as geometry rather than configuration.
GEOMETRY_FIELDS = (
    "sequence_length",
    "global_tokens",
    "microbatch_tokens",
    "gradient_accumulation",
)


def backward_prefetch(value):
    if value == "auto":
        return value
    if type(value) not in (str, int):
        raise ValueError("backward prefetch must be 'auto' or a positive integer")
    try:
        depth = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "backward prefetch must be 'auto' or a positive integer"
        ) from error
    if depth < 1:
        raise ValueError("backward prefetch must be 'auto' or a positive integer")
    return depth


@dataclasses.dataclass(frozen=True)
class Settings:
    sequence_length: int = 4096
    global_tokens: int = 2097152
    microbatch_tokens: int = 4096
    gradient_accumulation: int | None = None
    recompute: float = 1.0
    offload: str = "none"
    pin_memory: bool = True
    forward_prefetch: int = 0
    backward_prefetch: int | str = "auto"
    gradient_sync: str = "per-microbatch"
    reshard_after_forward: bool = True
    reshard_after_backward: bool = True
    compile_blocks: bool = False
    allocator_cap_gib: float | None = None
    warmup_steps: int = 3
    steps: int = 5
    seed: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    cpu_threads: int = 1
    tiny: bool = False
    verify_reference: bool = False
    nsys_capture: bool = False

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown FSDP2 settings: {sorted(unknown)}")
        if "backward_prefetch" in value:
            value = {
                **value,
                "backward_prefetch": backward_prefetch(value["backward_prefetch"]),
            }
        return cls(**value)

    def validate(self, world_size):
        # Reject JSON strings/bools where numerical values were requested.
        defaults = dataclasses.asdict(Settings())
        for key, value in dataclasses.asdict(self).items():
            default = defaults[key]
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            if type(default) is int and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            if type(default) is float and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
        accumulation = accumulation_steps(
            self.global_tokens, self.sequence_length, self.microbatch_tokens, world_size
        )
        if self.gradient_accumulation is not None and (
            type(self.gradient_accumulation) is not int
            or self.gradient_accumulation != accumulation
        ):
            raise ValueError(
                f"gradient_accumulation must be {accumulation}: "
                "global_tokens = world_size * microbatch_tokens * accumulation"
            )
        recompute_layers(2 if self.tiny else 48, self.recompute)
        if self.offload not in ("none", "cpu"):
            raise ValueError("offload must be 'none' or 'cpu'")
        if self.gradient_sync not in ("per-microbatch", "last-microbatch"):
            raise ValueError("gradient_sync must be per-microbatch or last-microbatch")
        if self.forward_prefetch < 0:
            raise ValueError("forward_prefetch must be nonnegative")
        if type(self.backward_prefetch) not in (str, int):
            raise ValueError("backward_prefetch must be 'auto' or a positive integer")
        backward_prefetch(self.backward_prefetch)
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
        if self.verify_reference and not self.tiny:
            raise ValueError("reference validation requires tiny=true")
        return accumulation

    def normalized(self, world_size):
        """These settings with the derived accumulation filled in; settings
        that cannot run at this world size are refused."""
        return dataclasses.replace(
            self, gradient_accumulation=self.validate(world_size)
        )

    def describe(self, world_size):
        """The model, geometry, precision and configuration a row records."""
        geometry = Geometry(
            self.sequence_length,
            self.global_tokens,
            self.microbatch_tokens,
            world_size,
        )
        configuration = {
            name: value
            for name, value in dataclasses.asdict(self).items()
            if name not in GEOMETRY_FIELDS
        }
        return {
            "model": {
                "model": MODELS[self.tiny],
                "implementation": "mlops",
                "mode": "full",
            },
            "geometry": geometry.as_dict(),
            "precision": Precision().as_dict(),
            "configuration": configuration,
        }

    def arguments(self):
        result = []
        for name, value in dataclasses.asdict(self).items():
            flag = name.replace("_", "-")
            if isinstance(value, bool):
                result.append(f"--{'' if value else 'no-'}{flag}")
            elif value is not None:
                result.extend([f"--{flag}", str(value)])
        return result


def command(python, world_size, settings, output):
    """The torchrun command line of one case."""
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "--max-restarts=0",
        "-m",
        "baselines.fsdp2.train",
        "--output",
        str(output),
        *settings.arguments(),
    ]


def add_arguments(parser):
    help_text = {
        "gradient_accumulation": "Microbatches per update; derived unless specified",
        "recompute": "Fraction of whole transformer blocks checkpointed, from 0 to 1",
        "offload": "CPU offload of parameters, gradients, and optimizer state",
        "pin_memory": "Pin CPU-offloaded shards (only used with --offload cpu)",
        "forward_prefetch": "Explicit lookahead depth; 0 = native forward behavior",
        "backward_prefetch": "Native auto prefetch or explicit depth >=1",
        "gradient_sync": "Reduce every microbatch, or retain full grads until last",
        "reshard_after_backward": "Reshard between accumulation microbatches",
        "allocator_cap_gib": "Allocator limit; excludes NCCL/external allocations",
        "cpu_threads": "Intra-op threads per rank, including CPU AdamW",
        "verify_reference": "Tiny model: compare updates to an eager reference",
        "nsys_capture": "Capture measured steps; not a throughput result",
    }
    defaults = dataclasses.asdict(Settings())
    for name, value in defaults.items():
        options = {"default": value, "help": help_text.get(name)}
        if isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        elif name == "backward_prefetch":
            options["type"] = backward_prefetch
        elif name == "allocator_cap_gib":
            options["type"] = float
        elif name == "gradient_accumulation":
            options["type"] = int
        else:
            options["type"] = type(value)
        if name == "offload":
            options["choices"] = ("none", "cpu")
        if name == "gradient_sync":
            options["choices"] = ("per-microbatch", "last-microbatch")
        parser.add_argument(f"--{name.replace('_', '-')}", **options)


def configure_prefetch(model, *, forward, backward):
    """Use public FSDP2 APIs. Root owns embedding/head; blocks execute in order."""
    blocks = list(model.model.blocks)
    model.set_modules_to_forward_prefetch(blocks[:forward])
    for index, block in enumerate(blocks):
        block.set_modules_to_forward_prefetch(blocks[index + 1 : index + 1 + forward])
    depth = backward_prefetch(backward)
    if depth != "auto":
        model.set_modules_to_backward_prefetch(list(reversed(blocks))[:depth])
        for index, block in enumerate(blocks):
            block.set_modules_to_backward_prefetch(
                list(reversed(blocks[:index]))[:depth]
            )
