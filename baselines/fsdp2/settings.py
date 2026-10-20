"""The settings of one FSDP2 case, shared by the trainer and the sweep.

A case names the catalog model and its mode (``baselines.common.models``),
the geometry of an update (``baselines.common.geometry``), the precision
contract (``baselines.common.precision``), the data, and this harness's own
knobs: whole-block recomputation; activation offloading through PyTorch's
``offload_wrapper`` or the stream-overlapped offloader vendored from
torchtune; FSDP2's CPU offload of parameters, gradients and optimizer
state; prefetch depths, resharding and gradient-synchronization policies;
compilation; the allocator cap; and the torchao scenarios, Float8 training,
the low-bit and CPU-offloaded AdamW optimizers and NF4-quantized frozen
weights for LoRA. What the harness cannot honor is refused before launch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

from baselines.common import geometry as geometry_flags
from baselines.common import models as model_flags
from baselines.common import precision as precision_flags
from baselines.common.geometry import TIER_TOKENS_PER_STEP, Geometry, recompute_layers
from baselines.common.models import MODES, ModelRequest
from baselines.common.precision import Capabilities, Precision, Unsupported

#: The sources whose digests identify this harness in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/fsdp2",
    "workloads/models",
    "workloads/lora",
    "workloads/full_model.py",
)

#: The fields a row records as the model, the geometry and the precision;
#: every other field is the configuration.
MODEL_FIELDS = ("model", "implementation", "mode", "lora", "model_config")
GEOMETRY_FIELDS = (
    "sequence_length",
    "tokens_per_step",
    "tokens_per_microbatch",
    "tier",
)
PRECISION_FIELDS = precision_flags.FIELDS

ACTIVATION_OFFLOADS = ("none", "cpu", "overlapped")
OFFLOADS = ("none", "cpu")
GRADIENT_SYNCS = ("per-microbatch", "last-microbatch")
OPTIMIZERS = ("adamw", "adamw8bit", "adamw4bit", "cpu-offload")

#: What this harness honors of the precision contract. FSDP2 keeps a
#: trained weight's shard at the dtype the weight was materialized at, the
#: master when one is kept, sums gradients into that shard and keeps AdamW's
#: moments at it, so the gradient and state dtypes must be that dtype; a
#: float16 model would need loss scaling the harness does not do; stochastic
#: rounding comes from torchao's AdamW and applies to bfloat16 weights.
CAPABILITIES = Capabilities(
    "fsdp2",
    model_dtypes=("bfloat16", "float32"),
    master_dtypes=(None, "float32"),
    grad_dtypes=(None, "float32", "bfloat16"),
    opt_state_dtypes=(None, "parameter", "float32", "bfloat16"),
    parameter_roundings=(None, "stochastic"),
    opt_state_roundings=(None,),
    round_accumulation_once=False,
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
    tier: str | None = None
    # The precision contract.
    model_dtype: str = "bfloat16"
    master_dtype: str | None = "float32"
    grad_dtype: str | None = "float32"
    opt_state_dtype: str | None = "float32"
    parameter_rounding: str | None = None
    opt_state_rounding: str | None = None
    round_accumulation_once: bool = False
    # The data and the weights.
    data: str | None = None
    checkpoint: str | None = None
    seed: int = 0
    # Memory mechanisms.
    recompute: float = 1.0
    activation_offload: str = "none"
    offload: str = "none"
    pin_memory: bool = True
    forward_prefetch: int = 0
    backward_prefetch: int | str = "auto"
    gradient_sync: str = "per-microbatch"
    reshard_after_forward: bool = True
    reshard_after_backward: bool = True
    compile_blocks: bool = False
    allocator_cap_gib: float | None = None
    # torchao scenarios.
    float8: bool = False
    optimizer: str = "adamw"
    qlora_nf4: bool = False
    # The run.
    warmup_steps: int = 3
    steps: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    cpu_threads: int = 1
    verify_reference: bool = False
    nsys_capture: bool = False

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown FSDP2 settings: {sorted(unknown)}")
        value = dict(value)
        if "backward_prefetch" in value:
            value["backward_prefetch"] = backward_prefetch(value["backward_prefetch"])
        for name in ("lora", "model_config"):
            if name in value:
                if isinstance(value[name], str):
                    value[name] = json.loads(value[name])
                if not isinstance(value[name], dict):
                    raise ValueError(f"{name} must be a mapping")
        return cls(**value)

    def request(self) -> ModelRequest:
        if self.model is None:
            raise ValueError("a case names its catalog model")
        return ModelRequest(
            self.model, self.implementation, self.mode, self.lora, self.model_config
        )

    def geometry(self, world_size) -> Geometry:
        tokens_per_step = self.tokens_per_step
        if self.tier is not None:
            if self.tier not in TIER_TOKENS_PER_STEP:
                raise ValueError(f"unknown tier {self.tier!r}")
            tier_tokens = TIER_TOKENS_PER_STEP[self.tier]
            if tokens_per_step is None:
                tokens_per_step = tier_tokens
            elif tokens_per_step != tier_tokens:
                raise ValueError(
                    f"tokens_per_step {tokens_per_step} contradicts tier {self.tier}"
                )
        if tokens_per_step is None:
            raise ValueError("a case names tokens_per_step or a tier")
        return Geometry(
            self.sequence_length,
            tokens_per_step,
            self.tokens_per_microbatch,
            world_size,
        )

    def precision(self) -> Precision:
        return Precision(**{name: getattr(self, name) for name in PRECISION_FIELDS})

    def validate(self, world_size):
        """Refuse what cannot run as asked; returns the geometry."""

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
        request = self.request()
        geometry = self.geometry(world_size)
        precision = self.precision()
        CAPABILITIES.check(precision)
        stepped = precision.stepped_dtype
        if (precision.grad_dtype or precision.model_dtype) != stepped:
            raise Unsupported(
                "fsdp2 sums gradients into the shard of the weight it steps, so"
                f" grad_dtype must be {stepped} here"
            )
        if precision.opt_state_dtype not in (None, "parameter", stepped):
            raise Unsupported(
                f"fsdp2 keeps AdamW's moments at the stepped dtype ({stepped})"
            )
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.optimizer in ("adamw8bit", "adamw4bit") and (
            precision.opt_state_dtype is not None
        ):
            raise Unsupported(
                f"{self.optimizer} keeps its state in low-bit blocks; leave"
                " opt_state_dtype none"
            )
        if precision.parameter_rounding == "stochastic" and stepped != "bfloat16":
            raise Unsupported("stochastic rounding applies to bfloat16 weights")
        layers = int(request.entry.config(**self.model_config).n_layers)
        recompute_layers(layers, self.recompute)
        if self.activation_offload not in ACTIVATION_OFFLOADS:
            raise ValueError(f"activation_offload must be one of {ACTIVATION_OFFLOADS}")
        if self.offload not in OFFLOADS:
            raise ValueError(f"offload must be one of {OFFLOADS}")
        if self.gradient_sync not in GRADIENT_SYNCS:
            raise ValueError(f"gradient_sync must be one of {GRADIENT_SYNCS}")
        if self.forward_prefetch < 0:
            raise ValueError("forward_prefetch must be nonnegative")
        backward_prefetch(self.backward_prefetch)
        if self.float8 and precision.model_dtype != "bfloat16":
            raise Unsupported("Float8 training computes on bfloat16 weights")
        if self.float8 and self.mode != "full":
            raise Unsupported("Float8 training converts the linears LoRA adapts")
        if self.qlora_nf4 and self.mode != "lora":
            raise ValueError("qlora_nf4 quantizes the frozen weights of LoRA")
        if self.float8 and self.qlora_nf4:
            raise ValueError("float8 and qlora_nf4 are exclusive")
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
        if self.verify_reference:
            if request.entry.size != "tiny":
                raise ValueError("the reference check materializes a tiny model")
            if (
                self.optimizer != "adamw"
                or self.float8
                or self.qlora_nf4
                or precision.parameter_rounding is not None
            ):
                raise ValueError(
                    "the reference check compares plain AdamW without Float8,"
                    " NF4 or stochastic rounding"
                )
        return geometry

    def normalized(self, world_size):
        """These settings with the tier resolved into tokens_per_step; settings
        that cannot run at this world size are refused."""
        geometry = self.validate(world_size)
        return dataclasses.replace(
            self, tokens_per_step=geometry.tokens_per_step, tier=None
        )

    def describe(self, world_size):
        """The model, geometry, precision and configuration a row records."""
        configuration = {
            name: value
            for name, value in dataclasses.asdict(self).items()
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
        for name, value in dataclasses.asdict(self).items():
            flag = "--" + name.replace("_", "-")
            if name in ("lora", "model_config"):
                result.extend([flag, json.dumps(value, sort_keys=True)])
            elif name == "round_accumulation_once":
                result.append(flag if value else "--no-round-accumulation-once")
            elif name in PRECISION_FIELDS:
                result.extend([flag, "none" if value is None else str(value)])
            elif isinstance(value, bool):
                result.append(f"--{'' if value else 'no-'}{flag[2:]}")
            elif value is not None:
                result.extend([flag, str(value)])
        return result


HELP = {
    "data": "a corpus directory prepare_corpus wrote; synthetic tokens otherwise",
    "checkpoint": "a local Hugging Face checkpoint of the model, read through"
    " the family's checkpoint.py; initialized weights otherwise",
    "seed": "synthetic data and initialization seed",
    "recompute": "fraction of whole transformer blocks checkpointed, from 0 to 1",
    "activation_offload": "none; cpu, PyTorch's offload_wrapper per block;"
    " overlapped, the stream-overlapped offloader vendored from torchtune",
    "offload": "FSDP2 CPU offload of parameters, gradients and optimizer state",
    "pin_memory": "pin CPU-offloaded shards (with --offload cpu)",
    "forward_prefetch": "explicit lookahead depth; 0 is native forward behavior",
    "backward_prefetch": "native auto prefetch or an explicit depth of at least 1",
    "gradient_sync": "reduce every microbatch, or keep full gradients until the last",
    "reshard_after_forward": "release each block's gathered weights after forward",
    "reshard_after_backward": "reshard between accumulation microbatches",
    "compile_blocks": "torch.compile each block",
    "allocator_cap_gib": "allocator limit; excludes NCCL and external allocations",
    "float8": "torchao Float8 training of every block's linears (tensorwise)",
    "optimizer": "AdamW: fused torch, torchao 8-bit or 4-bit, or torchao's"
    " CPU-offloaded",
    "qlora_nf4": "quantize the frozen weights of LoRA targets to NF4",
    "cpu_threads": "intra-op threads per rank, including CPU AdamW",
    "verify_reference": "tiny model: compare updates to an eager reference",
    "nsys_capture": "capture measured steps; not a throughput result",
}


def add_arguments(parser):
    """The trainer's flags: the common model, geometry and precision flags,
    then this harness's own."""

    model_flags.add_arguments(parser)
    geometry_flags.add_arguments(parser)
    precision_flags.add_arguments(parser)
    defaults = dataclasses.asdict(Settings())
    for name, value in defaults.items():
        if name in MODEL_FIELDS + GEOMETRY_FIELDS + PRECISION_FIELDS:
            continue
        options = {"default": value, "help": HELP.get(name)}
        if isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        elif name == "backward_prefetch":
            options["type"] = backward_prefetch
        elif name in ("allocator_cap_gib",):
            options["type"] = float
        elif name in ("data", "checkpoint"):
            options["type"] = str
        else:
            options["type"] = type(value)
        choices = {
            "activation_offload": ACTIVATION_OFFLOADS,
            "offload": OFFLOADS,
            "gradient_sync": GRADIENT_SYNCS,
            "optimizer": OPTIMIZERS,
        }
        if name in choices:
            options["choices"] = choices[name]
        parser.add_argument(f"--{name.replace('_', '-')}", **options)


def from_namespace(arguments) -> Settings:
    """The settings a parsed trainer command line describes."""
    fields = {field.name for field in dataclasses.fields(Settings)}
    return Settings.from_dict(
        {key: value for key, value in vars(arguments).items() if key in fields}
    )


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


__all__ = [
    "ACTIVATION_OFFLOADS",
    "CAPABILITIES",
    "GRADIENT_SYNCS",
    "MODES",
    "OFFLOADS",
    "OPTIMIZERS",
    "SOURCE_PATHS",
    "Settings",
    "add_arguments",
    "backward_prefetch",
    "command",
    "configure_prefetch",
    "from_namespace",
]
