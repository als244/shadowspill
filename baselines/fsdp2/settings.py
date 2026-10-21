"""The settings of one FSDP2 case, shared by the trainer and the sweep.

On top of what every engine takes (``baselines.common.settings``), this
engine's own knobs: activation offloading through PyTorch's
``offload_wrapper`` or the stream-overlapped offloader vendored from
torchtune; FSDP2's CPU offload of parameters, gradients and optimizer state;
prefetch depths, resharding and gradient-synchronization policies;
compilation; and the torchao scenarios, Float8 training and the low-bit and
CPU-offloaded AdamW optimizers. What the engine cannot honor is refused
before launch.
"""

from __future__ import annotations

import dataclasses

from baselines.common import settings as common
from baselines.common.precision import Capabilities, Unsupported
from baselines.common.settings import GEOMETRY_FIELDS, MODEL_FIELDS, PRECISION_FIELDS

#: The sources whose digests identify this engine in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/fsdp2",
    "workloads/models",
    "workloads/lora",
    "workloads/full_model.py",
)

#: The environment beside the launcher's that the trainer runs in.
ENVIRONMENT = "baselines"

ACTIVATION_OFFLOADS = ("none", "cpu", "overlapped")
OFFLOAD_POLICIES = ("none", "cpu")
GRADIENT_SYNCS = ("per-microbatch", "last-microbatch")
OPTIMIZERS = ("adamw", "adamw8bit", "adamw4bit", "cpu-offload")

#: What this engine honors of the precision contract. FSDP2 keeps a
#: trained weight's shard at the dtype the weight was materialized at, the
#: master when one is kept, sums gradients into that shard and keeps AdamW's
#: moments at it, so the gradient and state dtypes must be that dtype; a
#: float16 model would need loss scaling the engine does not do; stochastic
#: rounding comes from torchao's AdamW and applies to bfloat16 weights.
CAPABILITIES = Capabilities(
    "fsdp2",
    model_dtypes=("bfloat16", "float32"),
    master_dtypes=(None, "float32"),
    grad_dtypes=(None, "float32", "bfloat16"),
    opt_state_dtypes=(None, "parameter", "float32", "bfloat16"),
    parameter_roundings=(None, "stochastic"),
    opt_state_roundings=(None,),
    fuse_narrow_gradient_accumulation=False,
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
class Settings(common.Settings):
    CAPABILITIES: Capabilities = dataclasses.field(
        default=CAPABILITIES, init=False, repr=False, compare=False
    )

    # FSDP2 sums each gradient into the shard it steps, the float32 master
    # under the default contract, so its gradients default to float32 where
    # the shared default is the model dtype; autograd writes every weight
    # gradient before it is added, so it adds a narrower one after.
    grad_dtype: str | None = "float32"
    fuse_narrow_gradient_accumulation: bool = False
    # Memory mechanisms.
    activation_offload: str = "none"
    offload_policy: str = "none"
    pin_memory: bool = True
    forward_prefetch: int = 0
    backward_prefetch: int | str = "auto"
    gradient_sync: str = "per-microbatch"
    reshard_after_forward: bool = True
    reshard_after_backward: bool = True
    compile_blocks: bool = False
    # torchao scenarios.
    float8: bool = False
    optimizer: str = "adamw"
    #: torchao's CPUOffloadOptimizer frees each device gradient once it is
    #: on the host, which it does not do between accumulated microbatches.
    offload_gradients: bool = False

    @classmethod
    def coerce(cls, value):
        if "backward_prefetch" in value:
            value["backward_prefetch"] = backward_prefetch(value["backward_prefetch"])
        return value

    def engine_validation(self, request, geometry, precision):
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
        if self.offload_gradients and self.optimizer != "cpu-offload":
            raise ValueError("offload_gradients is the cpu-offload optimizer's option")
        if self.optimizer == "cpu-offload" and self.offload_policy == "cpu":
            raise Unsupported(
                "torchao's CPUOffloadOptimizer takes the weights on the device;"
                " under offload_policy cpu FSDP2 keeps their shards on the host"
            )
        if (
            self.optimizer == "cpu-offload"
            and precision.parameter_rounding == "stochastic"
        ):
            raise Unsupported(
                "torchao's CPUOffloadOptimizer steps with torch's AdamW, which"
                " rounds to nearest; stochastic rounding is torchao's AdamW"
            )
        if self.offload_gradients and geometry.accumulation != 1:
            raise Unsupported(
                "torchao's CPUOffloadOptimizer offloads gradients only with one"
                " microbatch per update"
            )
        if self.optimizer in ("adamw8bit", "adamw4bit") and (
            precision.opt_state_dtype is not None
        ):
            raise Unsupported(
                f"{self.optimizer} keeps its state in low-bit blocks; leave"
                " opt_state_dtype none"
            )
        if precision.parameter_rounding == "stochastic" and stepped != "bfloat16":
            raise Unsupported("stochastic rounding applies to bfloat16 weights")
        if self.activation_offload not in ACTIVATION_OFFLOADS:
            raise ValueError(f"activation_offload must be one of {ACTIVATION_OFFLOADS}")
        if self.offload_policy not in OFFLOAD_POLICIES:
            raise ValueError(f"offload_policy must be one of {OFFLOAD_POLICIES}")
        if self.gradient_sync not in GRADIENT_SYNCS:
            raise ValueError(f"gradient_sync must be one of {GRADIENT_SYNCS}")
        if self.forward_prefetch < 0:
            raise ValueError("forward_prefetch must be nonnegative")
        backward_prefetch(self.backward_prefetch)
        if self.float8 and precision.model_dtype != "bfloat16":
            raise Unsupported("Float8 training computes on bfloat16 weights")
        if self.float8 and self.mode != "full":
            raise Unsupported("Float8 training converts the linears LoRA adapts")
        if self.verify_reference and (
            self.optimizer != "adamw"
            or self.float8
            or precision.parameter_rounding is not None
        ):
            raise ValueError(
                "the reference check compares plain AdamW without Float8 or"
                " stochastic rounding"
            )


HELP = {
    "activation_offload": "none; cpu, PyTorch's offload_wrapper per block;"
    " overlapped, the stream-overlapped offloader vendored from torchtune",
    "offload_policy": "FSDP2's CPUOffloadPolicy: the weights the optimizer steps,"
    " their gradients and the optimizer state on the host, the step on the CPU",
    "pin_memory": "pin CPU-offloaded shards (with --offload cpu)",
    "forward_prefetch": "explicit lookahead depth; 0 is native forward behavior",
    "backward_prefetch": "native auto prefetch or an explicit depth of at least 1",
    "gradient_sync": "reduce every microbatch, or keep full gradients until the last",
    "reshard_after_forward": "release each block's gathered weights after forward",
    "reshard_after_backward": "reshard between accumulation microbatches",
    "compile_blocks": "torch.compile each block",
    "float8": "torchao Float8 training of every block's linears (tensorwise)",
    "optimizer": "AdamW: fused torch, torchao 8-bit or 4-bit, or torchao's"
    " CPU-offloaded",
    "offload_gradients": "with the cpu-offload optimizer, free each device gradient"
    " once it is copied to the host; one microbatch per update",
}


def add_arguments(parser):
    """The trainer's flags: the common ones, then this engine's own."""

    common.add_arguments(
        parser,
        Settings,
        help=HELP,
        choices={
            "activation_offload": ACTIVATION_OFFLOADS,
            "offload_policy": OFFLOAD_POLICIES,
            "gradient_sync": GRADIENT_SYNCS,
            "optimizer": OPTIMIZERS,
        },
        types={"backward_prefetch": backward_prefetch},
    )


def from_namespace(arguments) -> Settings:
    """The settings a parsed trainer command line describes."""
    return common.from_namespace(Settings, arguments)


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
    "ENVIRONMENT",
    "GEOMETRY_FIELDS",
    "GRADIENT_SYNCS",
    "MODEL_FIELDS",
    "OFFLOAD_POLICIES",
    "OPTIMIZERS",
    "PRECISION_FIELDS",
    "SOURCE_PATHS",
    "Settings",
    "add_arguments",
    "backward_prefetch",
    "configure_prefetch",
    "from_namespace",
]
