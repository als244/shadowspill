"""The settings of one DeepSpeed case, shared by the trainer and the sweep.

On top of what every harness takes (``baselines.common.settings``), the
ZeRO configuration: the stage, the optimizer and parameter offload to the
host or NVMe, the communication and prefetch sizes, activation checkpointing
with DeepSpeed's own ``checkpoint`` and its CPU checkpointing as the
activation offload; and DeepCompile, DeepSpeed's ``torch.compile`` backend,
whose graph passes take over ZeRO's reduction, gathering and prefetch and
offer their own offload of activations, optimizer state or parameters. The
optimizer is DeepSpeed's AdamW, fused on the device or its CPU
implementation when the optimizer is offloaded. What the harness cannot
honor is refused before launch.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from baselines.common import settings as common
from baselines.common.precision import Capabilities, Unsupported
from baselines.common.settings import GEOMETRY_FIELDS, MODEL_FIELDS, PRECISION_FIELDS

#: The sources whose digests identify this harness in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/deepspeed",
    "workloads/models",
    "workloads/lora",
    "workloads/full_model.py",
)

STAGES = (1, 2, 3)
OFFLOADS = ("none", "cpu", "nvme")
ACTIVATION_OFFLOADS = ("none", "cpu")
DEEPCOMPILE_BACKENDS = ("inductor", "eager")
DEEPCOMPILE_OFFLOADS = ("none", "activations", "optimizer-states", "parameters")
#: DeepCompile applies its planning passes at its sixth update and compiles
#: the graphs again; a measured step comes after that.
DEEPCOMPILE_WARMUP_STEPS = 8

#: What this harness honors of the precision contract. DeepSpeed's bf16 and
#: fp16 modes cast every parameter and keep float32 masters in the ZeRO
#: optimizer, whose AdamW moments are float32; fp16 runs under its dynamic
#: loss scaling. Gradients reduce and accumulate in the model dtype; stage
#: one's optimizer accumulates at float32 when asked, stages two and three
#: do not. No rounding choices.
CAPABILITIES = Capabilities(
    "deepspeed",
    model_dtypes=("bfloat16", "float16", "float32"),
    master_dtypes=(None, "float32"),
    grad_dtypes=(None, "float32", "bfloat16", "float16"),
    opt_state_dtypes=(None, "parameter", "float32"),
    parameter_roundings=(None,),
    opt_state_roundings=(None,),
    round_accumulation_once=False,
)


@dataclasses.dataclass(frozen=True)
class Settings(common.Settings):
    CAPABILITIES: Capabilities = dataclasses.field(
        default=CAPABILITIES, init=False, repr=False, compare=False
    )

    # ZeRO.
    zero_stage: int = 2
    offload_optimizer: str = "none"
    offload_param: str = "none"
    nvme_path: str | None = None
    pin_memory: bool = True
    overlap_comm: bool = True
    reduce_bucket_size: int = 500_000_000
    prefetch_bucket_size: int = 50_000_000
    max_live_parameters: int = 1_000_000_000
    param_persistence_threshold: int = 100_000
    # Activation checkpointing.
    activation_offload: str = "none"
    contiguous_checkpointing: bool = False
    # DeepCompile.
    deepcompile: bool = False
    deepcompile_backend: str = "inductor"
    deepcompile_offload: str = "none"
    deepcompile_free_activation: bool = False

    def harness_validation(self, request, geometry, precision):
        if self.zero_stage not in STAGES:
            raise ValueError(f"zero_stage must be one of {STAGES}")
        if self.offload_optimizer not in OFFLOADS or self.offload_param not in OFFLOADS:
            raise ValueError(f"offloads must be one of {OFFLOADS}")
        if self.offload_optimizer != "none" and self.zero_stage < 2:
            raise Unsupported(
                "DeepSpeed offloads the optimizer in stages two and three"
            )
        if self.offload_param != "none" and self.zero_stage != 3:
            raise Unsupported("DeepSpeed offloads parameters in stage three")
        if (
            "nvme" in (self.offload_optimizer, self.offload_param)
            and not self.nvme_path
        ):
            raise ValueError("NVMe offload needs nvme_path")
        if self.activation_offload not in ACTIVATION_OFFLOADS:
            raise ValueError(f"activation_offload must be one of {ACTIVATION_OFFLOADS}")
        if self.activation_offload == "cpu" and self.recompute == 0:
            raise ValueError(
                "CPU checkpointing offloads the inputs of recomputed blocks"
            )
        if min(self.reduce_bucket_size, self.prefetch_bucket_size) < 1:
            raise ValueError("bucket sizes must be positive")
        if self.deepcompile_backend not in DEEPCOMPILE_BACKENDS:
            raise ValueError(
                f"deepcompile_backend must be one of {DEEPCOMPILE_BACKENDS}"
            )
        if self.deepcompile_offload not in DEEPCOMPILE_OFFLOADS:
            raise ValueError(
                f"deepcompile_offload must be one of {DEEPCOMPILE_OFFLOADS}"
            )
        if not self.deepcompile:
            if self.deepcompile_offload != "none" or self.deepcompile_free_activation:
                raise ValueError(
                    "DeepCompile's offload and free-activation options need deepcompile"
                )
        else:
            both_on_host = (
                self.offload_optimizer == "cpu" and self.offload_param == "cpu"
            )
            resolved_grad = precision.grad_dtype or precision.model_dtype
            if self.deepcompile_offload != "none" and self.zero_stage != 3:
                raise Unsupported(
                    "DeepCompile offloads under ZeRO-3; stages one and two compile"
                    " the gradient reduction alone"
                )
            if (
                self.deepcompile_offload == "optimizer-states"
                and self.offload_optimizer != "none"
            ):
                raise Unsupported(
                    "DeepCompile's optimizer-state offload and ZeRO's optimizer"
                    " offload manage the same state; choose one"
                )
            if both_on_host and self.deepcompile_offload != "parameters":
                raise Unsupported(
                    "with parameters and optimizer state both offloaded to the host,"
                    " DeepCompile offloads parameters itself: deepcompile_offload"
                    " parameters"
                )
            if self.deepcompile_free_activation and self.deepcompile_backend != "eager":
                raise Unsupported(
                    "DeepCompile frees activations eagerly under its eager backend only"
                )
            if self.activation_offload == "cpu":
                raise Unsupported(
                    "DeepSpeed's CPU checkpointing runs outside the compiled graph;"
                    " DeepCompile's own activation offload is deepcompile_offload"
                    " activations"
                )
            if self.recompute:
                raise Unsupported(
                    "DeepCompile compiles the model whole, and DeepSpeed's"
                    " checkpointing inside its graphs loses the recomputed blocks'"
                    " gradients; DeepCompile offloads or frees activations instead"
                )
            if (
                self.deepcompile_offload == "parameters"
                and self.implementation == "mlops"
            ):
                raise Unsupported(
                    "DeepCompile's parameter offload pass hands a kernel's"
                    " tuple-valued last use to its offload operator; it takes the"
                    " pytorch implementation"
                )
            if self.warmup_steps < DEEPCOMPILE_WARMUP_STEPS:
                raise ValueError(
                    "DeepCompile plans its graphs again at its sixth update; warm up"
                    f" {DEEPCOMPILE_WARMUP_STEPS} steps or more"
                )
            if self.zero_stage == 1 and geometry.accumulation > 1:
                raise Unsupported(
                    "ZeRO-1 under DeepCompile takes one microbatch per update: its"
                    " reduce aborts at the second update when gradients accumulate"
                    " across microbatches (the buckets it released are flushed"
                    " before it makes them again)"
                )
            if resolved_grad == "float32" and precision.model_dtype != "float32":
                raise Unsupported(
                    "DeepCompile takes ZeRO's bit16 optimizer; float32 accumulation"
                    " under a narrower model is DeepSpeed's BF16 optimizer, which"
                    " it does not take"
                )
            if self.mode == "lora" and self.zero_stage != 3:
                raise Unsupported(
                    "DeepCompile's ZeRO-1 and 2 passes register the optimizer's"
                    " parameters only, so a frozen base parameter has no identity in"
                    " its graphs; LoRA under DeepCompile is ZeRO-3"
                )
            if precision.model_dtype == "float16":
                raise Unsupported(
                    "DeepCompile takes bfloat16 and float32 models here; its float16"
                    " path under DeepSpeed's loss scaling is not qualified"
                )
        if precision.master_dtype is None and precision.model_dtype != "float32":
            raise Unsupported(
                "DeepSpeed keeps float32 masters in its bf16 and fp16 modes"
            )
        if precision.opt_state_dtype not in (None, "parameter", "float32"):
            raise Unsupported("DeepSpeed's AdamW keeps float32 moments")
        resolved_grad = precision.grad_dtype or precision.model_dtype
        if resolved_grad not in (precision.model_dtype, "float32"):
            raise Unsupported(
                "DeepSpeed reduces gradients in the model dtype or float32"
            )
        if (
            resolved_grad == "float32"
            and precision.model_dtype != "float32"
            and self.zero_stage > 1
        ):
            raise Unsupported(
                "ZeRO stages two and three accumulate gradients in the model"
                " dtype; float32 accumulation is stage one's"
            )
        if self.verify_reference and precision.model_dtype == "float16":
            raise ValueError("the reference check does not model fp16 loss scaling")


def deepspeed_config(settings: Settings, geometry, precision) -> dict[str, Any]:
    """The DeepSpeed configuration one case runs under."""

    zero: dict[str, Any] = {
        "stage": settings.zero_stage,
        "overlap_comm": settings.overlap_comm,
        "contiguous_gradients": True,
        "reduce_bucket_size": settings.reduce_bucket_size,
    }
    for name, device in (
        ("offload_optimizer", settings.offload_optimizer),
        ("offload_param", settings.offload_param),
    ):
        if device != "none":
            zero[name] = {"device": device, "pin_memory": settings.pin_memory}
            if device == "nvme":
                zero[name]["nvme_path"] = settings.nvme_path
    if settings.zero_stage == 3:
        zero.update(
            {
                "stage3_prefetch_bucket_size": settings.prefetch_bucket_size,
                "stage3_max_live_parameters": settings.max_live_parameters,
                "stage3_param_persistence_threshold": (
                    settings.param_persistence_threshold
                ),
            }
        )
    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": geometry.sequences_per_microbatch,
        "gradient_accumulation_steps": geometry.accumulation,
        "train_batch_size": geometry.sequences_per_microbatch
        * geometry.accumulation
        * geometry.world_size,
        "zero_optimization": zero,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": settings.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": settings.weight_decay,
            },
        },
        "gradient_clipping": 0.0,
        "steps_per_print": 1_000_000_000,
        "wall_clock_breakdown": False,
    }
    if precision.model_dtype == "bfloat16":
        config["bf16"] = {"enabled": True}
    elif precision.model_dtype == "float16":
        config["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}
    if (precision.grad_dtype or precision.model_dtype) == "float32" and (
        precision.model_dtype != "float32"
    ):
        config["communication_data_type"] = "fp32"
        config["data_types"] = {"grad_accum_dtype": "fp32"}
    if settings.deepcompile:
        config["compile"] = {
            "deepcompile": True,
            "offload_activation": settings.deepcompile_offload == "activations",
            "offload_opt_states": settings.deepcompile_offload == "optimizer-states",
            "offload_parameters": settings.deepcompile_offload == "parameters",
            "free_activation": settings.deepcompile_free_activation,
        }
    return config


HELP = {
    "zero_stage": "ZeRO stage: 1 shards optimizer state, 2 gradients too,"
    " 3 parameters too",
    "offload_optimizer": "optimizer state and step on the host (cpu) or NVMe",
    "offload_param": "parameters on the host or NVMe between uses (stage 3)",
    "nvme_path": "the directory NVMe offload swaps to",
    "pin_memory": "pin host-offloaded state",
    "overlap_comm": "overlap gradient reduction with backward",
    "reduce_bucket_size": "elements per gradient reduction bucket",
    "prefetch_bucket_size": "elements prefetched ahead of use (stage 3)",
    "max_live_parameters": "elements of gathered parameters kept at once (stage 3)",
    "param_persistence_threshold": "parameters below this size stay gathered (stage 3)",
    "activation_offload": "none, or cpu: the inputs of recomputed blocks on the host",
    "contiguous_checkpointing": "one contiguous buffer for the checkpointed inputs",
    "deepcompile": "compile the model with DeepCompile, DeepSpeed's torch.compile"
    " backend whose graph passes take over ZeRO's hooks",
    "deepcompile_backend": "the compiler DeepCompile hands its graphs to",
    "deepcompile_offload": "what DeepCompile moves to pinned host memory as the"
    " memory budget requires (ZeRO-3): saved activations, optimizer state or"
    " parameters",
    "deepcompile_free_activation": "free activations eagerly in DeepCompile's"
    " backward (eager backend)",
}


def add_arguments(parser):
    common.add_arguments(
        parser,
        Settings,
        help=HELP,
        choices={
            "zero_stage": STAGES,
            "offload_optimizer": OFFLOADS,
            "offload_param": OFFLOADS,
            "activation_offload": ACTIVATION_OFFLOADS,
            "deepcompile_backend": DEEPCOMPILE_BACKENDS,
            "deepcompile_offload": DEEPCOMPILE_OFFLOADS,
        },
    )


def from_namespace(arguments) -> Settings:
    return common.from_namespace(Settings, arguments)


def command(python, world_size, settings, output):
    """The torchrun command line of one case; DeepSpeed takes its ranks from
    torchrun's environment."""
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "--max-restarts=0",
        "-m",
        "baselines.deepspeed.train",
        "--output",
        str(output),
        *settings.arguments(),
    ]


__all__ = [
    "ACTIVATION_OFFLOADS",
    "CAPABILITIES",
    "DEEPCOMPILE_BACKENDS",
    "DEEPCOMPILE_OFFLOADS",
    "DEEPCOMPILE_WARMUP_STEPS",
    "GEOMETRY_FIELDS",
    "MODEL_FIELDS",
    "OFFLOADS",
    "PRECISION_FIELDS",
    "SOURCE_PATHS",
    "STAGES",
    "Settings",
    "add_arguments",
    "command",
    "deepspeed_config",
    "from_namespace",
]
