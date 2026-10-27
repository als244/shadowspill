"""The settings of one NeMo AutoModel case, shared by the launcher and the
trainer.

The engine trains NVIDIA's implementation of the catalog model's
architecture (AutoModel's own model where it registers one, Hugging Face's
otherwise) with AutoModel's fine-tuning recipe and its own training loop,
over the tokens every engine trains on. The model is built from the case's
Hugging Face configuration, the catalog model's from its family's
``to_huggingface`` or a named checkpoint's own. The shared settings (the
model, the geometry, the precision contract, the run) become AutoModel's
recipe sections; AutoModel's own options (``OPTIONS``) are settings under
AutoModel's names, each left to AutoModel's default unless the case sets it.
``README.md`` says what each does and which go together.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from baselines.common import settings as common
from baselines.common.precision import Capabilities, Unsupported

from .options import OPTIONS

#: The sources whose digests identify this engine in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/engines/automodel",
)

#: The environment beside the launcher's that the trainer runs in.
ENVIRONMENT = "nvidia-training"

#: The model block's name for NVIDIA's implementation.
IMPLEMENTATION = "automodel"

#: What AutoModel honors of the precision contract. Its weights are kept at
#: the storage dtype; float32 masters of bfloat16 weights come from FSDP2's
#: mixed precision over float32 storage when FSDP2 shards (two or more
#: ranks), or from Transformer Engine's fused Adam, which keeps float32 or
#: float16 masters and its moments at any of three dtypes on any number of
#: ranks. Gradients accumulate at the weights' dtype, at float32 only under
#: FSDP2's float32 storage. Autograd writes every weight gradient before it
#: is added, so a narrower one is added after.
CAPABILITIES = Capabilities(
    IMPLEMENTATION,
    model_dtypes=("bfloat16", "float32"),
    master_dtypes=(None, "float32", "float16"),
    grad_dtypes=(None, "float32", "bfloat16"),
    opt_state_dtypes=(None, "parameter", "float32", "float16", "bfloat16"),
    parameter_roundings=(None,),
    opt_state_roundings=(None,),
    fuse_narrow_gradient_accumulation=False,
)


#: The common LoRA options AutoModel's adapter takes.
LORA_OPTIONS = ("rank", "alpha", "factor_dtype", "head", "shared_experts")


def parse_list(value):
    """A comma-separated list of names, kept as the string a case wrote."""
    names = [name.strip() for name in str(value).split(",") if name.strip()]
    if not names:
        raise ValueError("a list names at least one entry")
    return ",".join(names)


@dataclasses.dataclass(frozen=True)
class Base(common.Settings):
    CAPABILITIES: Capabilities = dataclasses.field(
        default=CAPABILITIES, init=False, repr=False, compare=False
    )

    implementation: str = IMPLEMENTATION
    # Autograd writes every weight gradient before AutoModel adds it, so a
    # narrower one is added after.
    fuse_narrow_gradient_accumulation: bool = False
    # AutoModel checkpoints every decoder layer or none: none leaves it to
    # activation_checkpointing, 0 is none and 1 is full.
    recompute: float | None = None

    def request(self):
        """The catalog's model of the same architecture, by one of our
        implementations (PyTorch's where there is one): what the case is
        validated against and its useful work counted from."""

        from baselines.common.models import ModelRequest
        from workloads.models import catalog

        implementations = catalog.entry(self.model).implementations
        implementation = (
            "pytorch" if "pytorch" in implementations else implementations[0]
        )
        return ModelRequest(
            self.model, implementation, self.mode, self.lora, self.model_config
        )

    def describe(self, world_size):
        value = super().describe(world_size)
        value["model"]["implementation"] = self.implementation
        return value

    def model_parallel(self) -> int:
        return (self.tp_size or 1) * (self.pp_size or 1) * (self.cp_size or 1)

    def geometry(self, world_size):
        """A replica is ``tensor x pipeline x context`` ranks."""

        from baselines.common.geometry import Geometry

        if self.tokens_per_step is None:
            raise ValueError("a case names its tokens_per_step")
        split = self.model_parallel()
        if split < 1 or world_size % split:
            raise ValueError(
                f"{world_size} ranks do not form replicas of {split}"
                " (tensor x pipeline x context parallel)"
            )
        return Geometry(
            self.sequence_length,
            self.tokens_per_step,
            self.tokens_per_microbatch,
            world_size,
            None if split == 1 else world_size // split,
        )

    def resolve(self, world_size) -> dict[str, Any]:
        """The Hugging Face configuration the model is built from (the named
        checkpoint's own, or the catalog model's), the loss and the linear
        layers' backend, and the precision mechanism the contract maps to on
        this many ranks."""

        resolved = {
            "loss": self.loss_kind(),
            "linear": self.linear_backend(),
            "precision_mechanism": self.mechanism(world_size),
            # Whether the model routes tokens to experts: a mixture AutoModel
            # builds from Hugging Face's classes trains no balancing loss.
            "mixture": self.request().entry.mixture,
        }
        if self.checkpoint is not None:
            path = Path(self.checkpoint) / "config.json"
            return {**resolved, "huggingface_config": json.loads(path.read_text())}
        entry = self.request().entry
        family = import_module(f"workloads.models.{entry.family}.config")
        convert = getattr(family, "to_huggingface", None)
        if convert is None:
            raise Unsupported(
                f"workloads.models.{entry.family}.config has no to_huggingface,"
                " so AutoModel cannot build its model"
            )
        return {
            **resolved,
            "huggingface_config": convert(entry.config(**dict(self.model_config))),
        }

    def loss_kind(self) -> str:
        """The case's loss, or the engine's: chunked, whose head and loss
        never hold the full logits, unless tensor parallelism shards the head
        over the vocabulary, where te_parallel is AutoModel's one loss that
        computes on the shards (the linear losses refuse a tensor-parallel
        head and masked gathers the whole logits)."""

        if self.loss is not None:
            return self.loss
        return "te_parallel" if (self.tp_size or 1) > 1 else "chunked"

    def linear_backend(self) -> str | None:
        """The case's linear layers, or torch's beside the chunked loss:
        AutoModel's models that build their head through the linear backend
        build Transformer Engine's where it imports, a head the chunked loss
        refuses, and AutoModel's own chunked recipes take torch's. None
        leaves AutoModel's default."""

        if self.linear is not None:
            return self.linear
        return "torch" if self.loss_kind() == "chunked" else None

    def mechanism(self, world_size) -> dict[str, Any]:
        """The AutoModel settings the contract maps to: the weights' storage
        dtype, the strategy's mixed precision (FSDP2's policy, Megatron-FSDP's
        float32 main weights and gradients), and the optimizer with its
        masters and moments. AutoModel parallelizes nothing on one rank:
        FSDP2 keeps the model as built, DDP and Megatron-FSDP cast it to
        bfloat16."""

        precision = self.precision()
        resolved = precision.resolved()
        model = precision.model_dtype
        gradients = resolved["grad_dtype"]
        moments = resolved["opt_state_dtype"]
        master = precision.master_dtype
        strategy = self.strategy or "fsdp2"
        # FSDP2 shards on any mesh of more than one rank; Megatron-FSDP only
        # where its data-parallel shard group spans more than one, and with
        # tensor, pipeline or context parallelism over every rank wraps
        # nothing.
        if strategy == "megatron_fsdp":
            sharded = world_size // self.model_parallel() > 1
        else:
            sharded = world_size > 1
        value = {
            "strategy": strategy,
            "storage": model,
            "optimizer": None,
            "master": None,
            "moments": None,
            "mp_param_dtype": model,
            "mp_reduce_dtype": gradients,
            "preserve_fp32_weights": None,
            "grad_reduce_in_fp32": None,
        }
        if model == "float32":
            # The weights are their own float32 masters.
            plain = moments in ("default", "float32")
            return {
                **value,
                "optimizer": self.optimizer or ("adamw" if plain else "fused_adam"),
                "moments": "float32" if plain else moments,
            }
        if strategy == "megatron_fsdp" and sharded:
            # Megatron-FSDP's own float32 main weights and gradients.
            return {
                **value,
                "optimizer": self.optimizer or "adamw",
                "moments": "float32" if master is not None else model,
                "preserve_fp32_weights": master == "float32",
                "grad_reduce_in_fp32": gradients == "float32",
                "master": master,
            }
        if master is None:
            plain = moments in ("default", model)
            return {
                **value,
                "optimizer": self.optimizer or ("adamw" if plain else "fused_adam"),
                "moments": model if plain else moments,
            }
        if gradients == "float32" and strategy == "fsdp2" and sharded:
            # AutoModel's float32 master weights: float32 storage computed at
            # the model dtype by FSDP2, gradients reduced and kept at float32.
            plain = moments in ("default", "float32")
            return {
                **value,
                "storage": "float32",
                "optimizer": self.optimizer or ("adamw" if plain else "fused_adam"),
                "moments": "float32" if plain else moments,
            }
        return {
            **value,
            "optimizer": self.optimizer or "fused_adam",
            "master": master,
            "moments": "float32" if moments == "default" else moments,
        }

    def validate(self, world_size):
        # The options' values before the geometry reads the parallel sizes,
        # and the checkpoint's kind before its weights are read.
        self.check_values()
        self.check_checkpoint()
        return super().validate(world_size)

    def check_checkpoint(self):
        """A checkpoint AutoModel would build more than the text model of:
        its configuration nests the text model's under ``text_config``."""

        if self.checkpoint is None:
            return
        config = json.loads((Path(self.checkpoint) / "config.json").read_text())
        if "text_config" in config:
            raise Unsupported(
                "the checkpoint is a model around a text model"
                f" ({', '.join(config.get('architectures') or [])}, the text"
                " model's configuration under text_config): AutoModel builds the"
                " whole of it, with what it adds (a vision tower, a"
                " multi-token-prediction layer whose loss joins the objective),"
                " and the shared objective is the text model's"
            )

    def engine_validation(self, request, geometry, precision):
        """The engine's own rules: NVIDIA's implementation, option values of
        the types and choices AutoModel takes, the shared settings and the
        precision contract as AutoModel's options can say them. What goes
        together in AutoModel is AutoModel's to decide as it sets up: a case
        it refuses fails there with its message, and ``README.md`` gathers
        its rules."""

        if self.implementation != IMPLEMENTATION:
            raise Unsupported(
                "AutoModel trains NVIDIA's implementation; implementation must be"
                f" {IMPLEMENTATION}, not {self.implementation}"
            )
        if self.verify_reference:
            raise Unsupported(
                "the reference check compares our implementation's updates;"
                " AutoModel trains NVIDIA's"
            )
        if self.recompute not in (None, 0, 1):
            raise Unsupported(
                "AutoModel checkpoints every decoder layer or none: recompute"
                " takes 0 or 1"
            )
        if self.recompute is not None and self.activation_checkpointing is not None:
            raise ValueError(
                "recompute and activation_checkpointing are two ways to say one"
                " thing; set one"
            )
        self.check_contract(precision, geometry.world_size)
        if self.mode == "lora":
            unknown = set(self.lora) - set(LORA_OPTIONS)
            if unknown:
                raise Unsupported(
                    f"AutoModel's LoRA takes {', '.join(LORA_OPTIONS)}; not"
                    f" {sorted(unknown)}"
                )
            if self.lora.get("head", "frozen") != "frozen":
                raise Unsupported("the engine's AutoModel LoRA leaves the head frozen")

    def check_values(self):
        """Each option's value of its type, and among AutoModel's choices."""

        for option in OPTIONS:
            value = getattr(self, option.name)
            if value is None:
                continue
            if option.kind is bool and not isinstance(value, bool):
                raise ValueError(f"{option.name} must be a boolean")
            if option.kind is int and type(value) is not int:
                raise ValueError(f"{option.name} must be an integer")
            if option.kind is float and type(value) not in (int, float):
                raise ValueError(f"{option.name} must be a number")
            if option.form == "names":
                names = parse_list(value).split(",")
                unknown = sorted(set(names) - set(option.choices))
                if unknown:
                    raise ValueError(
                        f"{option.name} takes {', '.join(option.choices)};"
                        f" not {unknown}"
                    )
            elif option.choices and value not in option.choices:
                raise ValueError(f"{option.name} must be one of {option.choices}")

    def check_contract(self, precision, world_size):
        """The precision contract as AutoModel's options can say it, and the
        case's own options where they would say otherwise."""

        resolved = precision.resolved()
        gradients = resolved["grad_dtype"]
        mechanism = self.mechanism(world_size)
        strategy = mechanism["strategy"]
        if (
            strategy in ("ddp", "megatron_fsdp")
            and world_size == 1
            and precision.model_dtype != "bfloat16"
        ):
            raise Unsupported(
                f"AutoModel's {strategy} casts the model to bfloat16 on one"
                f" rank; a {precision.model_dtype} model takes fsdp2"
            )
        storage = mechanism["storage"]
        master = precision.master_dtype
        moments = resolved["opt_state_dtype"]
        # Each branch of the mapping holds the contract or is refused here:
        # nothing is substituted.
        if precision.model_dtype == "float32" and master not in (None, "float32"):
            raise Unsupported(
                "a float32 model's weights are their own float32 masters; masters"
                f" at {master} are no run of AutoModel's"
            )
        if precision.model_dtype == "float32" and gradients != "float32":
            raise Unsupported(
                "AutoModel keeps a float32 model's gradients at float32, not"
                f" {gradients}"
            )
        if (
            storage == "float32"
            and precision.model_dtype != "float32"
            and master
            not in (
                None,
                "float32",
            )
        ):
            raise Unsupported(
                "FSDP2's float32 storage is the float32 master of bfloat16 compute;"
                f" masters at {master} are not"
            )
        if mechanism["preserve_fp32_weights"] is not None:
            held = "float32" if master is not None else precision.model_dtype
            if moments not in ("default", held):
                raise Unsupported(
                    f"Megatron-FSDP's Adam keeps its moments at {held} here (float32"
                    " with masters, the weights' dtype without); not"
                    f" {moments}"
                )
        if strategy != "fsdp2":
            named = [
                name
                for name in (
                    "cpu_offload",
                    "offload_pin_memory",
                    "mp_param_dtype",
                    "mp_reduce_dtype",
                    "mp_output_dtype",
                )
                if getattr(self, name) is not None
            ]
            if named:
                raise Unsupported(
                    f"{', '.join(named)} are FSDP2's; AutoModel's {strategy} reads"
                    " none of them"
                )
        main_gradients = mechanism["grad_reduce_in_fp32"]
        if gradients == "float32" and storage != "float32" and not main_gradients:
            raise Unsupported(
                "AutoModel accumulates gradients at the weights' dtype; float32"
                " gradients of bfloat16 weights take FSDP2's float32 storage with"
                " float32 masters, or Megatron-FSDP's float32 main gradients,"
                " sharded across two or more data-parallel ranks"
            )
        optimizer = mechanism["optimizer"]
        own_masters = mechanism["preserve_fp32_weights"]
        if mechanism["master"] == "float16" and own_masters is not None:
            raise Unsupported(
                "Megatron-FSDP keeps float32 main weights or none; float16 masters"
                " take fsdp2 or ddp with fused_adam"
            )
        if (
            mechanism["master"] is not None
            and not own_masters
            and optimizer != "fused_adam"
        ):
            raise Unsupported(
                f"{optimizer} keeps no master of {storage} weights; the"
                " contract's masters take fused_adam"
            )
        factors = (self.lora or {}).get("factor_dtype")
        if (
            self.mode == "lora"
            and world_size == 1
            and factors not in (None, precision.model_dtype)
        ):
            raise Unsupported(
                "on one rank AutoModel casts each LoRA module, its adapters"
                f" included, to the compute dtype ({precision.model_dtype}); factors"
                f" at {factors} take two or more ranks"
            )
        if self.cpu_offload and optimizer == "fused_adam":
            raise Unsupported(
                "Transformer Engine's FusedAdam steps on the device, and FSDP2's"
                " CPU offload keeps the shards on the host; a contract that takes"
                " FusedAdam (masters, or moments at another dtype) takes no"
                " cpu_offload"
            )
        stepped = "float32" if own_masters else storage
        if optimizer in ("adamw", "adam") and mechanism["moments"] != stepped:
            raise Unsupported(
                f"torch's {optimizer} keeps its moments at the stepped weights'"
                f" dtype ({stepped}); other moments take fused_adam"
            )
        if optimizer == "flash_adamw":
            raise Unsupported(
                "FlashAdamW's 24-bit masters are not a dtype the contract names"
            )
        for name in (
            "mp_param_dtype",
            "mp_reduce_dtype",
            "preserve_fp32_weights",
            "grad_reduce_in_fp32",
        ):
            value = getattr(self, name)
            expected = mechanism[name]
            if value is not None and expected is not None and value != expected:
                raise Unsupported(
                    f"the contract sets {name} to {expected}; the case's"
                    f" {value} disagrees"
                )


Settings = dataclasses.make_dataclass(
    "Settings",
    [
        (option.name, option.kind | None, dataclasses.field(default=None))
        for option in OPTIONS
    ],
    bases=(Base,),
    frozen=True,
    namespace={"__module__": __name__},
)
Settings.__doc__ = "An AutoModel case: the common settings and AutoModel's options."

HELP = {
    "recompute": "0 or 1: AutoModel's full activation checkpointing of every"
    " decoder layer, or none; none leaves it to activation_checkpointing",
    **{option.name: option.help for option in OPTIONS},
}


def add_arguments(parser):
    """The trainer's flags: the common ones, then AutoModel's own."""

    common.add_arguments(
        parser,
        Settings,
        help=HELP,
        # A list's names are checked one by one when the case is validated.
        choices={
            option.name: option.choices
            for option in OPTIONS
            if option.choices and option.form == "value"
        },
        types={
            "recompute": float,
            **{
                option.name: parse_list
                if option.form == "names"
                else str
                if option.form == "dtype"
                else option.kind
                for option in OPTIONS
                if option.kind is not bool
            },
        },
        booleans=frozenset(option.name for option in OPTIONS if option.kind is bool),
        implementations=(IMPLEMENTATION,),
    )


def from_namespace(arguments) -> Settings:
    """The settings a parsed trainer command line describes."""
    return common.from_namespace(Settings, arguments)


__all__ = [
    "CAPABILITIES",
    "ENVIRONMENT",
    "IMPLEMENTATION",
    "SOURCE_PATHS",
    "Settings",
    "add_arguments",
    "from_namespace",
]
