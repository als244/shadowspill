"""The settings of one Megatron-Core case, shared by the launcher and the
trainer.

The engine trains NVIDIA's prebuilt model of the catalog model's
architecture with Megatron Bridge's training loop over Megatron-Core. The
model is the one Bridge builds from the case's Hugging Face configuration,
the catalog model's from its family's ``to_huggingface`` or a named
checkpoint's own, whose weights Bridge then imports; or the model of the
Bridge recipe the case names, which must have the same architecture. The
shared settings (the model, the geometry, the precision contract, the run)
are applied on top of Bridge's defaults or the recipe's values; Megatron's
own options (``OPTIONS``) are settings under Megatron's names, each left to
Bridge or the recipe unless the case sets it. ``README.md`` says what each
does and which go together.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from baselines.common import settings as common
from baselines.common.geometry import Geometry
from baselines.common.precision import Capabilities, Unsupported
from baselines.common.settings import GEOMETRY_FIELDS, MODEL_FIELDS, PRECISION_FIELDS

from .options import OPTIONS
from .options.types import DEFAULT_RECOMPUTE_MODULES, MIXTURE_RECOMPUTE_MODULES

#: The sources whose digests identify this engine in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/engines/megatron_core",
)

#: The environment beside the launcher's that the trainer runs in.
ENVIRONMENT = "nvidia-training"

#: The model block's name for NVIDIA's implementation.
IMPLEMENTATION = "megatron_core"

#: What Megatron-Core honors of the precision contract. Under DDP its
#: optimizers keep master copies: float32 ones, or, through its
#: precision-aware optimizer and Transformer Engine's fused Adam (which
#: rounds to nearest), float32 or float16 masters with moments at float32,
#: float16 or bfloat16. Megatron-FSDP can step the compute weights
#: themselves (``megatron_fsdp_main_params_dtype`` none), the one way to
#: keep no master, with Transformer Engine's fused Adam keeping the moments at
#: float32. Gradients accumulate at the model dtype or float32; with
#: ``gradient_accumulation_fusion`` a weight-gradient matmul adds into a
#: bfloat16 accumulation as it writes it, the contract's fused narrow-gradient
#: accumulation. A float16 model would need loss scaling the contract does
#: not name.
CAPABILITIES = Capabilities(
    IMPLEMENTATION,
    model_dtypes=("bfloat16", "float32"),
    master_dtypes=(None, "float32", "float16"),
    grad_dtypes=(None, "float32", "bfloat16"),
    opt_state_dtypes=(None, "parameter", "float32", "float16", "bfloat16"),
    parameter_roundings=(None,),
    opt_state_roundings=(None,),
    fuse_narrow_gradient_accumulation=True,
)


#: The updates Megatron runs before capturing CUDA graphs when a case names
#: no ``cuda_graph_warmup_steps``.
CUDA_GRAPH_WARMUP_STEPS = 3

#: The common settings Megatron's options leave alone.
LORA_OPTIONS = ("rank", "alpha", "factor_dtype", "head", "shared_experts")


def aware(precision) -> bool:
    """Whether the contract needs Megatron's precision-aware optimizer:
    masters or moments at another dtype than float32. Gradients accumulate
    at either dtype without it (``grad_reduce_in_fp32``), and with no master
    (Megatron-FSDP stepping the compute weights) Transformer Engine's fused
    Adam keeps its moments at float32, the only moments the contract
    accepts there."""

    if precision.master_dtype is None:
        return False
    moments = precision.resolved()["opt_state_dtype"]
    return precision.master_dtype != "float32" or moments not in (
        "float32",
        "default",
    )


def parse_value(option, value):
    """A setting's value as Megatron takes it: a list for a list form, a
    dtype for a dtype form, the value itself otherwise."""

    import torch

    if option.form == "names":
        return parse_list(value).split(",")
    if option.form == "integers":
        return [int(item) for item in parse_list(value).split(",")]
    if option.form == "dtype":
        return getattr(torch, value)
    return value


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
    # Megatron-Core recomputes what its configuration says; none leaves it
    # to Bridge or the recipe, a fraction recomputes that share of the
    # layers whole (full granularity, block method).
    recompute: float | None = None
    #: A Bridge recipe whose configuration the case starts from, by its
    #: name in ``megatron.bridge.recipes``; none starts from Bridge's
    #: defaults around the model of the Hugging Face configuration.
    recipe: str | None = None

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
        return (
            (self.tensor_model_parallel_size or 1)
            * (self.pipeline_model_parallel_size or 1)
            * (self.context_parallel_size or 1)
        )

    def geometry(self, world_size) -> Geometry:
        """A replica is ``tensor x pipeline x context`` ranks."""

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
        """The Hugging Face configuration NVIDIA's model is built from (the
        named checkpoint's own, or the catalog model's), and the modules
        selective recomputation recomputes."""

        entry = self.request().entry
        modules = {"recompute_modules": self.selective_modules(entry)}
        if self.checkpoint is not None:
            path = Path(self.checkpoint) / "config.json"
            return {**modules, "huggingface_config": json.loads(path.read_text())}
        family = import_module(f"workloads.models.{entry.family}.config")
        convert = getattr(family, "to_huggingface", None)
        if convert is None:
            raise Unsupported(
                f"workloads.models.{entry.family}.config has no to_huggingface,"
                " so Megatron cannot build its model"
            )
        return {
            **modules,
            "huggingface_config": convert(entry.config(**dict(self.model_config))),
        }

    def selective_modules(self, entry):
        """The modules selective recomputation recomputes: the case's, or by
        default core_attn and layernorm, with moe_act in a mixture of
        experts; none without selective recomputation."""

        if self.recompute_granularity != "selective":
            return None
        if self.recompute_modules is not None:
            return parse_list(self.recompute_modules).split(",")
        extra = MIXTURE_RECOMPUTE_MODULES if entry.mixture else ()
        return [*DEFAULT_RECOMPUTE_MODULES, *extra]

    def environment(self):
        """Transformer Engine picks its activation-offload implementation from
        ``NVTE_CPU_OFFLOAD_V1`` once, as it is imported, and Bridge refuses
        fine-grained offloading without the version-1 one, the one whose
        marks keep Transformer Engine's weights out of Megatron's offloads;
        CUDA reads ``CUDA_DEVICE_MAX_CONNECTIONS`` as it starts. The case
        sets each or leaves it unset, whatever the shell had."""

        connections = self.cuda_device_max_connections
        return {
            "NVTE_CPU_OFFLOAD_V1": (
                "1" if self.fine_grained_activation_offloading else None
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": (
                None if connections is None else str(connections)
            ),
        }

    def validate(self, world_size):
        # The options' values before the geometry reads the parallel sizes.
        self.check_values()
        return super().validate(world_size)

    def engine_validation(self, request, geometry, precision):
        """The engine's own rules: NVIDIA's implementation, option values of
        the types and choices Megatron takes, the shared settings and the
        precision contract as Megatron's options can say them, and CUDA
        graphs captured before the measured steps. What goes together in
        Megatron is Megatron's and Bridge's to decide as they set up: a case
        they refuse fails there with their message, and ``README.md``
        gathers their rules."""

        if self.implementation != IMPLEMENTATION:
            raise Unsupported(
                "Megatron trains NVIDIA's implementation; implementation must be"
                f" {IMPLEMENTATION}, not {self.implementation}"
            )
        if self.verify_reference:
            raise Unsupported(
                "the reference check compares our implementation's updates;"
                " Megatron trains NVIDIA's"
            )
        if self.recipe is not None and not self.recipe.isidentifier():
            raise ValueError(f"{self.recipe!r} is not a recipe name")
        if self.recompute is not None:
            # The share becomes Megatron's block method over the first layers
            # of each stage and chunk, counted from an even split; Megatron's
            # own recomputation options would say it again or otherwise, and
            # uneven stages give each a count of its own.
            named = [
                name
                for name in (
                    "recompute_granularity",
                    "recompute_method",
                    "recompute_num_layers",
                    "recompute_modules",
                    "num_layers_in_first_pipeline_stage",
                    "num_layers_in_last_pipeline_stage",
                    "account_for_embedding_in_pipeline_split",
                    "account_for_loss_in_pipeline_split",
                )
                if getattr(self, name) not in (None, False)
            ]
            if named:
                raise ValueError(
                    "recompute (a share of whole layers, from an even split of"
                    f" each stage) is set with {', '.join(named)}; set Megatron's"
                    " recomputation options alone, or the share alone"
                )
        if self.recompute_num_layers is not None and self.recompute_num_layers < 1:
            raise ValueError(
                "recompute_num_layers counts layers and is at least 1 (Megatron's"
                " uniform method never ends with fewer)"
            )
        self.check_contract(precision)
        if self.mode == "lora":
            unknown = set(self.lora) - set(LORA_OPTIONS)
            if unknown:
                raise Unsupported(
                    f"Megatron's LoRA takes {', '.join(LORA_OPTIONS)}; not"
                    f" {sorted(unknown)}"
                )
            if self.lora.get("head", "frozen") != "frozen":
                raise Unsupported("Megatron's LoRA leaves the head frozen")
            factor = self.lora.get("factor_dtype")
            if factor is not None and factor != precision.model_dtype:
                raise Unsupported(
                    "Megatron's LoRA keeps its factors at the model's dtype"
                    f" ({precision.model_dtype}): its adapters take the model's"
                    f" precision, not a factor_dtype of {factor}"
                )
        if self.cuda_graph_impl not in (None, "none") or self.optimizer_cuda_graph:
            # The graphs are captured in the update this many updates in
            # (Megatron's default is 3): the layers' and the whole
            # iteration's by Bridge and Megatron-Core, the optimizer step's
            # inside that update's step; the capture is no measured step.
            capture = self.cuda_graph_warmup_steps
            capture = CUDA_GRAPH_WARMUP_STEPS if capture is None else capture
            if capture >= self.warmup_steps:
                raise ValueError(
                    f"the CUDA graphs are captured at update {capture}, which must"
                    f" be one of the {self.warmup_steps} warmup updates; raise"
                    " warmup_steps or lower cuda_graph_warmup_steps"
                )

    def check_values(self):
        """Each option's value of its type, and among Megatron's choices."""

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
            elif option.form == "integers":
                [int(item) for item in parse_list(value).split(",")]
            elif option.choices and value not in option.choices:
                raise ValueError(f"{option.name} must be one of {option.choices}")
        connections = self.cuda_device_max_connections
        if connections is not None and not 1 <= connections <= 32:
            raise ValueError("cuda_device_max_connections must be in [1, 32]")

    def fusing(self, precision):
        """Whether the weight-gradient matmuls add into the accumulated
        gradients as they write them: for gradients narrower than float32,
        the contract's fused narrow-gradient accumulation; for float32 ones,
        the case's gradient_accumulation_fusion, or none for Bridge's
        default (on wherever Transformer Engine imports)."""

        if precision.narrow_gradients:
            return precision.fuse_narrow_gradient_accumulation
        return self.gradient_accumulation_fusion

    def check_contract(self, precision):
        """The precision contract as Megatron's options can say it, and the
        case's own options where they would say otherwise."""

        sharded = self.use_megatron_fsdp
        gradients = precision.resolved()["grad_dtype"]
        if precision.master_dtype is None and not sharded:
            raise Unsupported(
                "Megatron-Core steps the weights themselves only under"
                " Megatron-FSDP (use_megatron_fsdp); its DDP optimizers keep masters"
            )
        if precision.master_dtype is None:
            moments = precision.resolved()["opt_state_dtype"]
            if moments not in ("default", "float32"):
                raise Unsupported(
                    "with no master, Megatron-FSDP's Adam (Transformer Engine's"
                    " fused Adam) keeps its moments at float32"
                )
        if not sharded and gradients not in (precision.model_dtype, "float32"):
            raise Unsupported(
                "Megatron's DDP accumulates gradients at the weights' dtype"
                f" ({precision.model_dtype}) or float32, not {gradients}"
            )
        if aware(precision) and self.use_precision_aware_optimizer is False:
            raise Unsupported(
                "this precision needs Megatron's precision-aware optimizer, which"
                " the case turns off"
            )
        if aware(precision) and self.use_distributed_optimizer is False and not sharded:
            raise Unsupported(
                "this precision needs Megatron's precision-aware optimizer, which"
                " needs the distributed optimizer the case turns off"
            )
        if (
            sharded
            and self.megatron_fsdp_version == 2
            and precision.narrow_gradients
            and precision.fuse_narrow_gradient_accumulation
        ):
            raise Unsupported(
                "Bridge turns gradient-accumulation fusion off under Megatron-FSDP"
                " version 2, so the contract's fused accumulation of narrow"
                " gradients is no run there; set fuse_narrow_gradient_accumulation"
                " false or use version 1"
            )
        fusion = self.gradient_accumulation_fusion
        if (
            precision.narrow_gradients
            and fusion is not None
            and fusion != precision.fuse_narrow_gradient_accumulation
        ):
            raise Unsupported(
                f"for {gradients} gradients gradient_accumulation_fusion is the"
                " contract's fuse_narrow_gradient_accumulation, and the two"
                " disagree; set the contract's"
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
Settings.__doc__ = (
    "A Megatron-Core case: the common settings, the recipe it starts from,"
    " and Megatron's options."
)

HELP = {
    "recipe": "a Bridge recipe to start from, by its name in"
    " megatron.bridge.recipes; none builds from the Hugging Face configuration",
    "recompute": "share of each stage's layers recomputed whole (full granularity,"
    " block method); none leaves recomputation to Megatron's options",
    **{option.name: option.help for option in OPTIONS},
}


def add_arguments(parser):
    """The trainer's flags: the common ones, then Megatron's own."""

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
                if option.form in ("names", "integers")
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
    "GEOMETRY_FIELDS",
    "IMPLEMENTATION",
    "MODEL_FIELDS",
    "PRECISION_FIELDS",
    "SOURCE_PATHS",
    "Settings",
    "add_arguments",
    "from_namespace",
]
