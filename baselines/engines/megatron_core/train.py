"""One Megatron-Core case: NVIDIA's model of the catalog model's
architecture, trained by Megatron Bridge's own loop (``pretrain``); the
trainer ``baselines.common.launch`` starts under torchrun in the
nvidia-training environment.

The model is the provider Bridge builds from the case's Hugging Face
configuration, or the named recipe's, whose architecture must be the same;
a named checkpoint's weights are Bridge's to import. Bridge trains with its
defaults, or the recipe's values, under the shared settings and the case's
Megatron options. The runner's clock runs from Bridge's step callbacks, and
the tokens are the ones every engine trains on, served by a dataset
provider in the order Megatron's sampler asks for them.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.common import runner
from baselines.engines.megatron_core import settings as engine_settings
from baselines.engines.megatron_core.settings import OPTIONS, aware, parse_value

#: The provider fields a recipe's model must share with the case's.
ARCHITECTURE = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "num_moe_experts",
    "moe_router_topk",
    "moe_ffn_hidden_size",
    "share_embeddings_and_output_weights",
)

#: The shared LoRA selection in Megatron's modules, by Bridge's own layout (one
#: adapter on each fused projection, and on each routed expert's): every
#: projection of attention, of Gated DeltaNet's linear attention (its fused
#: in_proj and its out_proj, which Bridge's default names miss), of a dense
#: MLP and of the routed experts, grouped or one module each; the shared
#: experts' only when the selection takes them (``shared_experts``). Bridge
#: matches a module by its own name or its full name against a pattern.
LORA_TARGETS = (
    "linear_qkv",
    "linear_proj",
    "in_proj",
    "out_proj",
    "*.mlp.linear_fc1",
    "*.mlp.linear_fc2",
    "*.experts.linear_fc1",
    "*.experts.linear_fc2",
    "*.experts.*.linear_fc1",
    "*.experts.*.linear_fc2",
)
SHARED_EXPERT_TARGETS = ("*.shared_experts.linear_fc1", "*.shared_experts.linear_fc2")


class Samples(torch.utils.data.Dataset):
    """The case's tokens as Megatron's samples. Megatron's pretraining
    sampler hands each data-parallel replica, in turn, ``micro_batch_size``
    consecutive indices of every run of ``micro_batch_size x replicas``, one
    run per microbatch; sample ``i`` is therefore the sequence at that
    place in the replica's tokens of the step, as every other engine cuts
    its microbatches."""

    def __init__(self, case, count, vocabulary, micro_batch_size, replicas):
        self.case = case
        self.count = count
        self.vocabulary = vocabulary
        self.micro_batch_size = micro_batch_size
        self.replicas = replicas
        self.microbatches = case.geometry.accumulation
        self.length = case.geometry.sequence_length
        self.streams = {}

    def __len__(self):
        return self.count

    def stream(self, replica):
        if replica not in self.streams:
            self.streams[replica] = runner.Tokens(
                self.case.settings,
                self.case.geometry,
                self.vocabulary,
                replica=replica,
                replicas=self.replicas,
                device=None,
            )
        return self.streams[replica]

    def place(self, index):
        """``(replica, step, position)`` of sample ``index``: whose tokens,
        which update, and the first token's column."""

        chunk, offset = divmod(index, self.micro_batch_size * self.replicas)
        replica, row = divmod(offset, self.micro_batch_size)
        step, microbatch = divmod(chunk, self.microbatches)
        start = (microbatch * self.micro_batch_size + row) * self.length
        return replica, step, start

    def __getitem__(self, index):
        replica, step, start = self.place(index)
        tokens, targets = self.stream(replica)(step)
        end = start + self.length
        return {
            "tokens": tokens[0, start:end],
            "labels": targets[0, start:end],
            "loss_mask": torch.ones(self.length, dtype=torch.float32),
            "position_ids": torch.arange(self.length),
        }


def huggingface_config(value):
    """The ``transformers`` configuration of a ``config.json`` mapping."""

    from transformers import AutoConfig

    fields = {key: item for key, item in value.items() if key != "model_type"}
    return AutoConfig.for_model(value["model_type"], **fields)


def dtype_of(name):
    return getattr(torch, name)


def base_config(case, vocabulary):
    """Bridge's configuration before the case's settings: the recipe's, or
    Bridge's defaults around the model of the Hugging Face configuration."""

    from megatron.bridge.models import AutoBridge
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        DistributedDataParallelConfig,
        LoggerConfig,
        OptimizerConfig,
        RNGConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
        ValidationConfig,
    )

    settings = case.settings
    bridge = AutoBridge.from_hf_config(
        huggingface_config(case.resolved["huggingface_config"])
    )
    model = bridge.to_megatron_provider(load_weights=False)
    if settings.recipe is None:
        return ConfigContainer(
            model=model,
            train=TrainingConfig(
                train_iters=1, global_batch_size=1, micro_batch_size=1
            ),
            validation=ValidationConfig(),
            optimizer=OptimizerConfig(optimizer="adam", lr=settings.learning_rate),
            scheduler=SchedulerConfig(),
            ddp=DistributedDataParallelConfig(),
            dataset=None,
            logger=LoggerConfig(),
            tokenizer=TokenizerConfig(
                tokenizer_type="NullTokenizer", vocab_size=vocabulary
            ),
            checkpoint=CheckpointConfig(),
            rng=RNGConfig(),
        )
    recipes = importlib.import_module("megatron.bridge.recipes")
    build = getattr(recipes, settings.recipe, None)
    if not callable(build):
        raise ValueError(f"megatron.bridge.recipes has no recipe {settings.recipe!r}")
    config = build()
    differences = {
        name: (getattr(config.model, name, None), getattr(model, name, None))
        for name in ARCHITECTURE
        if getattr(config.model, name, None) != getattr(model, name, None)
    }
    if differences:
        raise ValueError(
            f"recipe {settings.recipe} builds another model than"
            f" {settings.model}: {differences}"
        )
    # Bridge would set the recipe's environment variables itself, as its loop
    # starts: too late for the ones read as a library is imported or CUDA
    # starts, and outside the case. A case names its environment through its
    # own settings, so a recipe that wants another is refused.
    wanted = {name: str(value) for name, value in config.env_vars.items()}
    differs = {
        name: value for name, value in wanted.items() if os.environ.get(name) != value
    }
    if differs:
        raise ValueError(
            f"recipe {settings.recipe} sets {differs} in the environment, which"
            " the case does not"
        )
    config.env_vars = {}
    return config


#: The options a communication-overlap configuration also derives, which a
#: case that names one keeps over the derived value.
OVERLAP_DERIVED = (
    "overlap_grad_reduce",
    "overlap_param_gather",
    "overlap_param_gather_with_optimizer_step",
    "align_param_gather",
    "bucket_size",
    "overlap_p2p_comm",
    "batch_p2p_comm",
    "defer_embedding_wgrad_compute",
    "wgrad_deferral_limit",
    "overlap_moe_expert_parallel_comm",
    "delay_wgrad_compute",
)


def place(holder, section, name, value):
    """Set one of Megatron's options where Bridge holds it; a name the
    configuration does not have would otherwise be set and never read."""

    if not hasattr(holder, name):
        raise ValueError(f"Bridge's {section} configuration has no {name}")
    setattr(holder, name, value)


def apply(config, case, vocabulary, dataset):
    """The shared settings and the case's Megatron options over the base
    configuration."""

    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.training.comm_overlap import CommOverlapConfig
    from megatron.bridge.training.mixed_precision import MixedPrecisionConfig

    settings, geometry, precision = case.settings, case.geometry, case.precision
    resolved = precision.resolved()
    model, optimizer, ddp = config.model, config.optimizer, config.ddp
    # The geometry and the run.
    sequences = geometry.tokens_per_step // geometry.sequence_length
    config.train.train_iters = settings.warmup_steps + settings.steps
    config.train.global_batch_size = sequences
    config.train.micro_batch_size = geometry.sequences_per_microbatch
    config.train.manual_gc = False
    config.validation.eval_iters = 0
    config.validation.eval_interval = None
    model.seq_length = geometry.sequence_length
    config.dataset = dataset
    config.tokenizer.tokenizer_type = "NullTokenizer"
    config.tokenizer.vocab_size = vocabulary
    # Megatron refuses a seed of zero, every engine's default; its seed
    # sets NVIDIA's initialization alone, the tokens keep the case's.
    config.rng.seed = settings.seed + 1
    config.logger.log_interval = config.train.train_iters + 1
    config.logger.tensorboard_dir = None
    config.checkpoint.save = None
    config.checkpoint.load = None
    config.checkpoint.pretrained_checkpoint = settings.checkpoint
    # AdamW at the shared hyperparameters, a constant rate, no clipping, as
    # every engine steps.
    optimizer.optimizer = "adam"
    optimizer.lr = optimizer.min_lr = settings.learning_rate
    optimizer.weight_decay = settings.weight_decay
    optimizer.adam_beta1, optimizer.adam_beta2 = 0.9, 0.999
    optimizer.adam_eps = 1e-8
    optimizer.clip_grad = 0.0
    config.scheduler.lr_decay_style = "constant"
    config.scheduler.lr_warmup_iters = 0
    config.scheduler.start_weight_decay = settings.weight_decay
    config.scheduler.end_weight_decay = settings.weight_decay
    config.scheduler.weight_decay_incr_style = "constant"
    # The precision contract. Bridge copies the mixed-precision
    # configuration's fields onto the model, the optimizer and DDP, so the
    # contract and the case's FP8 or FP4 options are set there; with neither,
    # it names no FP8 recipe, since Bridge's default, tensorwise, would keep
    # the precision-aware optimizer from holding its masters itself.
    gradient_fp32 = resolved["grad_dtype"] == "float32"
    if precision.model_dtype == "bfloat16":
        mixed = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=gradient_fp32,
        )
    else:
        mixed = MixedPrecisionConfig(
            params_dtype=torch.float32,
            pipeline_dtype=torch.float32,
            grad_reduce_in_fp32=True,
        )
    if settings.fp8 is None:
        mixed.fp8_recipe = None
    config.mixed_precision = mixed
    ddp.grad_reduce_in_fp32 = gradient_fp32
    ddp.check_for_nan_in_grad = False
    if settings.use_megatron_fsdp:
        # Megatron-FSDP keeps its own main weights and gradients: none for
        # the weights steps the compute weights themselves.
        config.dist.use_megatron_fsdp = True
        ddp.megatron_fsdp_main_params_dtype = (
            None if precision.master_dtype is None else dtype_of(precision.master_dtype)
        )
        ddp.megatron_fsdp_main_grads_dtype = dtype_of(resolved["grad_dtype"])
    if aware(precision) or settings.use_precision_aware_optimizer:
        # Transformer Engine's fused Adam holds the masters and the moments
        # at the contract's dtypes; the contract needs it for any but
        # float32, with the distributed optimizer it requires.
        optimizer.use_precision_aware_optimizer = True
        if precision.master_dtype is not None:
            optimizer.main_params_dtype = dtype_of(precision.master_dtype)
        optimizer.main_grads_dtype = dtype_of(resolved["grad_dtype"])
        moments = resolved["opt_state_dtype"]
        moments = dtype_of("float32" if moments == "default" else moments)
        optimizer.exp_avg_dtype = optimizer.exp_avg_sq_dtype = moments
        if aware(precision):
            optimizer.use_distributed_optimizer = ddp.use_distributed_optimizer = True
    # A narrower gradient's accumulation is fused into the weight-gradient
    # matmuls exactly when the contract says (PRECISION.md); float32
    # gradients keep the case's choice, or Bridge's.
    fusion = settings.fusing(precision)
    if fusion is not None:
        model.gradient_accumulation_fusion = fusion
    # Recomputation by the shared share of whole layers, of each model
    # chunk: Megatron's block method counts each pipeline stage's layers, or
    # each virtual stage's.
    if settings.recompute is not None:
        stages = settings.pipeline_model_parallel_size or 1
        chunks = stages * (
            (settings.virtual_pipeline_model_parallel_size or 1) if stages > 1 else 1
        )
        count = round(model.num_layers // chunks * settings.recompute)
        model.recompute_granularity = "full" if count else None
        model.recompute_method = "block" if count else None
        model.recompute_num_layers = count or None
    # Megatron's options, by their names, where the case sets them.
    sections = {
        "model": model,
        "optimizer": optimizer,
        "ddp": ddp,
        "dist": config.dist,
        "mixed": mixed,
    }
    for option in OPTIONS:
        value = getattr(settings, option.name)
        if value is None or option.section in ("comm", "environment"):
            continue
        value = parse_value(option, value)
        for section in option.section.split("+"):
            place(sections[section], section, option.name, value)
    if settings.attention_backend is not None:
        # The provider compares its value with Megatron-Core's enum, which a
        # name never equals (local would leave Transformer Engine's attention
        # in place with every backend disabled).
        from megatron.core.transformer.enums import AttnBackend

        model.attention_backend = AttnBackend[settings.attention_backend]
    # Selective recomputation of the modules the launcher resolved: the
    # case's, or the engine's default.
    if settings.recompute_granularity == "selective":
        model.recompute_modules = case.resolved["recompute_modules"]
    if settings.tp_comm_overlap is not None:
        config.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=settings.tp_comm_overlap
        )
    if config.comm_overlap is not None:
        for name in OVERLAP_DERIVED:
            value = getattr(settings, name)
            if value is not None:
                place(config.comm_overlap, "comm", name, value)
    # The runner checks every measured loss; Megatron's own check of each
    # microbatch's loss would add a host synchronization no other engine
    # makes (and a whole-iteration CUDA graph refuses it).
    config.rerun_state_machine.check_for_nan_in_loss = False
    # CUDA graphs need Transformer Engine's random-number tracker, which
    # Bridge requires and does not turn on itself (Megatron-LM's command
    # line does).
    if settings.cuda_graph_impl not in (None, "none"):
        model.use_te_rng_tracker = True
        config.rng.te_rng_tracker = True
    # LoRA at the shared rank and alpha over the shared selection, the routed
    # experts' one adapter each; its factors follow the model's precision.
    if settings.mode == "lora":
        options = settings.lora
        targets = [*LORA_TARGETS]
        if options.get("shared_experts", False):
            targets.extend(SHARED_EXPERT_TARGETS)
        config.peft = LoRA(
            target_modules=targets,
            dim=int(options.get("rank", 16)),
            alpha=options.get("alpha", 16),
            dropout=0.0,
            share_expert_adapters=False,
        )
    else:
        # A full case trains every weight, whatever a recipe adapts.
        config.peft = None
    if settings.recipe is not None:
        check_recipe(config, settings)
    return config


def check_recipe(config, settings):
    """What the case's geometry, precision mapping and refusals take as given
    is what the recipe leaves, as its architecture must be: the parallel
    sizes that make a replica, the virtual stages, offloading, CUDA graphs
    and Megatron-FSDP; a recipe that sets one otherwise is refused."""

    model, optimizer = config.model, config.optimizer
    assumed = {
        "tensor_model_parallel_size": (model, settings.tensor_model_parallel_size or 1),
        "pipeline_model_parallel_size": (
            model,
            settings.pipeline_model_parallel_size or 1,
        ),
        "context_parallel_size": (model, settings.context_parallel_size or 1),
        "virtual_pipeline_model_parallel_size": (
            model,
            settings.virtual_pipeline_model_parallel_size,
        ),
        "cpu_offloading": (model, bool(settings.cpu_offloading)),
        "fine_grained_activation_offloading": (
            model,
            bool(settings.fine_grained_activation_offloading),
        ),
        "cuda_graph_impl": (model, settings.cuda_graph_impl or "none"),
        "optimizer_cuda_graph": (optimizer, bool(settings.optimizer_cuda_graph)),
        "use_megatron_fsdp": (config.dist, bool(settings.use_megatron_fsdp)),
    }
    differs = {
        name: getattr(holder, name, None)
        for name, (holder, value) in assumed.items()
        if getattr(holder, name, None) != value
    }
    if differs:
        raise ValueError(
            f"recipe {settings.recipe} sets {differs}, which the case does not;"
            " name them in the case"
        )


def count_dtypes(values):
    counts = {}
    for value in values:
        if value is None:
            continue
        key = str(value.dtype).removeprefix("torch.")
        counts[key] = counts.get(key, 0) + value.numel()
    return counts


def optimizer_parts(optimizer):
    """Megatron's optimizers under its chained wrapper."""
    return list(getattr(optimizer, "chained_optimizers", None) or [optimizer])


def inner_optimizers(part):
    """The torch optimizers a Megatron optimizer steps through: its own, or
    the host and device ones its CPU-offload optimizer divides the step
    between."""

    inner = getattr(part, "optimizer", part)
    return list(getattr(inner, "sub_optimizers", None) or [inner])


def stepped_of(part):
    """What a Megatron optimizer updates, the masters: Transformer Engine's
    ``master_param`` state under the precision-aware optimizer (whose groups
    then hold the compute weights), and otherwise the values in its groups,
    which are the float32 copies of a mixed-precision or distributed
    optimizer, the parameters of a float32 model themselves, or
    Megatron-FSDP's main weights."""

    found = []
    for inner in inner_optimizers(part):
        masters = [
            state["master_param"]
            for state in inner.state.values()
            if state.get("master_param") is not None
        ]
        found.extend(
            masters
            or [value for group in inner.param_groups for value in group["params"]]
        )
    return found


def pinned_parameters(model, precision):
    """The trainable parameters the model holds at a dtype other than the
    contract's when it is built (norms' or state-space parameters a model
    keeps at float32), by name, with their dtype."""

    return {
        name: str(parameter.dtype).removeprefix("torch.")
        for chunk in model
        for name, parameter in chunk.named_parameters()
        if parameter.requires_grad
        and str(parameter.dtype).removeprefix("torch.") != precision.model_dtype
    }


def precision_evidence(model, optimizer, precision, pinned):
    """The dtypes the weights, their accumulated gradients, the masters and
    Adam's moments are held at after the first update, with the elements at
    each; a contradiction of the contract is refused. A parameter the model
    pinned at another dtype when it was built is counted apart."""

    named = [
        (name, parameter)
        for chunk in model
        for name, parameter in chunk.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for name, parameter in named if name not in pinned]
    parts = optimizer_parts(optimizer)
    states = [
        state
        for part in parts
        for inner in inner_optimizers(part)
        for state in inner.state.values()
    ]
    masters = count_dtypes(value for part in parts for value in stepped_of(part))
    # Transformer Engine's fused Adam keeps a float32 master of a bfloat16
    # weight as the weight itself and the master's low 16 bits
    # (store_param_remainders): an int16 state that, with the weight, is the
    # float32 master exactly.
    remainders = masters.pop("int16", 0)
    if remainders:
        masters["float32"] = masters.get("float32", 0) + remainders
    evidence = {
        "weights": count_dtypes(parameters),
        "pinned_weights": count_dtypes(
            parameter for name, parameter in named if name in pinned
        ),
        "accumulated_gradients": count_dtypes(
            getattr(parameter, "main_grad", None) for parameter in parameters
        ),
        "masters": masters,
        "master_remainders": remainders,
        "first_moments": count_dtypes(state.get("exp_avg") for state in states),
        "second_moments": count_dtypes(state.get("exp_avg_sq") for state in states),
    }
    resolved = precision.resolved()
    moments = resolved["opt_state_dtype"]
    moments = "float32" if moments == "default" else moments
    expected = {
        "weights": precision.model_dtype,
        "first_moments": moments,
        "second_moments": moments,
        "accumulated_gradients": resolved["grad_dtype"],
    }
    for key, dtype in expected.items():
        held = set(evidence[key])
        if held and held != {dtype}:
            raise ValueError(f"{key} are held at {sorted(held)}, not {dtype}")
    # What the optimizer steps is held at the masters' dtype, or at the
    # weights' where the contract keeps no master; a pinned parameter may be
    # stepped at its own.
    stepped = set(masters)
    allowed = {precision.stepped_dtype, *evidence["pinned_weights"]}
    if precision.stepped_dtype not in stepped or not stepped <= allowed:
        raise ValueError(
            f"the optimizer steps values at {sorted(stepped)}, "
            f"not {precision.stepped_dtype}"
        )
    return evidence


def callbacks(case, clock, record):
    """The runner's clock and the precision evidence, on Bridge's events."""

    from megatron.bridge.training.callbacks import Callback

    class Steps(Callback):
        def __init__(self):
            self.step = 0
            self.evidence = None

        def on_train_start(self, context):
            chunks = context.model
            self.pinned = pinned_parameters(chunks, case.precision)
            # The configuration as Bridge finalized it before its loop (mixed
            # precision, communication overlap, the distributed optimizer's
            # and FSDP's settings, Megatron-Core's own defaults), in place of
            # the one handed to it.
            record["megatron"] = configuration_record(context.state.cfg, case)
            record["megatron_parameters"] = sum(
                parameter.numel()
                for chunk in chunks
                for parameter in chunk.parameters()
            )
            record["megatron_trainable_parameters"] = sum(
                parameter.numel()
                for chunk in chunks
                for parameter in chunk.parameters()
                if parameter.requires_grad
            )
            case.save()
            clock.ready()

        def on_train_step_start(self, context):
            clock.begin(self.step)

        def on_train_step_end(self, context):
            loss = None
            value = (context.loss_dict or {}).get("lm loss")
            if value is not None:
                value = value.float() if torch.is_tensor(value) else value
                if torch.is_tensor(value) and value.numel() == 2:
                    value = value[0] / value[1]
                loss = float(value)
            clock.end(self.step, loss)
            if self.step == 0:
                self.evidence = precision_evidence(
                    context.model, context.optimizer, case.precision, self.pinned
                )
            self.step += 1

    return Steps()


def configuration_record(config, case):
    """What the run was configured with, by Megatron's names: the model
    provider's class, the architecture, the precision, every option where
    Bridge holds it, and the environment the case set."""

    sections = {
        "model": config.model,
        "optimizer": config.optimizer,
        "ddp": config.ddp,
        "dist": config.dist,
        "mixed": config.mixed_precision,
        "comm": config.comm_overlap,
    }
    fixed = {
        "model": (
            *ARCHITECTURE,
            "seq_length",
            "vocab_size",
            "make_vocab_size_divisible_by",
            "bf16",
            "params_dtype",
        ),
        "optimizer": (
            "main_params_dtype",
            "main_grads_dtype",
            "exp_avg_dtype",
            "exp_avg_sq_dtype",
        ),
        "ddp": ("grad_reduce_in_fp32",),
    }

    def plain(value):
        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
        if isinstance(value, bool | int | float | str) or value is None:
            return value
        if isinstance(value, list | tuple):
            return [plain(item) for item in value]
        return str(value)

    def read(section, name):
        holder = sections[section]
        return None if holder is None else plain(getattr(holder, name, None))

    return {
        "recipe": case.settings.recipe,
        "provider": type(config.model).__name__,
        "peft": None
        if config.peft is None
        else {
            key: plain(value) for key, value in dataclasses.asdict(config.peft).items()
        },
        **{
            section: {name: read(section, name) for name in names}
            for section, names in fixed.items()
        },
        "options": {
            option.name: read(option.section.split("+")[0], option.name)
            for option in OPTIONS
            if option.section != "environment"
        },
        "environment": case.resolved["environment"],
    }


def finalize_expert_parallel():
    """Release Transformer Engine's NCCL EP (the ncclep dispatcher's), if a
    case set it up, while the process group it borrows its communicator from
    still stands: the runner destroys the group as the case ends, and NCCL
    EP's own handler at interpreter shutdown, running after that, fails on
    the destroyed communicator and leaves the process unable to exit."""

    from transformer_engine.pytorch import ep

    if ep.is_ep_bootstrapped():
        ep.ep_finalize()


def train(case, clock):
    """Build Bridge's configuration for the case and run its loop."""

    from megatron.bridge.data.base import DatasetProvider
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    # What compiles kernels when first used (NCCL EP's dispatch, HybridEP's)
    # compiles with the environment's own toolkit and NCCL, as its builds
    # did, whatever the shell has: a shell's NCCL_HOME gave NCCL EP's kernels
    # another NCCL's headers.
    toolkit = Path(sys.prefix) / "cuda-home"
    if toolkit.is_dir():
        os.environ["CUDA_HOME"] = str(toolkit)
    nccl_home = Path(sys.prefix) / "nccl-home"
    if nccl_home.is_dir():
        os.environ["NCCL_HOME"] = str(nccl_home)
    else:
        os.environ.pop("NCCL_HOME", None)
    for name in [name for name in os.environ if name.startswith("NCCL_EP_JIT_")]:
        del os.environ[name]
    # Bridge keeps a process group it did not create, so the runner can
    # gather the ranks' results once its loop returns.
    dist.init_process_group("nccl", device_id=case.device)
    case.stage = "model_initialization"
    geometry = case.geometry
    vocabulary = int(case.resolved["huggingface_config"]["vocab_size"])
    replicas = geometry.data_parallel
    # The corpus the request records; Megatron's sampler decides which
    # replica's tokens this rank serves.
    case.data(vocabulary, replica=0, replicas=replicas)
    served = []

    @dataclasses.dataclass
    class Tokens(DatasetProvider):
        """The case's tokens, every engine's, for Bridge's loaders."""

        seq_length: int = geometry.sequence_length
        skip_getting_attention_mask_from_dataset: bool = True
        dataloader_type: str = "single"
        num_workers: int = 0

        def build_datasets(self, context):
            samples = Samples(
                case,
                context.train_samples,
                vocabulary,
                geometry.sequences_per_microbatch,
                replicas,
            )
            served.append(samples)
            return samples, None, None

    config = base_config(case, vocabulary)
    config = apply(config, case, vocabulary, Tokens())
    case.record["megatron"] = configuration_record(config, case)
    case.save()
    steps = callbacks(case, clock, case.record)
    # Bridge's setup runs inside its loop; the clock says when training
    # begins.
    case.stage = "setup"
    try:
        pretrain(config, forward_step, callbacks=[steps])
    finally:
        finalize_expert_parallel()
    streams = [stream for samples in served for stream in samples.streams.values()]
    if streams:
        # The corpus wrap count is the furthest any served replica read.
        case.tokens = max(streams, key=lambda stream: stream.wrapped)
    return runner.Outcome(
        evidence=steps.evidence,
        provenance={
            "megatron": case.record["megatron"],
            # The model Megatron built, which layouts are compared by.
            "parameters": case.record.get("megatron_parameters"),
            "trainable_parameters": case.record.get("megatron_trainable_parameters"),
        },
    )


def main(argv=None):
    runner.main("megatron_core", engine_settings, train, argv)


if __name__ == "__main__":
    main()
