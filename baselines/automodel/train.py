"""One NeMo AutoModel case: NVIDIA's implementation of the catalog model's
architecture, trained by AutoModel's fine-tuning recipe and its own loop
(``run_train_validation_loop``); the trainer ``baselines.common.launch``
starts under torchrun in the nvidia-training environment.

The recipe's sections come from the shared settings and the case's AutoModel
options: the model from the case's Hugging Face configuration (or a named
checkpoint), the distributed strategy and its sizes, the loss, the optimizer
the precision contract maps to, and a step scheduler over the tokens every
harness trains on. The runner's clock times each optimizer update inside
AutoModel's own step, ``_run_train_optim_step``, with gradient clipping off
as every harness steps. Once the recipe is set up, the request records what
AutoModel built: the model's class, its head, its attention, its backend and
the loss.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.automodel import data as harness_data
from baselines.automodel import settings as harness_settings
from baselines.automodel.settings import OPTIONS, parse_list
from baselines.common import runner
from baselines.common.precision import Unsupported

#: The recipe classes behind the harness's names.
LOSSES = {
    "chunked": "nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy",
    "masked": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy",
    "fused_linear": "nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy",
    "te_parallel": (
        "nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy"
    ),
}
OPTIMIZERS = {
    "adamw": "torch.optim.AdamW",
    "adam": "torch.optim.Adam",
    "fused_adam": "transformer_engine.pytorch.optimizers.FusedAdam",
    "flash_adamw": "nemo_automodel.components.optim.optimizer.FlashAdamWConfig",
}


def huggingface_config(value, attention):
    """A ``transformers`` configuration from its dictionary, attending by
    ``attention``: AutoModel's own classes read it from the configuration,
    which ``from_config`` does not set."""

    from transformers import AutoConfig

    value = dict(value)
    return AutoConfig.for_model(
        value.pop("model_type"), attn_implementation=attention, **value
    )


def dtype_of(name):
    return None if name is None else getattr(torch, name)


def section(config, dotted):
    """The nested dictionary at ``dotted``, made where missing."""
    node = config
    for part in dotted.split("."):
        node = node.setdefault(part, {})
    return node


def recipe_config(case, mechanism, vocabulary, attention, own, targets=()):
    """AutoModel's recipe configuration for the case; ``own`` is whether
    AutoModel builds its own class of the architecture, which takes a
    backend, rather than Hugging Face's, which does not, and ``targets`` the
    modules LoRA adapts."""

    settings, geometry = case.settings, case.geometry
    sequences = geometry.tokens_per_step // geometry.sequence_length
    steps = settings.warmup_steps + settings.steps
    model = {
        "torch_dtype": dtype_of(mechanism["storage"]),
        "attn_implementation": attention,
    }
    if settings.checkpoint is not None:
        model["_target_"] = "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"
        model["pretrained_model_name_or_path"] = settings.checkpoint
    else:
        model["_target_"] = "nemo_automodel.NeMoAutoModelForCausalLM.from_config"
        # The configuration from its file, as AutoModel's recipes build it,
        # attending by ``attention``: AutoModel's own classes read it from the
        # configuration, which from_config leaves unset. AutoModel initializes
        # "te" through SDPA and injects Transformer Engine's attention after.
        model["config"] = {
            "_target_": "transformers.AutoConfig.from_pretrained",
            "pretrained_model_name_or_path": str(case.directory / "huggingface"),
            "attn_implementation": "sdpa" if attention == "te" else attention,
        }
    strategy = mechanism["strategy"]
    distributed = {"strategy": strategy}
    if settings.recompute is not None:
        distributed["activation_checkpointing"] = bool(settings.recompute)
    if strategy == "fsdp2":
        distributed["mp_policy"] = {
            "param_dtype": dtype_of(mechanism["mp_param_dtype"]),
            "reduce_dtype": dtype_of(mechanism["mp_reduce_dtype"]),
            "output_dtype": dtype_of(settings.mp_output_dtype),
            "cast_forward_inputs": False,
        }
        if settings.cpu_offload:
            distributed["offload_policy"] = {
                "_target_": "torch.distributed.fsdp.CPUOffloadPolicy",
                "pin_memory": settings.offload_pin_memory is not False,
            }
    for name in ("preserve_fp32_weights", "grad_reduce_in_fp32"):
        if mechanism[name] is not None:
            distributed[name] = mechanism[name]
    config = {
        "seed": settings.seed,
        "dist_env": {"backend": "nccl", "timeout_minutes": 30},
        "model": model,
        "distributed": distributed,
        "loss_fn": loss_config(settings),
        "optimizer": optimizer_config(settings, mechanism),
        "step_scheduler": {
            "global_batch_size": sequences,
            "local_batch_size": geometry.sequences_per_microbatch,
            "max_steps": steps,
            "num_epochs": 1,
            "ckpt_every_steps": steps + 1,
            "val_every_steps": None,
            "preemption_signal": None,
        },
        "checkpoint": {
            "enabled": False,
            "checkpoint_dir": str(case.directory / "automodel"),
        },
        # The case's tokens, every harness's, as AutoModel's datasets serve
        # batches; no tokenizer, which the recipe would otherwise load from
        # the model's path.
        "dataset": {
            "_target_": "baselines.automodel.data.CaseTokens",
            "steps": steps,
            "tokenizer": None,
        },
        "dataloader": {"_target_": "torch.utils.data.DataLoader", "batch_size": None},
    }
    # AutoModel's options, under their own names where the case sets them.
    for option in OPTIONS:
        value = getattr(settings, option.name)
        if value is None or option.section == "harness":
            continue
        if option.name == "activation_checkpointing":
            value = {"none": False, "full": True}.get(value, value)
        elif option.form == "names":
            value = parse_list(value).split(",")
        section(config, option.section)[option.field] = value
    named = [
        option.name
        for option in OPTIONS
        if getattr(settings, option.name) is not None
        and (
            option.section.startswith("model.backend")
            or option.name in ("linear", "te_fp8", "cuda_graph_modules")
        )
    ]
    if named and not own:
        raise ValueError(
            "AutoModel builds Hugging Face's class of this architecture, which"
            f" takes no backend options; not {', '.join(named)}"
        )
    if own and (named or case.resolved["linear"] is not None):
        section(config, "model.backend").update(
            {
                "_target_": "nemo_automodel.components.models.common.BackendConfig",
                **backend_options(settings, case.resolved),
            }
        )
    if (settings.pp_size or 1) > 1:
        # AutoModel needs the section whenever it pipelines; its defaults
        # apply to what the case does not name.
        section(config, "distributed.pipeline")
    if settings.sdpa_method is not None:
        config["sdpa_method"] = parse_list(settings.sdpa_method).split(",")
    if settings.fp8:
        section(config, "fp8")["enabled"] = True
    if settings.compile:
        section(config, "compile")["enabled"] = True
    if settings.mode == "lora":
        options = settings.lora
        section(config, "peft").update(
            {
                "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
                "target_modules": list(targets),
                "dim": int(options.get("rank", 16)),
                "alpha": options.get("alpha", 16),
                "lora_dtype": dtype_of(options.get("factor_dtype")),
            }
        )
        section(config, "peft").setdefault("dropout", 0.0)
    return config


def backend_options(settings, resolved):
    """The fields of AutoModel's BackendConfig the case sets, the harness's
    linear layers beside the chunked loss among them."""

    value = {
        option.field: getattr(settings, option.name)
        for option in OPTIONS
        if option.section == "model.backend"
        and getattr(settings, option.name) is not None
    }
    mok = {
        option.field: getattr(settings, option.name)
        for option in OPTIONS
        if option.section == "model.backend.mok"
        and getattr(settings, option.name) is not None
    }
    if mok:
        value["mok"] = mok
    if resolved["linear"] is not None:
        value["linear"] = resolved["linear"]
    if settings.te_fp8 is not None:
        value["te_fp8"] = {"recipe": settings.te_fp8}
    if settings.cuda_graph_modules is not None:
        value["cuda_graph"] = {
            "modules": parse_list(settings.cuda_graph_modules).split(",")
        }
    return value


def lora_targets(settings, hf_config, own, backend):
    """The modules the shared LoRA adapts, by name, in the class AutoModel
    builds: every linear layer but the head, and the routed experts of each
    mixture block; inside a mixture block nothing else unless
    ``shared_experts`` adds its linear layers other than its router. The
    class is built on the meta device, which holds no weights."""

    from nemo_automodel._transformers.model_init import (
        _filter_kwargs_for_init,
        _resolve_custom_model_cls_for_config,
    )
    from nemo_automodel.components._peft.module_matcher import _is_linear_module
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.moe.layers import MoE

    with torch.device("meta"):
        if own:
            cls = _resolve_custom_model_cls_for_config(hf_config)
            model = cls(
                hf_config,
                **_filter_kwargs_for_init(cls, {"backend": BackendConfig(**backend)}),
            )
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_config(hf_config)
    modules = dict(model.named_modules())
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if getter is not None else None
    if head is None:
        head = modules.get("lm_head")
    blocks = [name for name, module in modules.items() if isinstance(module, MoE)]
    shared = bool((settings.lora or {}).get("shared_experts", False))

    def block_of(name):
        return next((block for block in blocks if name.startswith(block + ".")), None)

    targets = []
    for name, module in modules.items():
        if name in blocks:
            targets.append(f"{name}.experts")
        elif _is_linear_module(module) and module is not head:
            block = block_of(name)
            if block is None or (
                shared and module is not getattr(modules[block], "gate", None)
            ):
                targets.append(name)
    return targets


def loss_config(settings):
    kind = settings.loss_kind()
    value = {"_target_": LOSSES[kind], "reduction": "sum"}
    if kind == "chunked":
        if settings.chunk_len is not None:
            value["chunk_len"] = settings.chunk_len
        if settings.chunked_compile is not None:
            value["compile"] = settings.chunked_compile
    elif kind == "fused_linear" and settings.fused_linear_impl is not None:
        value["impl"] = settings.fused_linear_impl
    elif kind == "masked" and settings.masked_fp32_upcast is not None:
        value["fp32_upcast"] = settings.masked_fp32_upcast
    return value


def optimizer_config(settings, mechanism):
    """AutoModel's optimizer section: Adam at the shared hyperparameters,
    with the masters and the moments the contract maps to."""

    kind = mechanism["optimizer"]
    value = {
        "_target_": OPTIMIZERS[kind],
        "lr": settings.learning_rate,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": settings.weight_decay,
    }
    if kind in ("adamw", "adam"):
        if settings.adamw_fused is not None:
            value["fused"] = settings.adamw_fused
    elif kind == "fused_adam":
        value["adam_w_mode"] = True
        value["master_weights"] = mechanism["master"] is not None
        if mechanism["master"] is not None:
            value["master_weight_dtype"] = dtype_of(mechanism["master"])
            if settings.store_param_remainders is not None:
                value["store_param_remainders"] = settings.store_param_remainders
        value["exp_avg_dtype"] = dtype_of(mechanism["moments"])
        value["exp_avg_sq_dtype"] = dtype_of(mechanism["moments"])
    elif kind == "flash_adamw" and settings.master_weight_bits is not None:
        value["master_weight_bits"] = settings.master_weight_bits
    return value


def count_dtypes(values):
    counts = {}
    for value in values:
        if value is None:
            continue
        local = value.to_local() if hasattr(value, "to_local") else value
        key = str(value.dtype).removeprefix("torch.")
        counts[key] = counts.get(key, 0) + local.numel()
    return counts


def precision_evidence(parameters, gradients, optimizers, mechanism, pinned=()):
    """The dtypes the weights, their gradients, the masters and Adam's
    moments are held at after the first update, with the elements at each;
    a contradiction of the mechanism is refused. ``pinned`` are the ids of
    the parameters the model itself holds at float32 inside a narrower
    model, counted apart."""

    states = [
        state
        for optimizer in optimizers
        for parameter, state in optimizer.state.items()
        if id(parameter) not in pinned
    ]
    masters = count_dtypes(state.get("master_param") for state in states)
    # Transformer Engine's remainders: a float32 master held as its bfloat16
    # weight and the 16 bits the weight drops.
    if "int16" in masters:
        masters["float32"] = masters.get("float32", 0) + masters.pop("int16")
    stepped = count_dtypes(
        value
        for optimizer in optimizers
        for group in optimizer.param_groups
        for value in group["params"]
        if id(value) not in pinned
    )
    evidence = {
        "mechanism": mechanism,
        "stepped": stepped,
        "weights": count_dtypes(p for p in parameters if id(p) not in pinned),
        "pinned_float32": sum(p.numel() for p in parameters if id(p) in pinned),
        "gradients": gradients,
        "masters": masters,
        "first_moments": count_dtypes(state.get("exp_avg") for state in states),
        "second_moments": count_dtypes(state.get("exp_avg_sq") for state in states),
    }
    expected = {
        "first_moments": mechanism["moments"],
        "second_moments": mechanism["moments"],
    }
    if mechanism["preserve_fp32_weights"] is None:
        expected["weights"] = mechanism["storage"]
        expected["gradients"] = mechanism["storage"]
    else:
        # Megatron-FSDP steps main weights and main gradients of its own: at
        # float32 where the contract keeps them so, else the model's dtype. A
        # parameter the model pins at float32 is stepped at its own dtype.
        model = mechanism["storage"]
        main = "float32" if mechanism["preserve_fp32_weights"] else model
        held = set(stepped)
        if main not in held or not held <= {main, *(["float32"] if pinned else [])}:
            raise ValueError(
                f"the optimizer steps values at {sorted(held)}, not {main}"
            )
        expected["gradients"] = "float32" if mechanism["grad_reduce_in_fp32"] else model
    if mechanism["master"] is not None and not mechanism["preserve_fp32_weights"]:
        expected["masters"] = mechanism["master"]
        if not masters:
            raise ValueError("the contract keeps masters and the optimizer holds none")
    for key, dtype in expected.items():
        held = set(evidence[key])
        if held and held != {dtype}:
            raise ValueError(f"{key} are held at {sorted(held)}, not {dtype}")
    return evidence


def plain(value):
    """A recipe configuration value as JSON holds it."""

    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    return str(value)


def built(model, loss):
    """What AutoModel built: the model's class, its head, its attention,
    its backend and the loss, inside DDP's or Megatron-FSDP's wrapper."""

    while not hasattr(model, "config") and hasattr(model, "module"):
        model = model.module
    head = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else None
    )
    if head is None:
        head = getattr(model, "lm_head", None)
    config = getattr(model, "config", None)
    backend = getattr(model, "backend", None)
    return {
        "model": f"{type(model).__module__}.{type(model).__qualname__}",
        "head": None
        if head is None
        else f"{type(head).__module__}.{type(head).__qualname__}",
        "attention": getattr(config, "_attn_implementation", None),
        "backend": None if backend is None else plain(backend),
        "loss": type(loss).__name__,
    }


def train(case, clock):
    """Build AutoModel's recipe for the case and run its loop."""

    from nemo_automodel._transformers.kernel_patches import DEFAULT_ATTN_IMPLEMENTATION
    from nemo_automodel._transformers.model_init import (
        _resolve_custom_model_cls_for_config,
    )
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.llm.train_ft import (
        TrainFinetuneRecipeForNextTokenPrediction,
    )

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
    # AutoModel keeps a process group it did not create, so the runner can
    # gather the ranks' results once its loop returns.
    dist.init_process_group("nccl", device_id=case.device)
    case.stage = "model_initialization"
    settings, geometry = case.settings, case.geometry
    mechanism = case.resolved["precision_mechanism"]
    vocabulary = int(case.resolved["huggingface_config"]["vocab_size"])
    # AutoModel's declared default where the case names none, on both paths.
    attention = settings.attn_implementation or DEFAULT_ATTN_IMPLEMENTATION
    hf_config = huggingface_config(case.resolved["huggingface_config"], "sdpa")
    own = not settings.force_hf and (
        _resolve_custom_model_cls_for_config(hf_config) is not None
    )
    if case.resolved["mixture"] and not own:
        raise Unsupported(
            "AutoModel builds Hugging Face's class of this mixture, which trains no"
            " balancing loss (the recipe takes the labels before the forward, and"
            " Hugging Face adds the term only to a loss it computes from them) and"
            " whose fused experts AutoModel's LoRA does not adapt; the shared"
            " objective has both"
        )
    targets = ()
    if settings.mode == "lora":
        targets = lora_targets(
            settings, hf_config, own, backend_options(settings, case.resolved)
        )
        case.record["automodel_lora_targets"] = targets
    if settings.checkpoint is None:
        directory = case.directory / "huggingface"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps(case.resolved["huggingface_config"], indent=2)
        )
    config = recipe_config(case, mechanism, vocabulary, attention, own, targets)
    case.record["automodel"] = plain(config)
    case.save()
    loss_class = LOSSES[case.resolved["loss"]].rsplit(".", 1)[1]
    seen = {"step": 0, "evidence": None, "peaks": {}, "gradients": {}, "pinned": set()}

    class Recipe(TrainFinetuneRecipeForNextTokenPrediction):
        def setup(self):
            super().setup()
            replicas = self._get_dp_group_size()
            if replicas != geometry.data_parallel:
                raise ValueError(
                    f"AutoModel's data-parallel ranks ({replicas}) are not the"
                    f" geometry's ({geometry.data_parallel})"
                )
            what = built(self.model_parts[0], self.loss_fn)
            case.record["automodel_built"] = what
            if targets:
                # AutoModel adapts a linear in place, with lora_A and lora_B
                # modules, and routed experts with lora_ parameters of their
                # own; a target it matched to nothing it passes over.
                adapted = {
                    name.removeprefix("module.")
                    for part in self.model_parts
                    for name, module in part.named_modules()
                    if any(
                        child.startswith("lora_")
                        for child, _ in [
                            *module.named_children(),
                            *module.named_parameters(recurse=False),
                        ]
                    )
                }
                missed = [name for name in targets if name not in adapted]
                if missed:
                    raise ValueError(
                        f"AutoModel adapted none of {missed[:4]}"
                        f"{' and more' if len(missed) > 4 else ''}, which name"
                        " modules of the model the shared LoRA selection takes"
                    )
            parameters = [
                parameter
                for part in self.model_parts
                for parameter in part.parameters()
            ]
            case.record["automodel_parameters"] = sum(p.numel() for p in parameters)
            case.record["automodel_trainable_parameters"] = sum(
                p.numel() for p in parameters if p.requires_grad
            )
            # The parameters the model itself holds at float32 inside a
            # narrower model, as Qwen 3.5's gated delta net its A_log and
            # dt_bias: the evidence counts them apart.
            pinned = sorted(
                name
                for part in self.model_parts
                for name, parameter in part.named_parameters()
                if mechanism["storage"] != "float32"
                and parameter.dtype == torch.float32
            )
            seen["pinned"] = {
                id(parameter)
                for part in self.model_parts
                for name, parameter in part.named_parameters()
                if name in pinned
            }
            case.record["automodel_pinned_float32"] = pinned
            case.save()
            if what["loss"] != loss_class:
                raise ValueError(
                    f"AutoModel replaced the case's {loss_class} with"
                    f" {what['loss']}; a case names a loss the model takes"
                )

            def gradients(optimizer, *_):
                if seen["step"] == 0 and not seen["gradients"]:
                    seen["gradients"] = count_dtypes(
                        parameter.grad
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                        if id(parameter) not in seen["pinned"]
                    )

            for optimizer in self.optimizer:
                optimizer.register_step_pre_hook(gradients)
            clock.ready()

        def _run_train_optim_step(self, batches, max_grad_norm=None):
            step = seen["step"]
            clock.begin(step)
            # No clipping, as every harness steps.
            sample = super()._run_train_optim_step(batches, None)
            clock.end(step, sample.metrics["loss"])
            if clock.measured(step):
                # AutoModel resets the allocator's peak as it logs each update.
                for name, value in (
                    ("peak_allocated_bytes", torch.cuda.max_memory_allocated()),
                    ("peak_reserved_bytes", torch.cuda.max_memory_reserved()),
                ):
                    seen["peaks"][name] = max(seen["peaks"].get(name, 0), value)
            if step == 0:
                parameters = [
                    parameter
                    for part in self.model_parts
                    for parameter in part.parameters()
                    if parameter.requires_grad
                ]
                seen["evidence"] = precision_evidence(
                    parameters,
                    seen["gradients"],
                    self.optimizer,
                    mechanism,
                    seen["pinned"],
                )
            seen["step"] += 1
            return sample

    harness_data.serve(case, vocabulary)
    recipe = Recipe(ConfigNode(config))
    case.stage = "setup"
    recipe.setup()
    case.stage = "training"
    recipe.run_train_validation_loop()
    # The corpus and the replica the dataset served.
    case.save()
    return runner.Outcome(
        evidence=seen["evidence"],
        peaks=seen["peaks"] or None,
        provenance={
            "automodel": case.record["automodel"],
            "automodel_built": case.record["automodel_built"],
            "parameters": case.record.get("automodel_parameters"),
            "trainable_parameters": case.record.get("automodel_trainable_parameters"),
        },
    )


def main(argv=None):
    runner.main("automodel", harness_settings, train, argv)


if __name__ == "__main__":
    main()
