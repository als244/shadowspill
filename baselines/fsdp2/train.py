"""One FSDP2 case of a catalog model, outside ShadowSpill: the trainer
``baselines.common.launch`` starts under torchrun.

The settings are ``baselines.fsdp2.settings``, validated by the launcher;
``baselines.common.runner`` writes the request, the result and the row
around ``train``, which builds the sharded model and runs every step under
the runner's clock.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)

from baselines.common import recompute_layers, runner
from baselines.fsdp2 import settings as engine_settings
from baselines.fsdp2.settings import configure_prefetch
from workloads.full_model import HEAD_LOSS_METRIC


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def check_precision(model, optimizer, stepped, pinned=()):
    """The trained shards, their gradients and AdamW's moments are at the
    stepped dtype, as the contract says, apart from the parameters the model
    pins at float32; counts are the evidence."""

    skipped = {id(parameter) for parameter in pinned}
    counts = {"parameters": 0, "gradient_elements": 0, "moment_elements": 0}
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in skipped:
            continue
        if parameter.dtype != stepped:
            raise ValueError(f"a trained shard is {parameter.dtype}, not {stepped}")
        counts["parameters"] += local_tensor(parameter).numel()
        if parameter.grad is not None:
            if parameter.grad.dtype != stepped:
                raise ValueError(f"a gradient shard is {parameter.grad.dtype}")
            counts["gradient_elements"] += local_tensor(parameter.grad).numel()
        for key in ("exp_avg", "exp_avg_sq"):
            value = optimizer.state[parameter].get(key)
            if value is not None:
                if value.dtype != stepped:
                    raise ValueError(f"an AdamW moment is {value.dtype}")
                counts["moment_elements"] += local_tensor(value).numel()
    if not counts["gradient_elements"] or not counts["moment_elements"]:
        raise ValueError("optimizer/gradient evidence is empty")
    return counts


def float8_filter(module, name):
    """Which linears Float8 training converts: every block linear whose
    dimensions the kernels take; the language-model head stays as it is."""

    if "lm_head" in name or "embed" in name:
        return False
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def build_optimizer(model, settings, precision, pinned=()):
    """AdamW over the trained parameters: PyTorch's fused implementation, on
    the device or, for CPU-offloaded shards, on the host; torchao's with
    stochastic rounding, its 8-bit or 4-bit variants, or its CPU-offloaded
    step. The pinned parameters, plain tensors beside the shards, step in a
    group of their own, since one fused call takes one kind of tensor."""

    skipped = {id(parameter) for parameter in pinned}
    parameters = [
        p for p in model.parameters() if p.requires_grad and id(p) not in skipped
    ]
    whole = [p for p in pinned if p.requires_grad]
    stochastic = precision.parameter_rounding == "stochastic"
    options = {
        "lr": settings.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": settings.weight_decay,
    }
    if settings.optimizer == "adamw" and not stochastic:
        groups = [{"params": parameters}]
        if whole:
            groups.append({"params": whole})
        return torch.optim.AdamW(groups, **options, fused=True)
    parameters.extend(whole)
    if settings.optimizer == "cpu-offload":
        from torchao.optim import CPUOffloadOptimizer

        return CPUOffloadOptimizer(
            parameters,
            torch.optim.AdamW,
            offload_gradients=settings.offload_gradients,
            fused=True,
            **options,
        )
    from torchao.optim import AdamW4bit, AdamW8bit, _AdamW

    variant = {"adamw": _AdamW, "adamw8bit": AdamW8bit, "adamw4bit": AdamW4bit}
    return variant[settings.optimizer](
        parameters, **options, bf16_stochastic_round=stochastic
    )


def train(case, clock):
    """Shard the catalog model with FSDP2 and run every step of the case."""

    settings, geometry, precision = case.settings, case.geometry, case.precision
    request = settings.request()
    record = case.record
    world, device = case.world, case.device
    if settings.float8:
        capability = torch.cuda.get_device_capability(device)
        if capability < (8, 9):
            raise RuntimeError(
                "torchao's Float8 training needs compute capability 8.9 or above;"
                f" this device has {capability[0]}.{capability[1]}"
            )
    stepped = getattr(torch, precision.stepped_dtype)
    model_dtype = getattr(torch, precision.model_dtype)
    dist.init_process_group("nccl", device_id=device)
    # MLOps imports may inspect CUDA; select this rank's device first.
    from baselines.fsdp2.model import (
        TrainingModel,
        canonical_name,
        initialize,
        materialize,
        pinned_parameters,
    )

    case.stage = "model_initialization"
    torch.manual_seed(settings.seed)
    model = TrainingModel(request, geometry, dtype=precision.stepped_dtype)
    blocks = model.model.blocks
    checkpointed = recompute_layers(len(blocks), settings.recompute)
    record["checkpointed_layers"] = checkpointed
    # Parameters the model keeps at float32 stay outside the sharding and
    # its casts: whole on every rank, materialized here since sharding's
    # to_empty leaves what it ignores alone, their gradients averaged by
    # hand.
    pinned_names = pinned_parameters(request)
    pinned = materialize(model.model, pinned_names, device)
    record["pinned_float32_parameters"] = sorted(pinned_names)
    case.save()
    if settings.float8:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        config = dataclasses.replace(
            Float8LinearConfig.from_recipe_name("tensorwise"),
            enable_fsdp_float8_all_gather=True,
        )
        convert_to_float8_training(
            model.model, config=config, module_filter_fn=float8_filter
        )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        offload_wrapper,
    )

    for index in range(len(blocks)):
        block = blocks[index]
        if index in checkpointed:
            block = checkpoint_wrapper(block, preserve_rng_state=False)
        if settings.activation_offload == "cpu":
            block = offload_wrapper(block)
        blocks[index] = block
    if settings.activation_offload == "overlapped":
        from baselines.fsdp2.offload_overlapped import blocks_context

        blocks_context(list(blocks), use_pin_memory=settings.pin_memory)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    policy = MixedPrecisionPolicy(
        param_dtype=model_dtype,
        reduce_dtype=getattr(torch, precision.grad_dtype or precision.model_dtype),
    )
    offload_policy = (
        CPUOffloadPolicy(pin_memory=settings.pin_memory)
        if settings.offload_policy == "cpu"
        else OffloadPolicy()
    )
    materialization_device = (
        torch.device("cpu") if settings.offload_policy == "cpu" else device
    )
    checkpoint = None
    if settings.checkpoint is not None:
        checkpoint = request.checkpoint(settings.checkpoint)
        record["checkpoint"] = {
            "directory": str(Path(settings.checkpoint).resolve()),
            "config": dataclasses.asdict(checkpoint.config),
        }
    ignored = set(pinned) or None
    for block in blocks:
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=policy,
            offload_policy=offload_policy,
            reshard_after_forward=settings.reshard_after_forward,
            ignored_params=ignored,
        )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=policy,
        offload_policy=offload_policy,
        reshard_after_forward=False,
        ignored_params=ignored,
    )
    model.to_empty(device=materialization_device)
    with torch.device(materialization_device):
        initialize(model)
    if checkpoint is not None:
        # Every rank reads the whole tensors and keeps its rows.
        record["loaded"] = model.load_checkpoint(checkpoint, rename=canonical_name)
        checkpoint.close()
        if record["loaded"]["unused"]:
            raise ValueError(
                "checkpoint keys the model did not take:"
                f" {record['loaded']['unused'][:5]}"
            )
    # Offloaded shards live on the host; rotary tables and other compute
    # buffers, and the pinned parameters, belong on the accelerator.
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            setattr(module, name, buffer.to(device))
    if any(parameter.device != device for parameter in pinned):
        values = [parameter.detach().to(device) for parameter in pinned]
        pinned = materialize(model.model, pinned_names, device)
        with torch.no_grad():
            for parameter, value in zip(pinned, values, strict=True):
                parameter.copy_(value)
    configure_prefetch(
        model,
        forward=settings.forward_prefetch,
        backward=settings.backward_prefetch,
    )
    model.train()
    if settings.compile_blocks:
        for block in blocks:
            block.compile()
    optimizer = build_optimizer(model, settings, precision, pinned)
    data = case.data(int(model.model.config.vocab_size), device=device)
    case.save()
    batch = geometry.tokens_per_microbatch
    microbatches = geometry.accumulation
    reference = None
    if settings.verify_reference:
        from baselines.fsdp2.reference import Access, Reference

        reference = Reference(
            model,
            request,
            geometry,
            precision,
            settings,
            device,
            world,
            access=Access(optimizer, device),
        )
    evidence = None
    clock.ready()
    for step in range(clock.steps):
        tokens, targets = data(step)
        expected_loss = (
            None if reference is None else reference.step(model, tokens, targets, batch)
        )
        clock.begin(step)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        head_sum = torch.zeros((), dtype=torch.float32, device=device)
        for index in range(microbatches):
            last = index == microbatches - 1
            model.set_is_last_backward(last)
            model.set_requires_gradient_sync(
                settings.gradient_sync == "per-microbatch" or last
            )
            # Do not leave stale gathered copies across an optimizer update.
            model.set_reshard_after_backward(settings.reshard_after_backward or last)
            start = index * batch
            loss, metrics = model(
                tokens[:, start : start + batch], targets[:, start : start + batch]
            )
            loss.backward()
            loss_sum += loss.detach().float()
            head_sum += metrics.get(HEAD_LOSS_METRIC, loss.detach()).float()
        if world > 1:
            for parameter in pinned:
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
        optimizer.step()
        if settings.float8:
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

            precompute_float8_dynamic_scale_for_fsdp(model)
        clock.end(step, loss_sum, head_sum)
        if reference is not None:
            reference.check(model, optimizer, loss_sum, expected_loss)
        if step == 0 and settings.optimizer == "adamw":
            evidence = check_precision(model, optimizer, stepped, pinned)
    return runner.Outcome(
        evidence=evidence,
        reference_checks=None if reference is None else reference.checks,
        provenance={"checkpointed_layers": checkpointed},
    )


def main(argv=None):
    runner.main("fsdp2", engine_settings, train, argv)


if __name__ == "__main__":
    main()
