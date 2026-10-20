"""One FSDP2 case of a catalog model; launch with torchrun, outside ShadowSpill.

    torchrun --standalone --nproc-per-node=N -m baselines.fsdp2.train \\
        --output DIR --model NAME --tokens-per-microbatch T \\
        --tokens-per-step S [--mode full|lora] ...

The settings are ``baselines.fsdp2.settings``; what the harness cannot honor
is refused before the process group exists. Every rank writes
``rank-NNNNN/request.json`` as it starts and ``result.json`` or
``failure.json`` as it ends; the first rank writes ``results.json`` with
every rank's measurements and ``row.json``, the result row every harness
shares.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import math
import os
import resource
import sys
import time
import traceback
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

from baselines.common import recompute_layers, summarize_cycles, write_json
from baselines.common.data import Corpus, synthetic_tokens
from baselines.common.models import useful_flops_per_token
from baselines.common.provenance import (
    environment_identity,
    hardware_identity,
    source_identity,
)
from baselines.common.results import memory_peaks
from baselines.common.results import row as result_row
from baselines.fsdp2.settings import (
    SOURCE_PATHS,
    add_arguments,
    configure_prefetch,
    from_namespace,
)
from workloads.full_model import HEAD_LOSS_METRIC


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    add_arguments(parser)
    return parser.parse_args(argv)


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
    """AdamW over the trained parameters: fused torch, torchao's with
    stochastic rounding, its 8-bit or 4-bit variants, or its CPU-offloaded
    step; on CPU-offloaded shards the single-tensor torch implementation.
    The pinned parameters, plain tensors beside the shards, step in a group
    of their own since a fused step takes one kind."""

    skipped = {id(parameter) for parameter in pinned}
    parameters = [
        p for p in model.parameters() if p.requires_grad and id(p) not in skipped
    ]
    whole = [p for p in pinned if p.requires_grad]
    on_cpu = settings.offload == "cpu"
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
            groups.append({"params": whole, "fused": False, "foreach": False})
        return torch.optim.AdamW(
            groups,
            **options,
            fused=not on_cpu,
            foreach=False if on_cpu else None,
        )
    parameters.extend(whole)
    if settings.optimizer == "cpu-offload":
        from torchao.optim import CPUOffloadOptimizer

        return CPUOffloadOptimizer(parameters, torch.optim.AdamW, fused=True, **options)
    from torchao.optim import AdamW4bit, AdamW8bit, _AdamW

    variant = {"adamw": _AdamW, "adamw8bit": AdamW8bit, "adamw4bit": AdamW4bit}
    return variant[settings.optimizer](
        parameters, **options, bf16_stochastic_round=stochastic
    )


def main(argv=None):
    args = parse(argv)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    settings = from_namespace(args)
    geometry = settings.validate(world)
    request = settings.request()
    precision = settings.precision()
    stepped = getattr(torch, precision.stepped_dtype)
    model_dtype = getattr(torch, precision.model_dtype)
    torch.set_num_threads(settings.cpu_threads)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    output = args.output / f"rank-{rank:05d}"
    output.mkdir(parents=True, exist_ok=False)
    stage = "distributed_initialization"
    description = settings.describe(world)
    record = {
        **description,
        "output": str(args.output),
        "rank": rank,
        "world_size": world,
        "hardware": hardware_identity(),
        "resolved_precision": precision.resolved(),
        "microbatches_per_step": geometry.accumulation,
        "shadowspill": False,
    }
    write_json(output / "request.json", record)
    try:
        dist.init_process_group("nccl", device_id=device)
        # MLOps imports may inspect CUDA; select this rank's device first.
        from baselines.fsdp2.model import (
            TrainingModel,
            canonical_name,
            initialize,
            materialize,
            pinned_parameters,
        )

        stage = "model_initialization"
        if settings.allocator_cap_gib is not None:
            total = torch.cuda.get_device_properties(device).total_memory
            fraction = settings.allocator_cap_gib * 2**30 / total
            if not 0 < fraction <= 1:
                raise ValueError("allocator cap must fit the physical device")
            torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.manual_seed(settings.seed)
        model = TrainingModel(request, geometry, dtype=precision.stepped_dtype)
        useful = (
            useful_flops_per_token(model.model, geometry.sequence_length)
            * geometry.tokens_per_step
        )
        blocks = model.model.blocks
        checkpointed = recompute_layers(len(blocks), settings.recompute)
        record["checkpointed_layers"] = checkpointed
        record["summary"] = request.summary(model.model)
        # Parameters the model keeps at float32 stay outside the sharding and
        # its casts: whole on every rank, materialized here since sharding's
        # to_empty leaves what it ignores alone, their gradients averaged by
        # hand.
        pinned_names = pinned_parameters(request)
        pinned = materialize(model.model, pinned_names, device)
        record["pinned_float32_parameters"] = sorted(pinned_names)
        write_json(output / "request.json", record)
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
            if settings.offload == "cpu"
            else OffloadPolicy()
        )
        materialization_device = (
            torch.device("cpu") if settings.offload == "cpu" else device
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
            record["loaded"] = checkpoint.load_into(model.model, rename=canonical_name)
        if checkpoint is not None:
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
        vocabulary = int(model.model.config.vocab_size)
        corpus = None
        if settings.data is not None:
            corpus = Corpus(settings.data)
            corpus.check_vocabulary(vocabulary)
            record["corpus"] = corpus.meta
        tokens_per_rank = geometry.tokens_per_rank_per_step
        if corpus is None:
            tokens, targets = synthetic_tokens(
                vocabulary, tokens_per_rank, seed=settings.seed, rank=rank
            )
            tokens, targets = tokens.to(device), targets.to(device)
        write_json(output / "request.json", record)
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
        cycles, losses, head_losses, wrapped = [], [], [], 0
        evidence = None
        stage = "training"
        setup_seconds = time.perf_counter() - started
        print(
            f"rank={rank} initialized seconds={setup_seconds:.3f} "
            f"accumulation={microbatches}",
            flush=True,
        )
        for step in range(settings.warmup_steps + settings.steps):
            if corpus is not None:
                tokens, targets, wrapped = corpus.window(
                    rank=rank, world_size=world, step=step, tokens=tokens_per_rank
                )
                tokens, targets = tokens.to(device), targets.to(device)
            expected_loss = (
                None
                if reference is None
                else reference.step(model, tokens, targets, batch)
            )
            measured = step >= settings.warmup_steps
            if step == settings.warmup_steps:
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                if settings.nsys_capture and rank == 0:
                    torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.synchronize()
            dist.barrier()
            tick = time.perf_counter()
            torch.cuda.nvtx.range_push(f"baseline.fsdp2/rank-{rank}/step-{step + 1}")
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
                model.set_reshard_after_backward(
                    settings.reshard_after_backward or last
                )
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
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            if not math.isfinite(loss_sum.item()):
                raise ValueError("training loss is not finite")
            if reference is not None:
                reference.check(model, optimizer, loss_sum, expected_loss)
            if measured:
                cycles.append(elapsed)
                losses.append(float(loss_sum.item()))
                head_losses.append(float(head_sum.item()))
            if step == 0 and settings.optimizer == "adamw":
                evidence = check_precision(model, optimizer, stepped, pinned)
            print(
                f"rank={rank} step={step + 1} measured={measured} "
                f"seconds={elapsed:.6f} loss={loss_sum.item():.6f}",
                flush=True,
            )
        if settings.nsys_capture:
            dist.barrier()
            if rank == 0:
                torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()
        forbidden = [
            name
            for name in sys.modules
            if name == "shadowspill" or name.startswith("shadowspill.")
        ]
        if forbidden:
            raise RuntimeError(
                f"standalone baseline imported ShadowSpill: {forbidden[:5]}"
            )
        performance = (
            reference is None
            and not settings.nsys_capture
            and request.entry.size not in ("tiny", "numerical")
        )
        result = {
            "status": "completed",
            "rank": rank,
            "cycles": cycles,
            "losses": losses,
            "head_losses": head_losses,
            "precision_evidence": evidence,
            "corpus_wrapped": wrapped,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "setup_seconds": setup_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "finished_at": datetime.datetime.now().astimezone().isoformat(),
            "qualified_against_reference": reference is not None,
            "reference_checks": None if reference is None else reference.checks,
            "performance_result": performance,
        }
        write_json(output / "result.json", result)
        all_ranks = [None] * world
        dist.all_gather_object(all_ranks, result)
        if rank == 0:
            summary = summarize_cycles(
                [row["cycles"] for row in all_ranks],
                global_tokens=geometry.tokens_per_step,
                useful_flops_per_step=useful,
            )
            summary["performance_result"] = performance
            summary["losses"] = [row["losses"] for row in all_ranks]
            summary["head_losses"] = [row["head_losses"] for row in all_ranks]
            write_json(
                args.output / "results.json",
                {
                    **summary,
                    "status": "completed",
                    "request": record,
                    "ranks": all_ranks,
                    "useful_flops_per_step": useful,
                    "flops_convention": "baselines.common.models"
                    ".useful_flops_per_token",
                },
            )
            write_json(
                args.output / "row.json",
                result_row(
                    harness="fsdp2",
                    status="completed",
                    **description,
                    hardware=record["hardware"],
                    provenance={
                        "source": source_identity(SOURCE_PATHS),
                        "environment": environment_identity(sys.executable),
                        "checkpointed_layers": checkpointed,
                        "model_summary": record["summary"],
                        "corpus": record.get("corpus"),
                        "checkpoint": record.get("checkpoint"),
                    },
                    measurement=summary,
                    memory=memory_peaks(all_ranks),
                ),
            )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "rank": rank,
                "stage": stage,
                "status": "cuda_out_of_memory"
                if isinstance(error, torch.OutOfMemoryError)
                else "failed",
                "exception_type": type(error).__name__,
                "error": traceback.format_exc(),
            },
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
