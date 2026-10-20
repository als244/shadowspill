"""One standalone FSDP2 case; launch with torchrun, outside ShadowSpill."""

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
    Settings,
    add_arguments,
    configure_prefetch,
)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_arguments(parser)
    return parser.parse_args(argv)


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def check_precision(model, optimizer):
    counts = {"parameters": 0, "gradient_elements": 0, "moment_elements": 0}
    for parameter in model.parameters():
        if parameter.dtype != torch.float32:
            raise ValueError("FSDP sharded master parameter is not FP32")
        counts["parameters"] += local_tensor(parameter).numel()
        if parameter.grad is not None:
            if parameter.grad.dtype != torch.float32:
                raise ValueError("FSDP accumulated gradient shard is not FP32")
            counts["gradient_elements"] += local_tensor(parameter.grad).numel()
        for key in ("exp_avg", "exp_avg_sq"):
            value = optimizer.state[parameter].get(key)
            if value is not None:
                if value.dtype != torch.float32:
                    raise ValueError("AdamW moment is not FP32")
                counts["moment_elements"] += local_tensor(value).numel()
    if not counts["gradient_elements"] or not counts["moment_elements"]:
        raise ValueError("optimizer/gradient evidence is empty")
    return counts


def main(argv=None):
    args = parse(argv)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    settings = Settings.from_dict(
        {key: value for key, value in vars(args).items() if key != "output"}
    )
    microbatches = settings.validate(world)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    output = args.output / f"rank-{rank:05d}"
    output.mkdir(parents=True, exist_ok=False)
    stage = "distributed_initialization"
    request = {
        **dataclasses.asdict(settings),
        "output": str(args.output),
        "rank": rank,
        "world_size": world,
        "microbatches_per_step": microbatches,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "hardware": hardware_identity(),
        "compute_dtype": "bfloat16",
        "parameter_shard_dtype": "float32",
        "gradient_reduction_and_accumulation_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "optimizer_device": "cpu" if args.offload == "cpu" else "cuda",
        "optimizer_implementation": "single-tensor"
        if args.offload == "cpu"
        else "fused",
        "root_reshard_after_forward": False,
        "expert_parallel_size": 1,
        "routing": "real_topk",
        "shadowspill": False,
        "gradient_note": (
            "Native FSDP2: BF16 autograd gradients are converted to FP32 "
            "before reduction and accumulation."
        ),
    }
    write_json(output / "request.json", request)
    try:
        dist.init_process_group("nccl", device_id=device)
        # MLOps imports may inspect CUDA; select this rank's device first.
        from baselines.fsdp2.model import TrainingModel, configuration, initialize

        stage = "model_initialization"
        config = configuration(tiny=args.tiny, sequence_length=args.sequence_length)
        checkpointed = recompute_layers(config.n_layers, args.recompute)
        request["checkpointed_layers"] = checkpointed
        write_json(output / "request.json", request)
        if args.allocator_cap_gib is not None:
            total = torch.cuda.get_device_properties(device).total_memory
            fraction = args.allocator_cap_gib * 2**30 / total
            if not 0 < fraction <= 1:
                raise ValueError("allocator cap must fit the physical device")
            torch.cuda.set_per_process_memory_fraction(fraction, device)
        from mlops.dispatch import set_weight_gradient_dtype

        set_weight_gradient_dtype(torch.float32)
        torch.manual_seed(args.seed)
        with torch.device("meta"):
            model = TrainingModel(
                config,
                local_tokens_per_step=args.global_tokens // world,
                sequence_length=args.sequence_length,
            )
        useful = (
            useful_flops_per_token(model.model, args.sequence_length)
            * args.global_tokens
        )
        if checkpointed:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
            )

            for index in checkpointed:
                model.model.blocks[index] = checkpoint_wrapper(
                    model.model.blocks[index], preserve_rng_state=False
                )
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        offload_policy = (
            CPUOffloadPolicy(pin_memory=args.pin_memory)
            if args.offload == "cpu"
            else OffloadPolicy()
        )
        for block in model.model.blocks:
            fully_shard(
                block,
                mesh=mesh,
                mp_policy=policy,
                offload_policy=offload_policy,
                reshard_after_forward=args.reshard_after_forward,
            )
        fully_shard(
            model,
            mesh=mesh,
            mp_policy=policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )
        materialization_device = (
            torch.device("cpu") if args.offload == "cpu" else device
        )
        model.to_empty(device=materialization_device)
        with torch.device(materialization_device):
            initialize(model)
        # CPUOffloadPolicy owns parameter/gradient shards, but rotary and other
        # nonpersistent compute buffers still belong on the accelerator.
        for module in model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                setattr(module, name, buffer.to(device))
        configure_prefetch(
            model, forward=args.forward_prefetch, backward=args.backward_prefetch
        )
        model.train()
        if args.compile_blocks:
            for block in model.model.blocks:
                block.compile()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            fused=args.offload != "cpu",
            foreach=False if args.offload == "cpu" else None,
        )
        generator = torch.Generator().manual_seed(args.seed * 1000003 + rank)
        shape = (1, args.global_tokens // world)
        tokens = torch.randint(config.vocab_size, shape, generator=generator).to(device)
        targets = torch.randint(config.vocab_size, shape, generator=generator).to(
            device
        )
        batch = args.microbatch_tokens
        reference = None
        if args.verify_reference:
            from baselines.fsdp2.reference import Reference

            reference = Reference(model, config, args, device, world)
        cycles, losses = [], []
        precision = None
        stage = "training"
        setup_seconds = time.perf_counter() - started
        print(
            f"rank={rank} initialized seconds={setup_seconds:.3f} "
            f"accumulation={microbatches}",
            flush=True,
        )
        for step in range(args.warmup_steps + args.steps):
            expected_loss = (
                None
                if reference is None
                else reference.step(model, tokens, targets, batch)
            )
            measured = step >= args.warmup_steps
            if step == args.warmup_steps:
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                if args.nsys_capture and rank == 0:
                    torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.synchronize()
            dist.barrier()
            tick = time.perf_counter()
            torch.cuda.nvtx.range_push(f"baseline.fsdp2/rank-{rank}/step-{step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), dtype=torch.float32, device=device)
            for index in range(microbatches):
                last = index == microbatches - 1
                model.set_is_last_backward(last)
                model.set_requires_gradient_sync(
                    args.gradient_sync == "per-microbatch" or last
                )
                # Do not leave stale gathered copies across an optimizer update.
                model.set_reshard_after_backward(args.reshard_after_backward or last)
                start = index * batch
                loss = model(
                    tokens[:, start : start + batch], targets[:, start : start + batch]
                )
                loss.backward()
                loss_sum += loss.detach().float()
            optimizer.step()
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
            if step == 0:
                precision = check_precision(model, optimizer)
            print(
                f"rank={rank} step={step + 1} measured={measured} "
                f"seconds={elapsed:.6f} loss={loss_sum.item():.6f}",
                flush=True,
            )
        if args.nsys_capture:
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
        result = {
            "status": "completed",
            "rank": rank,
            "cycles": cycles,
            "losses": losses,
            "precision": precision,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "setup_seconds": setup_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "finished_at": datetime.datetime.now().astimezone().isoformat(),
            "qualified_against_reference": reference is not None,
            "reference_checks": None if reference is None else reference.checks,
            "performance_result": reference is None
            and not args.nsys_capture
            and not args.tiny,
        }
        write_json(output / "result.json", result)
        all_ranks = [None] * world
        dist.all_gather_object(all_ranks, result)
        if rank == 0:
            summary = summarize_cycles(
                [row["cycles"] for row in all_ranks],
                global_tokens=args.global_tokens,
                useful_flops_per_step=useful,
            )
            summary["performance_result"] = result["performance_result"]
            write_json(
                args.output / "results.json",
                {
                    **summary,
                    "status": "completed",
                    "request": request,
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
                    **settings.describe(world),
                    hardware=request["hardware"],
                    provenance={
                        "source": source_identity(SOURCE_PATHS),
                        "environment": environment_identity(sys.executable),
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
