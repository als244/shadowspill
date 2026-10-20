"""One DeepSpeed case of a catalog model; launch with torchrun, outside
ShadowSpill.

    torchrun --standalone --nproc-per-node=N -m baselines.deepspeed.train \\
        --output DIR --model NAME --tokens-per-microbatch T \\
        (--tokens-per-step S | --tier NAME) [--zero-stage 1|2|3] ...

The settings are ``baselines.deepspeed.settings``. Every rank writes
``rank-NNNNN/request.json`` as it starts and ``result.json`` or
``failure.json`` as it ends; the first rank writes ``results.json`` with
every rank's measurements and ``row.json``, the result row every harness
shares.
"""

from __future__ import annotations

import argparse
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
from torch import nn

from baselines.common import recompute_layers, summarize_cycles, write_json
from baselines.common.data import Corpus, synthetic_tokens
from baselines.common.models import useful_flops_per_token
from baselines.common.provenance import (
    environment_identity,
    hardware_identity,
    source_identity,
)
from baselines.common.reference import Access
from baselines.common.results import memory_peaks
from baselines.common.results import row as result_row
from baselines.common.training import TrainingModel, pinned_parameters
from baselines.deepspeed.settings import (
    SOURCE_PATHS,
    add_arguments,
    deepspeed_config,
    from_namespace,
)
from workloads.full_model import HEAD_LOSS_METRIC


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    add_arguments(parser)
    return parser.parse_args(argv)


class Checkpointed(nn.Module):
    """A block recomputed in backward by DeepSpeed's checkpointing; the
    attribute is named as PyTorch's wrapper names it, so the same name
    stripping applies."""

    def __init__(self, block):
        super().__init__()
        self._checkpoint_wrapped_module = block

    def forward(self, hidden, *rest, **kwargs):
        import deepspeed

        # DeepSpeed's checkpoint rebuilds its arguments and would turn the
        # block's packing tuple into a plain one; everything but the hidden
        # state is closed over, as plain inputs recomputation reuses. Its
        # reentrant checkpoint builds no graph when no input needs a
        # gradient, which a frozen embedding before a LoRA block leaves
        # true, so the hidden state is marked as needing one.
        if torch.is_grad_enabled() and not hidden.requires_grad:
            hidden = hidden.detach().requires_grad_(True)

        def block(value):
            return self._checkpoint_wrapped_module(value, *rest, **kwargs)

        return deepspeed.checkpointing.checkpoint(block, hidden)


class DeepSpeedAccess(Access):
    """Whole values through DeepSpeed's accessors: the float32 master of a
    trained parameter, a gathered partitioned parameter otherwise, the
    gradient as snapshotted before the step (DeepSpeed clears it in the
    step), and the optimizer state."""

    def __init__(self, engine, device):
        super().__init__(engine.optimizer, device)
        self.engine = engine
        self.gradients = {}

    def snapshot(self, model):
        # The accumulated gradient of the update, at the accumulation dtype:
        # DeepSpeed's generic accessor would hand back the parameter's own
        # low-precision gradient of the last microbatch when it still holds
        # one, so the optimizer's copy is read directly.
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if hasattr(parameter, "ds_id"):
                value = parameter._z3_optimizer.get_fp32_grad_for_param(parameter)
            elif hasattr(parameter, "_hp_mapping"):
                value = parameter.get_full_hp_grad()
            else:
                value = parameter.grad
            self.gradients[id(parameter)] = (
                None if value is None else value.detach().clone().to(self.device)
            )

    def parameter(self, parameter):
        from deepspeed.utils import safe_get_full_fp32_param

        if parameter.requires_grad:
            value = safe_get_full_fp32_param(parameter)
            if value is not None:
                return value.to(self.device)
        if hasattr(parameter, "ds_id"):
            import deepspeed

            with deepspeed.zero.GatheredParameters([parameter]):
                return parameter.detach().clone().to(self.device)
        return parameter.detach().to(self.device)

    def gradient(self, parameter):
        return self.gradients.get(id(parameter))

    def state(self, parameter, key):
        from deepspeed.utils import safe_get_full_optimizer_state

        value = safe_get_full_optimizer_state(parameter, key)
        return None if value is None else value.to(self.device)


def precision_evidence(model, access):
    """The masters and moments DeepSpeed keeps are float32; counts are the
    evidence."""

    counts = {"parameters": 0, "moment_elements": 0}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        master = access.parameter(parameter)
        if master.dtype != torch.float32:
            raise ValueError(f"a master is {master.dtype}, not float32")
        counts["parameters"] += master.numel()
        for key in ("exp_avg", "exp_avg_sq"):
            value = access.state(parameter, key)
            if value is not None:
                if value.dtype != torch.float32:
                    raise ValueError(f"an AdamW moment is {value.dtype}")
                counts["moment_elements"] += value.numel()
    if not counts["moment_elements"]:
        raise ValueError("optimizer evidence is empty")
    return counts


def main(argv=None):
    args = parse(argv)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    settings = from_namespace(args)
    geometry = settings.validate(world)
    request = settings.request()
    precision = settings.precision()
    model_dtype = getattr(torch, precision.model_dtype)
    torch.set_num_threads(settings.cpu_threads)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    output = args.output / f"rank-{rank:05d}"
    output.mkdir(parents=True, exist_ok=False)
    stage = "distributed_initialization"
    description = settings.describe(world)
    config = deepspeed_config(settings, geometry, precision)
    record = {
        **description,
        "output": str(args.output),
        "rank": rank,
        "world_size": world,
        "hardware": hardware_identity(),
        "resolved_precision": precision.resolved(),
        "microbatches_per_step": geometry.accumulation,
        "deepspeed_config": config,
        "shadowspill": False,
    }
    write_json(output / "request.json", record)
    try:
        # The op builder refuses a toolkit whose minor version differs from
        # PyTorch's build; the extensions built here do not depend on it.
        os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl")
        stage = "model_initialization"
        if settings.allocator_cap_gib is not None:
            total = torch.cuda.get_device_properties(device).total_memory
            fraction = settings.allocator_cap_gib * 2**30 / total
            if not 0 < fraction <= 1:
                raise ValueError("allocator cap must fit the physical device")
            torch.cuda.set_per_process_memory_fraction(fraction, device)
        from mlops.dispatch import set_weight_gradient_dtype

        set_weight_gradient_dtype(model_dtype)
        torch.manual_seed(settings.seed)
        # Every parameter is built with its values, at the model dtype
        # DeepSpeed casts to anyway; in stage three inside zero.Init, which
        # partitions each parameter as its module is constructed.
        pinned_names = pinned_parameters(request)
        record["pinned_float32_parameters_cast"] = sorted(pinned_names)
        if settings.zero_stage == 3:
            remote = "cpu" if settings.offload_param == "cpu" else None
            with deepspeed.zero.Init(
                config_dict_or_path=config,
                dtype=model_dtype,
                remote_device=remote,
                pin_memory=settings.pin_memory,
            ):
                model = TrainingModel(
                    request, geometry, dtype=precision.model_dtype, device="cpu"
                )
        else:
            model = TrainingModel(
                request, geometry, dtype=precision.model_dtype, device="cpu"
            )
        useful = (
            useful_flops_per_token(model.model, geometry.sequence_length)
            * geometry.tokens_per_step
        )
        blocks = model.model.blocks
        checkpointed = recompute_layers(len(blocks), settings.recompute)
        record["checkpointed_layers"] = checkpointed
        record["summary"] = request.summary(model.model)
        write_json(output / "request.json", record)
        if settings.checkpoint is not None:
            checkpoint = request.checkpoint(settings.checkpoint)
            record["checkpoint"] = {
                "directory": str(Path(settings.checkpoint).resolve()),
                "config": checkpoint.config.__dict__
                if not hasattr(checkpoint.config, "__dataclass_fields__")
                else {
                    name: getattr(checkpoint.config, name)
                    for name in checkpoint.config.__dataclass_fields__
                },
            }

            def copy(parameter, value):
                # A partitioned parameter is gathered, written on the first
                # rank and partitioned again; a whole one is written as is.
                if hasattr(parameter, "ds_id"):
                    with deepspeed.zero.GatheredParameters(
                        [parameter], modifier_rank=0
                    ):
                        if rank == 0:
                            parameter.data.copy_(value.to(parameter.dtype))
                else:
                    parameter.data.copy_(value.to(parameter.dtype))

            with checkpoint:
                record["loaded"] = checkpoint.load_into(model.model, copy=copy)
            if record["loaded"]["unused"]:
                raise ValueError(
                    "checkpoint keys the model did not take:"
                    f" {record['loaded']['unused'][:5]}"
                )
        if checkpointed:
            deepspeed.checkpointing.configure(
                None,
                partition_activations=False,
                contiguous_checkpointing=settings.contiguous_checkpointing,
                num_checkpoints=len(checkpointed)
                if settings.contiguous_checkpointing
                else None,
                checkpoint_in_cpu=settings.activation_offload == "cpu",
                synchronize=False,
                profile=False,
            )
            for index in checkpointed:
                blocks[index] = Checkpointed(blocks[index])
        trainable = [p for p in model.parameters() if p.requires_grad]
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable, config=config
        )
        engine.train()
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
        access = DeepSpeedAccess(engine, device)
        reference = None
        if settings.verify_reference:
            from baselines.common.reference import Reference

            reference = Reference(
                model,
                request,
                geometry,
                precision,
                settings,
                device,
                world,
                access=access,
                cast_pinned=precision.model_dtype != "float32",
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
            torch.cuda.nvtx.range_push(
                f"baseline.deepspeed/rank-{rank}/step-{step + 1}"
            )
            loss_sum = torch.zeros((), dtype=torch.float32, device=device)
            head_sum = torch.zeros((), dtype=torch.float32, device=device)
            for index in range(microbatches):
                start = index * batch
                loss, metrics = engine(
                    tokens[:, start : start + batch], targets[:, start : start + batch]
                )
                # DeepSpeed takes each microbatch's mean loss and divides the
                # gradients by the accumulation count itself (in its own
                # backward or its per-tensor hooks); the objective is the
                # microbatch's share of the update, the mean over the
                # accumulation, so it is handed over multiplied back.
                engine.backward(loss * microbatches)
                loss_sum += loss.detach().float()
                head_sum += metrics.get(HEAD_LOSS_METRIC, loss.detach()).float()
                if reference is not None and index == microbatches - 1:
                    access.snapshot(model)
                engine.step()
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
            if step == 0:
                evidence = precision_evidence(model, access)
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
                    harness="deepspeed",
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
                        "deepspeed_config": config,
                        "pinned_float32_parameters_cast": record[
                            "pinned_float32_parameters_cast"
                        ],
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
