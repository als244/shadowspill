"""One DeepSpeed case of a catalog model, outside ShadowSpill: the trainer
``baselines.common.launch`` starts under torchrun.

The settings are ``baselines.engines.deepspeed.settings``, validated by the
launcher; ``baselines.common.runner`` writes the request, the result and
the row around ``train``, which builds the engine and runs every step under
the runner's clock.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn

from baselines.common import recompute_layers, runner
from baselines.common.reference import Access
from baselines.common.training import TrainingModel, pinned_parameters
from baselines.engines.deepspeed import settings as engine_settings
from baselines.engines.deepspeed.settings import deepspeed_config
from workloads.full_model import HEAD_LOSS_METRIC


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
    """Whole values through DeepSpeed's accessors: the master the optimizer
    steps for a trained parameter, a gathered partitioned parameter
    otherwise, the gradient as snapshotted before the step (DeepSpeed clears
    it in the step), and the optimizer state."""

    def __init__(self, engine, device, *, gradients=True):
        super().__init__(engine.optimizer, device)
        self.engine = engine
        self.gradients = {}
        if not gradients:
            # After a backward DeepCompile ran inside its graphs, DeepSpeed's
            # accessors refuse the gradient (their window between backward
            # and step is not seen), so the moments, which the first update
            # makes out of the gradient alone, are judged instead.
            self.exposes = ("first_moment", "second_moment", "master")

    def snapshot(self, model):
        # The accumulated gradient of the update, at the accumulation dtype:
        # DeepSpeed's generic accessor would hand back the parameter's own
        # low-precision gradient of the last microbatch when it still holds
        # one, so the optimizer's copy is read directly.
        if "gradient" not in self.exposes:
            return
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


def precision_evidence(model, access, precision):
    """The values DeepSpeed steps and its AdamW moments are at the dtypes the
    contract resolves to, float32 where the optimizer keeps its own; counts
    are the evidence."""

    stepped = getattr(torch, precision.stepped_dtype)
    moments = precision.resolved()["opt_state_dtype"]
    moment_dtype = torch.float32 if moments == "default" else getattr(torch, moments)

    def held(parameter, whole, key=None):
        # ZeRO-1 and 2 gather a whole value into a float32 buffer whatever
        # their fragment holds, so the dtype is the fragment's, on the rank
        # that holds one; ZeRO-3's whole value keeps its dtype.
        if not hasattr(parameter, "_hp_mapping"):
            return whole
        mapping = parameter._hp_mapping
        return None if mapping is None else mapping.get_hp_fragment(key)

    counts = {"parameters": 0, "moment_elements": 0}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        master = access.parameter(parameter)
        stored = held(parameter, master)
        if stored is not None and stored.dtype != stepped:
            raise ValueError(f"a master is {stored.dtype}, not {stepped}")
        counts["parameters"] += master.numel()
        for key in ("exp_avg", "exp_avg_sq"):
            value = access.state(parameter, key)
            if value is not None:
                stored = held(parameter, value, key)
                if stored is not None and stored.dtype != moment_dtype:
                    raise ValueError(
                        f"an AdamW moment is {stored.dtype}, not {moment_dtype}"
                    )
                counts["moment_elements"] += value.numel()
    if not counts["moment_elements"]:
        raise ValueError("optimizer evidence is empty")
    return counts


def train(case, clock):
    """Build the DeepSpeed engine over the catalog model and run every step
    of the case."""

    settings, geometry, precision = case.settings, case.geometry, case.precision
    request = settings.request()
    record = case.record
    rank, world, device = case.rank, case.world, case.device
    model_dtype = getattr(torch, precision.model_dtype)
    config = deepspeed_config(settings, geometry, precision)
    record["deepspeed_config"] = config
    case.save()
    # The op builder refuses a toolkit whose minor version differs from
    # PyTorch's build; the extensions built here do not depend on it.
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
    # The environment's own CUDA home when setup.sh assembled one: the
    # toolkit with the NCCL headers of the NCCL PyTorch links, for the
    # extensions that include them (DeepCompile's). PyTorch's extension
    # builder fixes its CUDA home when it is imported, which the model
    # imports already did, so the loaded value is set as well.
    cuda_home = Path(sys.prefix) / "cuda-home"
    if cuda_home.is_dir():
        os.environ["CUDA_HOME"] = str(cuda_home)
        builder = sys.modules.get("torch.utils.cpp_extension")
        if builder is not None:
            builder.CUDA_HOME = str(cuda_home)
        record["cuda_home"] = str(cuda_home)
    import deepspeed

    deepspeed.init_distributed(dist_backend="nccl")
    case.stage = "model_initialization"
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
    blocks = model.model.blocks
    checkpointed = recompute_layers(len(blocks), settings.recompute)
    record["checkpointed_layers"] = checkpointed
    case.save()
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
                with deepspeed.zero.GatheredParameters([parameter], modifier_rank=0):
                    if rank == 0:
                        parameter.data.copy_(value.to(parameter.dtype))
            else:
                parameter.data.copy_(value.to(parameter.dtype))

        with checkpoint:
            record["loaded"] = model.load_checkpoint(checkpoint, copy=copy)
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
    if settings.deepcompile:
        # DeepCompile is DeepSpeed's own torch.compile backend: its passes
        # replace ZeRO's hooks with reduction, gather, release and offload
        # nodes in the captured graphs. The engine falls back to plain
        # torch.compile when it declines a configuration, silently, so
        # the measurement asks whether the passes are in place.
        case.stage = "compilation"
        engine.compile(backend=settings.deepcompile_backend)
        if not engine.is_deepcompile_active():
            raise RuntimeError(
                "DeepSpeed compiled the model without DeepCompile's passes"
            )
        record["deepcompile"] = config["compile"]
    data = case.data(int(model.model.config.vocab_size), device=device)
    case.save()
    batch = geometry.tokens_per_microbatch
    microbatches = geometry.accumulation
    access = DeepSpeedAccess(engine, device, gradients=not settings.deepcompile)
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
    evidence = None
    clock.ready()
    for step in range(clock.steps):
        tokens, targets = data(step)
        expected_loss = (
            None if reference is None else reference.step(model, tokens, targets, batch)
        )
        clock.begin(step)
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
        clock.end(step, loss_sum, head_sum)
        if reference is not None:
            reference.check(model, optimizer, loss_sum, expected_loss)
        if step == 0:
            evidence = precision_evidence(model, access, precision)
    return runner.Outcome(
        evidence=evidence,
        reference_checks=None if reference is None else reference.checks,
        provenance={
            "checkpointed_layers": checkpointed,
            "deepspeed_config": config,
            "pinned_float32_parameters_cast": record["pinned_float32_parameters_cast"],
        },
    )


def main(argv=None):
    runner.main("deepspeed", engine_settings, train, argv)


if __name__ == "__main__":
    main()
