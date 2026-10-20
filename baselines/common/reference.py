"""Tiny-model check of a harness's accumulation, reduction and optimizer
ownership against an unsharded copy of the same catalog model, compiled
whole as the repository's numerical gate compiles its reference.

The harness supplies an ``Access`` to its own storage: the whole value of a
parameter, of its gradient and of an optimizer state, however it shards or
offloads them."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .training import TrainingModel, canonical_name, retie, shared_parameter_groups

# The repository's numerical gate's bounds (``qualification/numerical/
# tolerances.py``), held here because no harness but ShadowSpill's imports
# ShadowSpill; a unit test pins them to the gate's. A loss agrees within an
# absolute and a relative bound. A weight agrees when its cosine, relative
# L2 and sign agreement are within the weight bounds; a moment or a
# gradient, an accumulator of small values summed in another order, gets
# twice the relative L2 and no sign test, its near-zero elements changing
# sign with any rounding. A weight that is still nothing but its optimizer's
# steps has no scale of its own: it agrees when no element is more than two
# learning rates from the reference, the most one step's rounding
# disagreement can move it. The gate tells such a weight by every element
# lying within its steps, in learning rates, of zero; AdamW's bias-corrected
# step exceeds one learning rate a little whenever a gradient grows, so a
# zero-started LoRA factor whose gradients keep their sign passes that line.
# Here such a weight is told by its start: every element zero before the
# first update, as the gate's own example, a bias started at zero, is.
LOSS_RELATIVE_TOLERANCE = 0.01
LOSS_ABSOLUTE_TOLERANCE = 2e-5
MINIMUM_COSINE = 0.999
MAXIMUM_RELATIVE_L2 = 0.025
MAXIMUM_RELATIVE_L2_OPTIMIZER = 0.05
MINIMUM_SIGN_AGREEMENT = 0.99
STEP_QUANTUM_DISAGREEMENTS = 2


@dataclass(frozen=True)
class TensorMetrics:
    cosine: float
    relative_l2: float
    sign_agreement: float
    maximum_absolute_error: float
    reference_maximum_absolute: float


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> TensorMetrics:
    """The gate's per-tensor metrics, reduced in float64."""

    left = reference.detach().float().reshape(-1)
    right = actual.detach().float().reshape(-1)
    if left.numel() == 0:
        return TensorMetrics(1.0, 0.0, 1.0, 0.0, 0.0)
    difference = right - left
    left_norm = torch.linalg.vector_norm(left, dtype=torch.float64)
    right_norm = torch.linalg.vector_norm(right, dtype=torch.float64)
    denominator = left_norm * right_norm
    cosine = (
        1.0
        if float(denominator) == 0.0 and torch.equal(left, right)
        else float(
            torch.sum(left * right, dtype=torch.float64) / denominator.clamp_min(1e-30)
        )
    )
    relative_l2 = float(
        torch.linalg.vector_norm(difference, dtype=torch.float64)
        / left_norm.clamp_min(1e-30)
    )
    sign_agreement = float((torch.sign(left) == torch.sign(right)).float().mean())
    return TensorMetrics(
        cosine=cosine,
        relative_l2=relative_l2,
        sign_agreement=sign_agreement,
        maximum_absolute_error=float(difference.abs().max()),
        reference_maximum_absolute=float(left.abs().max()),
    )


def meets_tolerance(metric, *, weight, step_size, zero_started=False):
    """Whether one tensor agrees with the reference, by the gate's rule:
    the relative bounds, or for a weight that started at zero and so is
    still nothing but its steps, the absolute bound of two learning rates."""

    bound = MAXIMUM_RELATIVE_L2 if weight else MAXIMUM_RELATIVE_L2_OPTIMIZER
    # Signs are a weight's: a gradient or a moment is an accumulator whose
    # near-zero elements change sign with any rounding, so it is judged by
    # its cosine and relative L2 alone.
    if (
        metric.cosine >= MINIMUM_COSINE
        and metric.relative_l2 <= bound
        and (not weight or metric.sign_agreement >= MINIMUM_SIGN_AGREEMENT)
    ):
        return True
    return bool(
        weight
        and zero_started
        and metric.maximum_absolute_error <= STEP_QUANTUM_DISAGREEMENTS * step_size
    )


class Access:
    """How a harness exposes a parameter's whole value, gradient and
    optimizer state, and a buffer's value; the default reads plain tensors
    and FSDP2 shards."""

    #: The kinds of value this access exposes, of the four the check knows;
    #: a harness that consumes its gradients inside its step leaves that
    #: one out, and the moments the first update makes are the gradient's.
    exposes = ("gradient", "first_moment", "second_moment", "master")

    def __init__(self, optimizer, device):
        self.optimizer = optimizer
        self.device = device

    def dense(self, value):
        # Offloaded DTensors still use a CUDA/NCCL mesh: gather a temporary
        # CUDA copy for the tiny oracle, outside measured performance runs.
        value = value.to(self.device)
        return value.full_tensor() if hasattr(value, "full_tensor") else value

    def parameter(self, parameter):
        return self.dense(parameter)

    def buffer(self, name, buffer):
        return buffer.to(self.device)

    def gradient(self, parameter):
        return None if parameter.grad is None else self.dense(parameter.grad)

    def state(self, parameter, key):
        value = self.optimizer.state[parameter].get(key)
        return None if value is None else self.dense(value)


class Reference:
    """The same model at the compute dtype on one device, its trained
    parameters' masters at the stepped dtype, stepped by plain AdamW over
    the all-reduced mean of the ranks' gradients."""

    def __init__(
        self,
        sharded,
        request,
        geometry,
        precision,
        settings,
        device,
        world,
        access=None,
        cast_pinned=False,
    ):
        self.world = world
        self.device = device
        self.access = access or Access(None, device)
        self.stepped = getattr(torch, precision.stepped_dtype)
        self.model_dtype = getattr(torch, precision.model_dtype)
        # Gradients accumulate over the microbatches at the contract's
        # gradient dtype, as the harness does, before the masters take them.
        self.grad_dtype = getattr(torch, precision.grad_dtype or precision.model_dtype)
        self.learning_rate = settings.learning_rate
        self.model = TrainingModel(request, geometry, dtype=precision.model_dtype)
        groups = shared_parameter_groups(self.model)
        self.model.to_empty(device=device)
        retie(self.model, groups)
        if cast_pinned:
            # The harness casts every parameter to the model dtype, the ones
            # the model pins at float32 included; so does this copy.
            for parameter in self.model.parameters():
                parameter.data = parameter.data.to(self.model_dtype)
        self.model.train()
        self.parameters = dict(self.model.named_parameters())
        self.trained = {
            canonical_name(name)
            for name, parameter in sharded.named_parameters()
            if parameter.requires_grad
        }
        self.masters = {}
        with torch.no_grad():
            buffers = dict(self.model.named_buffers())
            for name, buffer in sharded.named_buffers():
                # Preserve the actual FP32 rotary tables. Casting the whole
                # reference module to BF16 can round these before _apply
                # restores their dtype, changing the attention computation.
                buffers[canonical_name(name)].copy_(self.access.buffer(name, buffer))
            for name, parameter in sharded.named_parameters():
                name = canonical_name(name)
                value = self.access.parameter(parameter).detach()
                self.parameters[name].copy_(value)
                if name in self.trained:
                    # A parameter the model pins at another dtype keeps it as
                    # its master; every other master is at the stepped dtype.
                    self.masters[name] = torch.nn.Parameter(
                        value.to(self.master_dtype(name)).clone()
                    )
        if set(self.parameters) != {
            canonical_name(n) for n, _ in sharded.named_parameters()
        }:
            raise ValueError("reference parameter inventory differs from FSDP")
        for name, parameter in self.parameters.items():
            parameter.requires_grad_(name in self.trained)
        self.optimizer = torch.optim.AdamW(
            self.masters.values(),
            lr=settings.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=settings.weight_decay,
            foreach=False,
        )
        self.checks = []
        #: The copy's microbatch objective, compiled whole on the first
        #: update, once the harness has finished the copy's parameters.
        self.compiled = None
        #: The trained weights every element of which was zero before the
        #: first update: still nothing but their optimizer's steps.
        self.zero_started = None

    def master_dtype(self, name):
        own = self.parameters[name].dtype
        return self.stepped if own == self.model_dtype else own

    def step(self, sharded, tokens, targets, microbatch_tokens):
        # Compare each update from exactly the same master weights. Tiny FP32
        # CPU/fused-Adam rounding differences can cross a BF16 rounding boundary
        # on the next forward and amplify through top-k routing. Every preceding
        # update has already passed the master/gradient/moment checks; retain
        # the independent optimizer moments across updates.
        with torch.no_grad():
            for raw_name, parameter in sharded.named_parameters():
                name = canonical_name(raw_name)
                value = self.access.parameter(parameter).detach()
                self.parameters[name].copy_(value)
                if name in self.masters:
                    self.masters[name].copy_(value.to(self.master_dtype(name)))
        if self.zero_started is None:
            self.zero_started = {
                name for name, master in self.masters.items() if not torch.any(master)
            }
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float32, device=tokens.device)
        accumulated = {}
        for start in range(0, tokens.numel(), microbatch_tokens):
            self.model.zero_grad(set_to_none=True)
            if self.compiled is None:
                self.compiled = torch.compile(self.model, fullgraph=True, dynamic=False)
            loss, _metrics = self.compiled(
                tokens[:, start : start + microbatch_tokens],
                targets[:, start : start + microbatch_tokens],
            )
            loss.backward()
            loss_sum += loss.detach().float()
            for name in self.trained:
                parameter = self.parameters[name]
                if parameter.grad is None:
                    raise ValueError(f"reference gradient missing: {name}")
                dtype = (
                    self.grad_dtype
                    if parameter.dtype == self.model_dtype
                    else self.master_dtype(name)
                )
                gradient = parameter.grad.detach().to(dtype)
                if self.world > 1:
                    dist.all_reduce(gradient)
                    gradient /= self.world
                if name in accumulated:
                    accumulated[name].add_(gradient)
                else:
                    accumulated[name] = gradient
        for name, gradient in accumulated.items():
            self.masters[name].grad = gradient.to(self.master_dtype(name))
        self.optimizer.step()
        with torch.no_grad():
            for name, master in self.masters.items():
                self.parameters[name].copy_(master)
        return float(loss_sum.item())

    def check(self, model, optimizer, loss, expected_loss):
        actual_loss = float(loss.detach().float())
        if abs(actual_loss - expected_loss) > (
            LOSS_ABSOLUTE_TOLERANCE + LOSS_RELATIVE_TOLERANCE * abs(expected_loss)
        ):
            raise ValueError(
                f"loss {actual_loss} against the reference's {expected_loss}"
            )
        worst = {}
        lowest_cosine = {}
        lowest_sign_agreement = {}
        failures = []
        for raw_name, parameter in model.named_parameters():
            name = canonical_name(raw_name)
            if name not in self.trained:
                continue
            master = self.masters[name]
            access = self.access
            values = {
                "gradient": (access.gradient(parameter), master.grad),
                "first_moment": (
                    access.state(parameter, "exp_avg"),
                    self.optimizer.state[master]["exp_avg"],
                ),
                "second_moment": (
                    access.state(parameter, "exp_avg_sq"),
                    self.optimizer.state[master]["exp_avg_sq"],
                ),
                "master": (access.parameter(parameter), master),
            }
            for kind, (actual, expected) in values.items():
                if kind not in access.exposes:
                    continue
                if actual is None or expected is None:
                    raise ValueError(f"missing {kind}: {name}")
                metric = tensor_metrics(expected, actual)
                if not meets_tolerance(
                    metric,
                    weight=kind == "master",
                    step_size=self.learning_rate,
                    zero_started=name in self.zero_started,
                ):
                    failures.append((kind, name, metric))
                worst[kind] = max(metric.relative_l2, worst.get(kind, 0.0))
                lowest_cosine[kind] = min(metric.cosine, lowest_cosine.get(kind, 1.0))
                lowest_sign_agreement[kind] = min(
                    metric.sign_agreement, lowest_sign_agreement.get(kind, 1.0)
                )
        if failures:
            order = {"gradient": 0, "first_moment": 1, "second_moment": 2, "master": 3}
            failures.sort(key=lambda item: (order[item[0]], -item[2].relative_l2))
            rate = self.learning_rate
            listed = "; ".join(
                f"{kind} of {name}: relative L2 {metric.relative_l2:.3e}, cosine"
                f" {metric.cosine:.6f}, sign agreement {metric.sign_agreement:.4f},"
                f" largest error {metric.maximum_absolute_error / rate:.2f} lr"
                for kind, name, metric in failures[:6]
            )
            raise ValueError(
                f"mismatch against the reference, {len(failures)} of"
                f" {len(self.access.exposes) * len(self.trained)} checks: {listed};"
                f" worst relative L2 per kind {worst}"
            )
        self.checks.append(
            {
                "loss": float(loss.item()),
                "expected_loss": expected_loss,
                "worst_relative_errors": worst,
                "lowest_cosines": lowest_cosine,
                "lowest_sign_agreements": lowest_sign_agreement,
            }
        )
