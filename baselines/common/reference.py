"""Tiny-model check of a harness's accumulation, reduction and optimizer
ownership against an unsharded eager copy of the same catalog model.

The harness supplies an ``Access`` to its own storage: the whole value of a
parameter, of its gradient and of an optimizer state, however it shards or
offloads them."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .training import TrainingModel, canonical_name, retie, shared_parameter_groups


class Access:
    """How a harness exposes a parameter's whole value, gradient and
    optimizer state; the default reads plain tensors and FSDP2 shards."""

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
        # A master kept at float32 reproduces to 1e-4 of its norm; one stepped
        # at bfloat16 rounds every update, so to bfloat16's own precision.
        self.master_limit = 1e-4 if self.stepped == torch.float32 else 5e-3
        # The share of an update a weight that is all update (a zero-started
        # LoRA factor) may differ by: the gradient tolerance when gradients
        # accumulate at float32, five times it when they accumulate at a
        # narrower dtype, whose rounding differs with the order of the sums
        # and which Adam amplifies on the smallest gradients.
        self.update_limit = 0.01 if self.grad_dtype == torch.float32 else 0.05
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
                buffers[canonical_name(name)].copy_(buffer.to(device))
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
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float32, device=tokens.device)
        accumulated = {}
        for start in range(0, tokens.numel(), microbatch_tokens):
            self.model.zero_grad(set_to_none=True)
            loss, _metrics = self.model(
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
        torch.testing.assert_close(
            loss.detach().float(),
            loss.new_tensor(expected_loss).float(),
            rtol=1e-3,
            atol=1e-5,
        )
        worst = {}
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
                if actual is None or expected is None:
                    raise ValueError(f"missing {kind}: {name}")
                actual = actual.detach().float()
                expected = expected.detach().float()
                gap = actual - expected
                if kind == "master":
                    # An element whose gradient rounds to zero under the
                    # accumulation dtype may step either way: such a flip
                    # moves it by two learning rates. A defect moves every
                    # element; rounding flips a few, so flips are counted
                    # apart and the rest is judged by its norm.
                    flipped = gap.abs() > self.learning_rate
                    flips = flipped.float().mean().item()
                    worst["flipped_fraction"] = max(
                        flips, worst.get("flipped_fraction", 0)
                    )
                    if flips > 0.005:
                        failures.append(("master flips", name, flips, 0.005))
                    gap = gap.masked_fill(flipped, 0.0)
                difference = torch.linalg.vector_norm(gap).item()
                scale = torch.linalg.vector_norm(expected).item()
                relative = difference / max(scale, 1e-12)
                limit = 0.01
                if kind == "master":
                    # An update is at most the learning rate per element, so a
                    # one-percent update error (the gradient tolerance) moves
                    # a master by at most this much of its norm; a weight that
                    # starts at zero, such as a LoRA factor, is all update.
                    update = self.learning_rate * math.sqrt(expected.numel())
                    limit = self.master_limit + self.update_limit * update / max(
                        scale, 1e-12
                    )
                if not math.isfinite(relative) or relative > limit:
                    failures.append((kind, name, relative, limit))
                worst[kind] = max(relative, worst.get(kind, 0))
        if failures:
            order = {
                "gradient": 0,
                "first_moment": 1,
                "second_moment": 2,
                "master": 3,
                "master flips": 4,
            }
            failures.sort(key=lambda item: (order[item[0]], -item[2]))
            listed = "; ".join(
                f"{kind} of {name}: {relative:.3e} (limit {limit:.1e})"
                for kind, name, relative, limit in failures[:6]
            )
            raise ValueError(
                f"FSDP mismatch against the eager reference, {len(failures)} of"
                f" {4 * len(self.trained)} checks: {listed}; worst per kind {worst}"
            )
        self.checks.append(
            {
                "loss": float(loss.item()),
                "expected_loss": expected_loss,
                "worst_relative_errors": worst,
            }
        )
