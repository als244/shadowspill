"""Tiny-model check of FSDP2's accumulation, reduction and optimizer
ownership against an unsharded eager copy of the same catalog model."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .model import TrainingModel, canonical_name, retie, shared_parameter_groups


def dense(value, device):
    # Offloaded DTensors still use a CUDA/NCCL mesh: gather a temporary CUDA
    # copy for the tiny oracle, outside measured performance runs.
    value = value.to(device)
    return value.full_tensor() if hasattr(value, "full_tensor") else value


class Reference:
    """The same model at the compute dtype on one device, its trained
    parameters' masters at the stepped dtype, stepped by plain AdamW over
    the all-reduced mean of the ranks' gradients."""

    def __init__(self, sharded, request, geometry, precision, settings, device, world):
        self.world = world
        self.device = device
        self.stepped = getattr(torch, precision.stepped_dtype)
        self.model_dtype = getattr(torch, precision.model_dtype)
        self.learning_rate = settings.learning_rate
        # A master kept at float32 reproduces to 1e-4 of its norm; one stepped
        # at bfloat16 rounds every update, so to bfloat16's own precision.
        self.master_limit = 1e-4 if self.stepped == torch.float32 else 5e-3
        self.model = TrainingModel(request, geometry, dtype=precision.model_dtype)
        groups = shared_parameter_groups(self.model)
        self.model.to_empty(device=device)
        retie(self.model, groups)
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
                value = dense(parameter, device).detach()
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
                value = dense(parameter, self.device).detach()
                self.parameters[name].copy_(value)
                if name in self.masters:
                    self.masters[name].copy_(value.to(self.master_dtype(name)))
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float32, device=tokens.device)
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
                gradient = parameter.grad.detach().to(self.master_dtype(name))
                dist.all_reduce(gradient)
                gradient /= self.world
                master = self.masters[name]
                if master.grad is None:
                    master.grad = gradient
                else:
                    master.grad.add_(gradient)
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
            values = {
                "gradient": (parameter.grad, master.grad),
                "first_moment": (
                    optimizer.state[parameter]["exp_avg"],
                    self.optimizer.state[master]["exp_avg"],
                ),
                "second_moment": (
                    optimizer.state[parameter]["exp_avg_sq"],
                    self.optimizer.state[master]["exp_avg_sq"],
                ),
                "master": (parameter, master),
            }
            for kind, (actual, expected) in values.items():
                if actual is None or expected is None:
                    raise ValueError(f"missing {kind}: {name}")
                actual = dense(actual, self.device).detach().float()
                expected = expected.detach().float()
                difference = torch.linalg.vector_norm(actual - expected).item()
                scale = torch.linalg.vector_norm(expected).item()
                relative = difference / max(scale, 1e-12)
                limit = 0.01
                if kind == "master":
                    # An update is at most the learning rate per element, so a
                    # one-percent update error (the gradient tolerance) moves
                    # a master by at most this much of its norm; a weight that
                    # starts at zero, such as a LoRA factor, is all update.
                    update = self.learning_rate * math.sqrt(expected.numel())
                    limit = self.master_limit + 0.01 * update / max(scale, 1e-12)
                if not math.isfinite(relative) or relative > limit:
                    failures.append((kind, name, relative, limit))
                worst[kind] = max(relative, worst.get(kind, 0))
        if failures:
            order = {"gradient": 0, "first_moment": 1, "second_moment": 2, "master": 3}
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
