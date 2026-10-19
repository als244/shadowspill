"""Small-model check of FSDP accumulation, reduction, and optimizer ownership."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .model import TrainingModel


def canonical_name(name):
    return name.replace("._checkpoint_wrapped_module.", ".").replace("._orig_mod.", ".")


def dense(value, device):
    # Offloaded DTensors still use a CUDA/NCCL mesh: gather a temporary CUDA
    # copy for the tiny oracle, outside measured performance runs.
    value = value.to(device)
    return value.full_tensor() if hasattr(value, "full_tensor") else value


class Reference:
    def __init__(self, sharded, config, args, device, world):
        self.world = world
        self.device = device
        with torch.device(device):
            self.model = TrainingModel(
                config,
                local_tokens_per_step=args.global_tokens // world,
                sequence_length=args.sequence_length,
            ).to(dtype=torch.bfloat16)
        self.model.train()
        self.parameters = dict(self.model.named_parameters())
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
                value = dense(parameter, device).detach().float()
                self.masters[name] = torch.nn.Parameter(value.clone())
                self.parameters[name].copy_(value)
        if set(self.parameters) != set(self.masters):
            raise ValueError("reference parameter inventory differs from FSDP")
        self.optimizer = torch.optim.AdamW(
            self.masters.values(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            foreach=False,
        )
        self.checks = []

    def step(self, sharded, tokens, targets, microbatch_tokens):
        # Compare each update from exactly the same master weights. Tiny FP32
        # CPU/fused-Adam rounding differences can cross a BF16 rounding boundary
        # on the next forward and amplify through top-k routing. Every preceding
        # update has already passed the master/gradient/moment checks; retain
        # the independent optimizer moments across updates.
        with torch.no_grad():
            for raw_name, parameter in sharded.named_parameters():
                name = canonical_name(raw_name)
                value = dense(parameter, self.device).detach().float()
                self.masters[name].copy_(value)
                self.parameters[name].copy_(value)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float32, device=tokens.device)
        for start in range(0, tokens.numel(), microbatch_tokens):
            self.model.zero_grad(set_to_none=True)
            loss = self.model(
                tokens[:, start : start + microbatch_tokens],
                targets[:, start : start + microbatch_tokens],
            )
            loss.backward()
            loss_sum += loss.detach().float()
            for name, parameter in self.parameters.items():
                if parameter.grad is None:
                    raise ValueError(f"reference gradient missing: {name}")
                gradient = parameter.grad.detach().float()
                dist.all_reduce(gradient)
                gradient /= self.world
                master = self.masters[name]
                if master.grad is None:
                    master.grad = gradient
                else:
                    master.grad.add_(gradient)
        self.optimizer.step()
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                parameter.copy_(self.masters[name])
        return float(loss_sum.item())

    def check(self, model, optimizer, loss, expected_loss):
        torch.testing.assert_close(
            loss.detach().float(),
            loss.new_tensor(expected_loss).float(),
            rtol=1e-3,
            atol=1e-5,
        )
        worst = {}
        for raw_name, parameter in model.named_parameters():
            name = canonical_name(raw_name)
            master = self.masters[name]
            values = {
                "master": (parameter, master),
                "gradient": (parameter.grad, master.grad),
                "first_moment": (
                    optimizer.state[parameter]["exp_avg"],
                    self.optimizer.state[master]["exp_avg"],
                ),
                "second_moment": (
                    optimizer.state[parameter]["exp_avg_sq"],
                    self.optimizer.state[master]["exp_avg_sq"],
                ),
            }
            for kind, (actual, expected) in values.items():
                if actual is None or expected is None:
                    raise ValueError(f"missing {kind}: {name}")
                actual = dense(actual, self.device).detach().float()
                expected = expected.detach().float()
                difference = torch.linalg.vector_norm(actual - expected).item()
                scale = torch.linalg.vector_norm(expected).item()
                relative = difference / max(scale, 1e-12)
                limit = 1e-4 if kind == "master" else 0.01
                if not math.isfinite(relative) or relative > limit:
                    raise ValueError(
                        f"FSDP {kind} mismatch for {name}: relative error {relative}"
                    )
                worst[kind] = max(relative, worst.get(kind, 0))
        self.checks.append(
            {
                "loss": float(loss.item()),
                "expected_loss": expected_loss,
                "worst_relative_errors": worst,
            }
        )
