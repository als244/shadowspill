"""Small CPU EP oracle: explicit experts, full head, router formula and AdamW.

No communication/provider kernel or engine capture computes reference values.
One model holds every expert; explicit lists give each logical input owner its
own physical microbatch. Gate/up rows are interleaved, and weighted expert
outputs round before their FP32 sum. Eager autograd supplies BF16 gradients;
observed linear products retain FP32 gradients before leaf rounding when needed.
"""

from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.parallelism import InputFieldPlacement, InputPlacement
from shadowspill.training.definitions import load_registry
from shadowspill_client import Objective
from shadowspill_workloads.definitions.expert_parallel import ownership
from shadowspill_workloads.initialization.decoder import Initialize as DecoderInitialize
from shadowspill_workloads.initialization.expert_parallel import BANKS, Initialize
from workload_tests.definitions.test_expert_definitions import settings
from workload_tests.service.oracles._moe_oracle import WeightGradients as DenseGradients


class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        e, h, d = config.n_experts, config.d_ff_expert, config.d_model
        self.router_weight = nn.Parameter(torch.empty(e, d, dtype=torch.bfloat16))
        self.gate_up_weight = nn.Parameter(
            torch.empty(e, 2 * h, d, dtype=torch.bfloat16)
        )
        self.down_weight = nn.Parameter(torch.empty(e, d, h, dtype=torch.bfloat16))


class ReferenceMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert not config.d_ff_shared
        self.config, self.experts, self.observe = config, Experts(config), None
        self.gradient_observer = None

    def forward(self, hidden, residual, *, router_mask=None):
        x = hidden.flatten(0, -2)

        def linear(inputs, name, expert=None):
            weight = getattr(self.experts, name)
            if expert is not None:
                weight = weight[expert]
            output = F.linear(inputs.float(), weight.float())
            if self.gradient_observer is not None:
                self.gradient_observer(name, expert, inputs, output)
            return output

        logits = linear(x, "router_weight")
        if self.observe is not None:
            self.observe.append(logits)
        selected, ids = logits.topk(self.config.top_k, -1)
        scores = selected.softmax(-1)
        result = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(self.config.n_experts):
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            # The up epilogue consumes FP32 products before storing BF16
            # activations. Adjacent rows are gate/up, not two contiguous halves.
            projection = linear(x[rows], "gate_up_weight", expert)
            activation = (F.silu(projection[:, 0::2]) * projection[:, 1::2]).bfloat16()
            raw = linear(activation, "down_weight", expert)
            weighted = (raw * scores[rows, slots, None]).bfloat16()
            result = result.index_add(0, rows, weighted.float())
        mask = (
            torch.ones_like(logits[:, 0])
            if router_mask is None
            else router_mask.flatten().float()
        )
        counts = (F.one_hot(ids, self.config.n_experts).sum(1) * mask[:, None]).sum(0)
        probabilities = (logits.softmax(-1) * mask[:, None]).sum(0)
        return residual + result.bfloat16().reshape_as(hidden), counts, probabilities


class WeightGradients(DenseGradients):
    """Observe FP32 products directly in the independent expert equations."""

    def __init__(self, model):
        super().__init__(model)
        self.experts = []
        for name, module in model.named_modules():
            if isinstance(module, ReferenceMoE):
                self.experts.append(module)
                module.gradient_observer = lambda local, expert, x, y, name=name: (
                    self._observe(f"{name}.experts.{local}", expert, x, y)
                )

    def close(self):
        super().close()
        for module in self.experts:
            module.gradient_observer = None


def initial_model(*, grad_dtype="bfloat16"):
    assert grad_dtype in ("bfloat16", "float32")
    requested = settings("qwen3moe")
    requested = replace(
        requested,
        config={
            "architecture": {
                **requested.config["architecture"],
                "n_layers": 1,
                "vocab_size": 33,
                "max_seq_len": 4,
            }
        },
        precision={"grad_dtype": grad_dtype},
    )
    # Reuse only the supplied dense CPU architecture, replacing every MoE with
    # elementary tensor equations. The service still constructs real Quack EP.
    dense = replace(requested, implementation="mlops")
    model = (
        load_registry(["shadowspill-workloads"])
        .resolve_model(dense.definition, settings=dense)
        .create()
    )
    for block in model.blocks:
        block.moe = ReferenceMoE(model.config)
    base = DecoderInitialize("qwen3moe", model.config, seed=17)
    with torch.no_grad():
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            sharded = name.endswith(("experts.gate_up_weight", "experts.down_weight"))
            for rank, part in enumerate(value.chunk(2) if sharded else (value,)):
                fill = Initialize(
                    base,
                    expert_rank=rank,
                    ranks=2 if sharded else 1,
                    banks=BANKS["qwen3moe"],
                    std=model.config.initializer_range,
                    router_dtype=torch.bfloat16,
                )
                fill(
                    TensorChunk(name, tuple(part.shape), part.dtype, 0, part.numel()),
                    part.reshape(-1),
                )
    return requested, model


def records():
    return [
        {
            "inputs": {"tokens": tokens},
            "loss_inputs": {
                "target_tokens": targets,
                "loss_mask": mask,
                "weights": weights,
            },
        }
        for tokens, targets, mask, weights in (
            ([1, 2, 3], [2, 0, 4], [True, False, True], [1.0, 1.0, -0.5]),
            ([4, 5], [5, 6], [True, True], [0.0, 1.2]),
            ([6, 7], [7, 8], [True, True], [0.3, 0.8]),
            ([8], [9], [True], [-0.2]),
        )
    ]


# Explicit, independently checked best-fit geometry. The first split call has
# an empty owner; every owner still executes its four physical token rows.
WHOLE = ((0, 3), (1, 2))
SPLIT = (((0,), ()), ((1, 2), (3,)))


def loss(model, records, indices, coefficient, *, size=4):
    rows = [records[i] for i in indices]
    tokens = [token for row in rows for token in row["inputs"]["tokens"]]
    padding = size - len(tokens)
    assert padding >= 0
    inputs = torch.tensor([tokens + [0] * padding])
    lengths = torch.tensor(
        [len(row["inputs"]["tokens"]) for row in rows]
        + [
            min(model.config.max_seq_len, padding - start)
            for start in range(0, padding, model.config.max_seq_len)
        ],
        dtype=torch.int32,
    )
    observations = []
    for block in model.blocks:
        block.moe.observe = observations
    try:
        output = model(inputs, sequence_lengths=lengths).float()
    finally:
        for block in model.blocks:
            block.moe.observe = None
    fields = {
        key: torch.tensor(
            [[v for row in rows for v in row["loss_inputs"][key]] + [0] * padding]
        )
        for key in ("target_tokens", "loss_mask", "weights")
    }
    valid = fields["loss_mask"].bool()
    logprobs = (
        output.log_softmax(-1)
        .gather(-1, fields["target_tokens"].long()[..., None])
        .squeeze(-1)
    )
    primary = torch.where(valid, -logprobs * fields["weights"], 0).sum()
    if not tokens:
        return primary  # No router statistic or objective participation.
    routes = torch.stack(observations)[:, : len(tokens)].flatten(0, 1)
    counts = (
        F.one_hot(routes.topk(model.config.top_k, -1).indices, model.config.n_experts)
        .sum((0, 1))
        .float()
    )
    auxiliary = (
        model.config.n_experts
        * (counts / routes.shape[0] * routes.softmax(-1).mean(0)).sum()
    )
    return (
        primary
        + coefficient * auxiliary * torch.where(valid, fields["weights"], 0).sum()
    )


def options(*, grad_dtype="bfloat16"):
    requested, initial = initial_model(grad_dtype=grad_dtype)
    rows, coefficient = records(), 0.07
    optim = {
        name: float(torch.tensor(value))
        for name, value in dict(
            lr=0.002, beta1=0.8, beta2=0.95, eps=0.01, weight_decay=0.01
        ).items()
    }

    def snapshot(model, masters):
        with torch.no_grad():
            return {
                "weights": {
                    name: value.detach().float().tolist()
                    for name, value in masters.items()
                },
                "logprobs": [
                    model(torch.tensor([row["inputs"]["tokens"]]))
                    .float()[0]
                    .log_softmax(-1)
                    .gather(
                        -1, torch.tensor(row["loss_inputs"]["target_tokens"])[..., None]
                    )
                    .squeeze(-1)
                    .tolist()
                    for row in rows
                ],
            }

    oracles = {}
    for split in (False, True):
        model = deepcopy(initial)
        masters = {
            name: nn.Parameter(p.detach().float().clone())
            for name, p in model.named_parameters()
        }
        optimizer = torch.optim.AdamW(
            list(masters.values()),
            lr=optim["lr"],
            betas=(optim["beta1"], optim["beta2"]),
            eps=optim["eps"],
            weight_decay=optim["weight_decay"],
            foreach=False,
        )
        trajectory = [snapshot(model, masters)]
        observer = WeightGradients(model) if grad_dtype == "float32" else None
        for index, scale in enumerate((0.25, 1.0)):
            for owners in SPLIT if split and index == 0 else (WHOLE,):
                for indices in owners:
                    loss(model, rows, indices, coefficient).backward()
            if observer is not None:
                assert observer.values.keys() == masters.keys()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None
                gradient = (
                    parameter.grad.float()
                    if observer is None
                    else observer.values[name]
                )
                masters[name].grad = gradient * scale
            if observer is not None:
                observer.values.clear()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(masters[name])
            model.zero_grad(set_to_none=True)
            trajectory.append(snapshot(model, masters))
        if observer is not None:
            observer.close()
        oracles["split" if split else "whole"] = trajectory
    return dict(
        model=requested.to_dict(),
        objective=Objective.sum(
            {
                "tokens": Objective("shadowspill/cross_entropy@1"),
                "balance": Objective("shadowspill/moe_balance@1"),
            },
            weights={"balance": 0.001},
        ).to_dict(),
        optim=optim,
        controls={"balance.weight": coefficient},
        records=rows,
        mask="loss_mask",
        trainable=list(oracles["whole"][0]["weights"]),
        oracles=oracles,
        whole_loss=sum(
            float(loss(initial, rows, part, coefficient).detach()) for part in WHOLE
        ),
        loss_parts=[
            sum(
                float(loss(initial, rows, part, coefficient).detach())
                for part in owners
            )
            for owners in SPLIT
        ],
        whole_split_equivalence=False,
        output_name="tokens.logprobs",
        microbatch_sizes=[4],
        microbatch_counts=[1, 2],
        batch_limits={
            "token_capacity": 16,
            "max_sequence_length": 4,
            "max_sequences": 4,
        },
        parallelism=replace(
            ownership(2, 1),
            inputs=InputPlacement.independent(
                2,
                fields=[
                    InputFieldPlacement(path)
                    for path in (
                        ("inputs", "tokens"),
                        ("loss_inputs", "target_tokens"),
                        ("loss_inputs", "loss_mask"),
                        ("loss_inputs", "weights"),
                        ("loss_inputs", "router_mask"),
                    )
                ],
            ),
        ).to_dict(),
        preparation_timeout=1200,
        client_timeout=1800,
        device_gib=8,
        gradient_dtype=grad_dtype,
        separate_fp32_masters=True,
        weight_rtol=0.04,
        weight_atol=2e-5,
        weight_norm_rtol=0.03,
        weight_max_rtol=0.20,
        logprob_rtol=0.003,
        logprob_atol=0.008,
        loss_rtol=0.003,
        loss_atol=0.008,
        exact_resume=True,
        client_script="_text_service_client.py",
        model_description=(
            "Qwen3 full training, two EP workers, "
            f"BF16 compute/{grad_dtype} gradients, FP32 AdamW masters/moments"
        ),
        oracle_description=(
            "independent CPU per-expert projections with adjacent gate/up rows, "
            "explicit per-owner router balancing excluding padding, full vocabulary "
            "projection and AdamW; whole/split trajectories; "
            + (
                "FP32 weight products observed before BF16 leaf rounding"
                if grad_dtype == "float32"
                else "BF16 autograd gradients"
            )
        ),
    )
