"""Compare EP's first update to a combined-data, pure-PyTorch CPU oracle.

No QuackMoE, MoonEP, MLOps, or ShadowSpill operations run in the oracle. Expert
weights are assembled by their global expert IDs; replicated gradients SUM
the two local batches. Full-model routing is computed independently, so the
report records approximation error rather than claiming bitwise equivalence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from workloads.pytorch.olmoe import OLMoE, OLMoEConfig


class Experts(nn.Module):
    def __init__(self, c, router_dtype=torch.float32):
        super().__init__()
        self.router_weight = nn.Parameter(torch.empty(c.n_experts, c.d_model, dtype=router_dtype))
        self.gate_up_weight = nn.Parameter(torch.empty(c.n_experts, 2 * c.d_ff_expert, c.d_model,
                                                       dtype=torch.bfloat16))
        self.down_weight = nn.Parameter(torch.empty(c.n_experts, c.d_model, c.d_ff_expert,
                                                   dtype=torch.bfloat16))


class MoE(nn.Module):
    def __init__(self, c, router_dtype=torch.float32):
        super().__init__()
        self.config, self.experts = c, Experts(c, router_dtype)

    def forward(self, hidden, residual):
        c, weights = self.config, self.experts
        x = hidden.reshape(-1, c.d_model)
        logits = F.linear(x.to(weights.router_weight.dtype), weights.router_weight).float()
        probabilities = logits.softmax(-1)
        ids = logits.topk(c.top_k, dim=-1).indices
        selected = probabilities.gather(-1, ids)
        counts = torch.bincount(ids.flatten(), minlength=c.n_experts)
        auxiliary = c.n_experts * (
            counts.float() / (x.shape[0] * c.top_k) * probabilities.mean(0)
        ).sum()
        output = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(c.n_experts):
            token, slot = (ids == expert).nonzero(as_tuple=True)
            pre = F.linear(x[token], weights.gate_up_weight[expert])
            middle = (F.silu(pre[:, 0::2].float()) * pre[:, 1::2].float()).bfloat16()
            raw = F.linear(middle, weights.down_weight[expert])
            contribution = (raw.float() * selected[token, slot, None]).bfloat16()
            output = output.index_add(0, token, contribution.float())
        return residual + output.bfloat16().reshape_as(hidden), auxiliary


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = a.norm(), b.norm()
    return dict(
        max_absolute=float((a - b).abs().max()),
        relative_l2=float((a - b).norm() / nb.clamp_min(1e-30)),
        cosine=float(torch.dot(a, b) / (na * nb).clamp_min(1e-30)) if nb else 1.0,
        reference_l2=float(nb), observed_l2=float(na),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    directories = sorted(args.outdir.glob("rank-*/case.json"))
    metadata = json.loads(directories[0].read_text())
    c, world = OLMoEConfig(**metadata["config"]), metadata["world"]
    assert len(directories) == world
    initial, updated, batches, losses = [], [], [], []
    for path in directories:
        directory = path.parent
        initial.append(torch.load(directory / "initial.pt", weights_only=False, map_location="cpu"))
        updated.append(torch.load(directory / "step-001.pt", weights_only=False, map_location="cpu"))
        batches.append(torch.load(directory / "batches.pt", weights_only=True)[0])
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        losses.append(next(e["local_loss"] for e in events if e.get("step") == 1))
    model = OLMoE(c).bfloat16()
    for block in model.blocks:
        block.moe = MoE(c, initial[0]["model"]["blocks.0.moe.experts.router_weight"].dtype)
    expert_names = set(metadata["experts"])
    state = {}
    for name, value in initial[0]["model"].items():
        state[name] = torch.cat([rank["model"][name] for rank in initial]) if name in expert_names else value
    model.load_state_dict(state)
    gradients = {name: torch.zeros_like(parameter, dtype=torch.float32)
                 for name, parameter in model.named_parameters()}
    expected_losses = []
    for tokens, targets in batches:
        model.zero_grad(set_to_none=True)
        loss = model.loss(tokens, targets, reduction="sum", aux_coef=0.01) / (world * metadata["tokens"])
        loss.backward()
        expected_losses.append(float(loss.detach()))
        for name, p in model.named_parameters():
            gradients[name].add_(p.grad.float())
    comparisons = {}
    for index, name in enumerate(metadata["parameters"]):
        moments = [rank["optimizer"]["state"][index]["exp_avg"] for rank in updated]
        if name in expert_names:
            observed = torch.cat(moments).reshape_as(gradients[name]) / 0.1
        else:
            # SUM gradients have already been reduced; each rank owns a padded
            # contiguous moment shard. Concatenate once, dropping only padding.
            observed = torch.cat([value.flatten() for value in moments])[:gradients[name].numel()]
            observed = observed.reshape_as(gradients[name]) / 0.1
        comparisons[name] = metrics(observed, gradients[name])
    relative_loss = abs(sum(losses) - sum(expected_losses)) / max(abs(sum(expected_losses)), 1e-30)
    failed = [name for name, value in comparisons.items()
              if value["relative_l2"] > 0.05 or value["cosine"] < 0.995]
    report = dict(
        passed=not failed and relative_loss <= 0.001,
        tolerances=dict(loss_relative=0.001, gradient_relative_l2=0.05, gradient_cosine=0.995),
        observed_losses=losses, expected_losses=expected_losses,
        relative_loss=relative_loss, failed_parameters=failed, gradients=comparisons,
        note="Independent full-model routing; BF16 projection/fusion rounding can change near-tied routes.",
    )
    (args.outdir / "oracle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "gradients"}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
