"""Save/recompute head-loss training check through unmodified ShadowSpill."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import mlops
import torch
from torch import nn
from torch.nn import functional as F

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.ir import TaskAlternativeChoice
from shadowspill.training import Trainer
from shadowspill.training.backends import ShadowSpill


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(128, 128, bias=False)
        self.head = nn.Parameter(torch.randn(1024, 128) * 0.02, requires_grad=False)
        self.lora_a = nn.Parameter(torch.randn(16, 128) * 0.02)
        self.lora_b = nn.Parameter(torch.randn(1024, 16) * 0.02)

    def forward(self, hidden, targets):
        hidden = self.projection(hidden)
        return mlops.lora_head_loss(
            hidden, self.head, self.lora_a, self.lora_b, targets,
            scale=0.7, chunk_size=23,
        )


def objective(model, batch):
    return model(*batch)


def reference(model, batch):
    hidden, targets = batch
    x = model.projection(hidden)
    logits = x @ model.head.T + 0.7 * (x @ model.lora_a.T) @ model.lora_b.T
    return F.cross_entropy(logits, targets, reduction="sum") / targets.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("save", "recompute"), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(481)
    source = Model()
    oracle = copy.deepcopy(source)
    before_head = source.head.detach().clone()
    optimizer = torch.optim.SGD(
        (p for p in oracle.parameters() if p.requires_grad), lr=0.1, foreach=False,
    )
    data = torch.randn(67, 128), torch.randint(0, 1024, (67,))
    data[1][2] = -100
    def choices(program, _shares):
        return (tuple(
            TaskAlternativeChoice(group.group_id, args.variant)
            for group in program.task_alternative_groups
        ),)
    rows = []
    with (
        patch("shadowspill.planner.search.algorithms.pressurefit.resolutions", choices),
        ShadowSpill(
            device="cuda:0", execution_gib=2, spill_gib=1,
            artifact_store=args.outdir / "artifacts", partition="whole",
            profiling_options=CORRECTNESS_PROFILING,
        ) as backend,
        Trainer(
            source, objective=objective, optimizer=torch.optim.SGD,
            optimizer_args={"lr": 0.1, "foreach": False}, backend=backend,
        ) as trainer,
    ):
        trainer.prepare(data)
        call = trainer._execution.call
        stages = call.plan_report.diagnostics.unique_stages
        pairs = [stage.as_dict() for stage in stages]
        (args.outdir / "graphpairs.json").write_text(json.dumps(pairs, indent=2))
        selections = call.plan_report.execution_plan.selections
        print("SELECTIONS", selections, flush=True)
        assert selections and all(choice.option_id == args.variant for choice in selections)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            expected = reference(oracle, data)
            expected.backward()
            optimizer.step()
            actual = trainer.step(data)
            torch.testing.assert_close(
                torch.tensor(actual.loss), expected.detach(), atol=3e-6, rtol=3e-6,
            )
            rows.append({"step": step, "loss": actual.loss, "reference_loss": expected.item()})
            print("STEP", rows[-1], flush=True)
        checkpoint = call.state_dict()
        state = checkpoint["model"]
        torch.testing.assert_close(state, oracle.state_dict(), atol=3e-6, rtol=3e-5)
        assert torch.equal(state["head"], before_head)
        table = ["| Stage | Variant | Direction | Input bytes | Mutated bytes | Output bytes | Workspace bytes | Runtime ms |",
                 "|---|---|---|---:|---:|---:|---:|---:|"]
        for stage in stages:
            for pair in stage.graph_pairs:
                for direction, profile in (("fwd", pair.forward), ("bwd", pair.backward)):
                    if profile is None:
                        continue
                    table.append(
                        f"| {stage.unique_stage_id} | {pair.variant} | {direction} | "
                        f"{profile.input_allocation_bytes} | {profile.mutation_allocation_bytes} | "
                        f"{profile.output_allocation_bytes} | {profile.task_workspace_bytes} | "
                        f"{profile.runtime_ns / 1e6:.4f} |"
                    )
        (args.outdir / "graphpairs.md").write_text("\n".join(table) + "\n")
        print("\n".join(table), flush=True)
    summary = {"variant": args.variant, "status": "passed", "steps": rows, "frozen_head_unchanged": True}
    (args.outdir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PASSED", args.variant, flush=True)


if __name__ == "__main__":
    main()
