"""Combined region/chunk/retention fixture with a globally normalized DP oracle."""

import json
from collections.abc import Mapping
from dataclasses import replace

import torch
from torch._inductor import config as inductor_config
from torch.utils._pytree import tree_map

from shadowspill import Parallelism, build_forward_backward, build_step
from shadowspill.pytorch import (
    ExecutionLifetimes,
    GraphPairPolicy,
    LifetimeRule,
    LifetimeSelection,
    RetentionPolicy,
    RetentionRule,
)
from shadowspill.pytorch.chunking import AxisChunks, ChunkRegions
from shadowspill.pytorch.partition import IsolateOperations
from tests.shadowspill.execution._oracle import optimizer_values
from tests.shadowspill.pytorch.chunking.test_regions import RegionChunks


class Block(torch.nn.Module):
    def __init__(
        self,
        lora,
        *,
        device="cpu",
        dtype=torch.float32,
        gate_dtype=None,
        lifetime_mode=None,
    ):
        super().__init__()
        self.gate_dtype = gate_dtype
        self.lifetime_mode = lifetime_mode
        width = 18 if lifetime_mode == "replay_compact" else 11
        self.weight = torch.nn.Parameter(
            torch.randn(7, width, device=device, dtype=dtype) / 7,
            requires_grad=not lora,
        )
        self.a = torch.nn.Parameter(torch.randn(7, 3, device=device, dtype=dtype) / 7)
        self.b = torch.nn.Parameter(
            torch.randn(3, width, device=device, dtype=dtype) / 7
        )
        self.out = torch.nn.Parameter(
            torch.randn(11, 7, device=device, dtype=dtype) / 11,
            requires_grad=not lora,
        )

    def forward(self, x):
        early = x.sin() if self.lifetime_mode is not None else x
        source = early if self.lifetime_mode == "replay_compact" else x
        projected = source @ self.weight + (source @ self.a) @ self.b
        if self.lifetime_mode == "replay_compact":
            projected, gate = projected.split((11, 7), dim=-1)
        activated = torch.nn.functional.silu(projected.softmax(1) @ self.out)
        if self.lifetime_mode == "replay_compact":
            activated = activated * gate.sin()
        if self.gate_dtype is not None:
            activated = activated * activated.to(self.gate_dtype).sigmoid().float()
        return activated + early


class Network(torch.nn.Module):
    def __init__(
        self,
        lora,
        *,
        device="cpu",
        dtype=torch.float32,
        gate_dtype=None,
        lifetime_mode=None,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                Block(
                    lora,
                    device=device,
                    dtype=dtype,
                    gate_dtype=gate_dtype,
                    lifetime_mode=lifetime_mode,
                )
                for _ in range(2)
            ]
        )
        self.blocks[1].out = self.blocks[0].out
        self.blocks[1].b = self.blocks[0].b

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def objective(model, x, *, labels, sample_weights):
    return ((model(x) - labels).square() * sample_weights).sum(), {}


def examples():
    generator = torch.Generator().manual_seed(95)
    calls = []
    count = 0
    for rank in range(2):
        local = []
        for microbatch in range(2):
            mask = (torch.arange(17) < 17 - 3 * rank - 2 * microbatch).float()
            mask = mask.reshape(1, 17, 1).expand(2, 17, 1).clone()
            count += int(mask.sum()) * 7
            local.append(
                (
                    (torch.randn(2, 17, 7, generator=generator),),
                    {
                        "labels": torch.randn(2, 17, 7, generator=generator),
                        "sample_weights": mask,
                    },
                )
            )
        calls.append(local)
    for local in calls:
        for _, kwargs in local:
            kwargs["sample_weights"].div_(count)
    return calls


def parameters(model):
    # Canonical names match shard export; state_dict also contains tied aliases.
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def oracle(kind, lora, *, gate_dtype=None, lifetime_mode=None, device="cpu"):
    torch.manual_seed(731)
    model = Network(
        lora, gate_dtype=gate_dtype, lifetime_mode=lifetime_mode, device=device
    )
    initial = parameters(model)
    calls = tree_map(lambda value: value.to(device), examples())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    expected = []
    for invocation in range(3):
        accumulate = kind == "forward_backward" and invocation == 1
        if not accumulate:
            model.zero_grad(set_to_none=True)
        outputs = []
        for local in calls:
            values = []
            for args, kwargs in local:
                loss, _ = objective(model, *args, **kwargs)
                loss.backward()
                values.append(loss.detach().clone())
            outputs.append(values)
        metrics = {}
        state = {}
        if kind == "step":
            if invocation != 1:
                metrics = {
                    name: {
                        "param_norm": value.detach().norm(),
                        "grad_norm": value.grad.norm(),
                    }
                    for name, value in model.named_parameters()
                    if value.requires_grad
                }
            optimizer.step()
            state.update(optimizer_values(model, optimizer))
        else:
            assert kind == "forward_backward"
            state.update(
                {
                    f"gradient.{name}": value.grad.detach().clone()
                    for name, value in model.named_parameters()
                    if value.requires_grad
                }
            )
        state.update(parameters(model))
        expected.append(
            dict(
                outputs=outputs,
                state=state,
                accumulate=accumulate,
                parameter_metrics=metrics,
            )
        )
    fixture = dict(
        initial=initial,
        examples=calls,
        expected=expected,
        precision="float32",
        trace=True,
        exact_names=[
            name for name, value in model.named_parameters() if not value.requires_grad
        ],
    )
    return tree_map(
        lambda value: (
            value.detach().cpu() if isinstance(value, torch.Tensor) else value
        ),
        fixture,
    )


class InnerChunks:
    def configure_stage(self, stage):
        axes = {
            index: domains["tokens"]
            for index, domains in enumerate(stage.input_domains)
            if "tokens" in domains
        }
        size = 7 if any("silu" in target for target in stage.operators) else 3
        return AxisChunks(size, axes, domain="tokens") if axes else None


class ComposedChunks(RegionChunks):
    def configure_stage(self, stage):
        selected = super().configure_stage(stage)
        if selected is None or not any("silu" in target for target in stage.operators):
            return selected
        return replace(
            selected,
            chunk_size=11,
            regions=ChunkRegions(
                IsolateOperations(
                    ("aten.silu.default", "aten.silu_backward.default"), base="whole"
                ),
                InnerChunks(),
            ),
        )


class LifetimeChoices:
    """The selector must be resolved during build, never called by execution."""

    def __init__(self, mode):
        assert mode in {"move", "replay_compact"}
        self.mode = mode
        self.building = True
        self.calls = 0

    def __call__(self, operation):
        assert self.building, "lifetime selector ran after build"
        self.calls += 1
        if operation.target == "aten.sin.default":
            return (
                LifetimeSelection(move=True)
                if self.mode == "move"
                else LifetimeSelection(replay=True)
            )
        if self.mode == "replay_compact" and operation.target in {
            "aten.view.default",
            "aten._unsafe_view.default",
            "aten.slice.Tensor",
            "aten.split.Tensor",
            "aten.split_with_sizes.default",
            "<built-in function getitem>",
        }:
            return LifetimeSelection(compact=True)
        return None


def check_lifetime_reports(reports, mode):
    """Require real transformations and protected memory evidence, not opt-in alone."""
    assert reports
    required = ("schedule",) if mode == "move" else ("compaction", "replay")
    counts = {}
    for kind in ("schedule", "compaction", "replay"):
        parts = [(report.get(kind) or {}) for report in reports]
        counts[kind] = sum(
            decision["applied"]
            for part in parts
            for decision in part.get("decisions", ())
        )
        assert all(
            after <= before
            for part in parts
            for _, before, after in part.get("barrier_bytes", ())
        ), kind
    assert all(counts[kind] > 0 for kind in required), counts
    if mode == "replay_compact":
        selected = {
            item["target"]
            for report in reports
            for item in report["selections"]
            if item.get("replay")
        }
        assert selected == {"aten.sin.default"}, selected
    return counts


def policies(options):
    storage = {
        "replicated": ("replicated", "replicated", "replicated"),
        "optimizer": ("replicated", "replicated", "sharded"),
        "gradient_optimizer": ("replicated", "sharded", "sharded"),
        "sharded": ("sharded", "sharded", "sharded"),
        "cutoff": ("sharded", "sharded", "sharded"),
    }
    weights, gradients, optimizer = storage[options["storage"]]
    parallelism = Parallelism.dp(
        ranks_per_replica=2,
        weights=weights,
        gradients=gradients,
        optimizer=optimizer,
        min_shard_bytes=100 if options["storage"] == "cutoff" else 0,
    )
    checkpoint = (
        IsolateOperations(("aten.softmax.int",), base="auto")
        if options["checkpoint_cuts"]
        else "auto"
    )
    barriers = ("aten._softmax.default", "aten._softmax_backward_data.default")
    lifetime_mode = options.get("lifetime_mode")
    lifetimes = None
    if lifetime_mode:
        if lifetime_mode == "replay_compact":
            barriers += ("aten.silu.default",)
        lifetimes = ExecutionLifetimes(
            rules=tuple(LifetimeRule(target, barrier=True) for target in barriers),
            select=LifetimeChoices(lifetime_mode),
        )
    execution = ChunkRegions(
        IsolateOperations(
            barriers,
            base="whole",
        ),
        (ComposedChunks if options["nested"] else RegionChunks)(17, 2),
        lifetimes=lifetimes,
    )
    retention = GraphPairPolicy(
        save=RetentionPolicy(default="cost"),
        recompute=RetentionPolicy(
            default="recompute",
            rules=(RetentionRule("aten._softmax.default", "retain"),)
            if options["retain_sequence"]
            else (),
        ),
    )
    return dict(
        partition=checkpoint,
        chunking=execution,
        graph_pair_policy=retention,
        parallelism=parallelism,
    )


def build(path, kind):
    options = json.loads((path / "regions.json").read_text())
    gate_dtype = (
        getattr(torch, options["gate_dtype"]) if options.get("gate_dtype") else None
    )
    lifetime_mode = options.get("lifetime_mode")
    emulate = gate_dtype is not None or lifetime_mode is not None
    # CUDA's native low-precision derivative differs from the CPU implementation.
    # The independent reference uses ordinary CUDA autograd, never our compiler.
    fixture = oracle(
        kind,
        options["lora"],
        gate_dtype=gate_dtype,
        lifetime_mode=lifetime_mode,
        device="cuda" if emulate else "cpu",
    )
    template = Network(
        options["lora"],
        device="meta",
        gate_dtype=gate_dtype,
        lifetime_mode=lifetime_mode,
    )
    common = dict(
        objective=objective,
        example_calls=fixture["examples"][0],
        grad_dtype=torch.float32,
        **policies(options),
    )
    with inductor_config.patch(emulate_precision_casts=emulate):
        program = (
            build_step(
                template,
                optimizer=torch.optim.AdamW(
                    template.parameters(), lr=0.001, foreach=False
                ),
                master_dtype=None,
                **common,
            )
            if kind == "step"
            else build_forward_backward(template, **common)
        )
    if gate_dtype is not None:
        assert any(
            "cuda.sigmoid_backward.scalar_rounding.v1" in row["numerical_contracts"]
            for row in program.bindings["task_metadata"].values()
        ), "The fixture did not exercise compiled native scalar rounding"
    if emulate:
        assert any(
            "cuda.silu_backward.opmath.v1" in row["numerical_contracts"]
            for row in program.bindings["task_metadata"].values()
        ), "The fixture did not exercise compiled native SiLU opmath"
    lifetime_counts = {}
    if lifetime_mode:
        selector = common["chunking"].lifetimes.select
        assert selector.calls > 0
        selector.building = False
        lifetime_counts = check_lifetime_reports(
            [
                report
                for row in program.bindings["task_metadata"].values()
                for report in row.get("execution_lifetimes", ())
            ],
            lifetime_mode,
        )
    sizes, nested = set(), []

    def inspect(value):
        if isinstance(value, Mapping):
            if "row_chunks" in value:
                sizes.add(value["row_chunks"]["chunk_size"])
            if "child_regions" in value:
                nested.append(value)
            for child in value.values():
                inspect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                inspect(child)

    inspect(program.bindings["task_metadata"])
    assert {3, 7} <= sizes, sizes
    assert bool(nested) == options["nested"]
    assert program.bindings.get("num_examples") == 2
    if options["checkpoint_cuts"]:
        assert all(len(stages) > 2 for stages in program.bindings["training_templates"])
    program.save(path / "program.ssp")
    torch.save(fixture, path / "oracle.pt")
    (path / "regions-built.json").write_text(
        json.dumps(
            dict(
                options=options,
                resolved_chunk_sizes=sorted(sizes),
                nested_compositions=len(nested),
                lifetime_decisions=lifetime_counts,
                graph_pair_policy=common["graph_pair_policy"].identity(),
                parallelism=common["parallelism"].to_dict(),
            ),
            indent=2,
        )
        + "\n"
    )
    assert not torch.distributed.is_initialized()
