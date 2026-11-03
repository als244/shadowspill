"""Combined region/chunk/retention fixture with a globally normalized DP oracle."""

import json
from collections.abc import Mapping
from dataclasses import replace

import torch

from shadowspill import Parallelism, build_forward_backward, build_step
from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule
from shadowspill.pytorch.chunking import AxisChunks, ChunkRegions
from shadowspill.pytorch.partition import IsolateOperations
from tests.shadowspill.execution._oracle import optimizer_values
from tests.shadowspill.pytorch.chunking.test_regions import RegionChunks


class Block(torch.nn.Module):
    def __init__(self, lora, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(7, 11, device=device, dtype=dtype) / 7,
            requires_grad=not lora,
        )
        self.a = torch.nn.Parameter(torch.randn(7, 3, device=device, dtype=dtype) / 7)
        self.b = torch.nn.Parameter(torch.randn(3, 11, device=device, dtype=dtype) / 7)
        self.out = torch.nn.Parameter(
            torch.randn(11, 7, device=device, dtype=dtype) / 11,
            requires_grad=not lora,
        )

    def forward(self, x):
        projected = x @ self.weight + (x @ self.a) @ self.b
        return torch.nn.functional.silu(projected.softmax(1) @ self.out) + x


class Network(torch.nn.Module):
    def __init__(self, lora, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [Block(lora, device=device, dtype=dtype) for _ in range(2)]
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


def oracle(kind, lora):
    torch.manual_seed(731)
    model = Network(lora)
    initial = parameters(model)
    calls = examples()
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
    return dict(
        initial=initial,
        examples=calls,
        expected=expected,
        precision="float32",
        trace=True,
        exact_names=[
            name for name, value in model.named_parameters() if not value.requires_grad
        ],
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
    execution = ChunkRegions(
        IsolateOperations(
            ("aten._softmax.default", "aten._softmax_backward_data.default"),
            base="whole",
        ),
        (ComposedChunks if options["nested"] else RegionChunks)(17, 2),
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
    fixture = oracle(kind, options["lora"])
    template = Network(options["lora"], device="meta")
    common = dict(
        objective=objective,
        example_calls=fixture["examples"][0],
        grad_dtype=torch.float32,
        **policies(options),
    )
    program = (
        build_step(
            template,
            optimizer=torch.optim.AdamW(template.parameters(), lr=0.001, foreach=False),
            master_dtype=None,
            **common,
        )
        if kind == "step"
        else build_forward_backward(template, **common)
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
                graph_pair_policy=common["graph_pair_policy"].identity(),
                parallelism=common["parallelism"].to_dict(),
            ),
            indent=2,
        )
        + "\n"
    )
    assert not torch.distributed.is_initialized()
