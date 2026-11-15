"""Installed Llama preparation and packed semantics against the PyTorch model."""

import subprocess
import sys
from contextlib import ExitStack, contextmanager

import numpy as np
import pytest
import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.definitions import load_registry
from shadowspill.training.definitions.initialization import (
    InitializationContext,
    InitializationRuntime,
)
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.packing import pack_batch, select_schedule
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.inputs.validation import validate_batch
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    Datum,
    LoRA,
    Memory,
    Model,
    Objective,
    PlanningOptions,
    Preparation,
    Weights,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.payload import snapshot_batch
from shadowspill_workloads.models.llama3.config import Config
from shadowspill_workloads.models.llama3.implementations.pytorch import (
    Model as Reference,
)


def settings(*, capacity=4, lora=None):
    return Model(
        "shadowspill/llama3@1",
        config={
            "architecture": {
                "n_layers": 1,
                "d_model": 16,
                "n_heads": 2,
                "n_kv_heads": 1,
                "d_ff": 24,
                "vocab_size": 19,
                "max_seq_len": capacity,
            }
        },
        precision={"compute_dtype": "float32", "lora_dtype": "float32"},
        lora=lora,
        initialize=Weights.random(seed=17),
    )


def prepared(*, cap=8, length=4, count=4, sizes=(8,), counts=(1,), lora=None):
    registry = load_registry(["shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=settings(capacity=length, lora=lora),
        objective=Objective("shadowspill/cross_entropy@1"),
        batch=BatchConfig(
            limits={
                "token_capacity": cap,
                "max_sequence_length": length,
                "max_sequences": count,
            }
        ),
        memory=Memory(1, 1),
        operations={"forward_backward": ["metrics"], "forward": ["logprobs"]},
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=sizes),
            planning=PlanningOptions(microbatch_counts=counts),
        ),
    )
    return registry, setup, prepare_layouts(setup)


@contextmanager
def bound(setup, layouts, datums, *, forward=False):
    key = (
        OperationKey("forward", "logprobs", "evaluation")
        if forward
        else (OperationKey("forward_backward", "metrics"))
    )
    candidate = layouts.families[key][0]
    binding = candidate.bindings[0, 0]
    budget = BufferBudget(8 << 20, max_requests=8)
    with ExitStack() as stack:
        snapshot = stack.enter_context(snapshot_batch(datums, budget))
        batch = validate_batch(
            snapshot,
            setup.context.schema,
            setup.context.batches[0],
            required_paths=binding.contract.required_paths,
        )
        assignment = pack_batch(
            batch,
            setup.context.schema,
            batch.config,
            setup.context.parallelism.inputs,
            candidate.candidate.schedule,
        )
        physical = stack.enter_context(
            materialize_buffer(assignment, setup.context.schema, 0, budget)
        )
        adapted = stack.enter_context(
            binding.inputs.adapt(
                route_buffer(physical, setup.context.parallelism.inputs, 0), budget
            )
        )
        yield binding, adapted
    assert budget.used == (0, 0)


def datums():
    return [
        Datum(
            {"tokens": [1, 2, 3]},
            {
                "target_tokens": [2, np.nan, 4],
                "loss_mask": [True, False, True],
                "weights": [1.0, np.nan, -0.3],
            },
        ),
        Datum({"tokens": []}, {"target_tokens": [], "loss_mask": [], "weights": []}),
        Datum(
            {"tokens": [5, 6]},
            {
                "target_tokens": [6, 7],
                "loss_mask": [True, True],
                "weights": [0, 0.7],
            },
        ),
    ]


def test_setup_roundtrip_padding_metadata_and_capture_inventory():
    registry, setup, layouts = prepared()
    assert resolve_training(registry, setup.config).config == setup.config
    inventory = capture_inventory(layouts)
    assert inventory.templates
    assert all(
        t.model_build.grad_dtype == "float32" for t in inventory.templates.values()
    )
    with bound(setup, layouts, datums()) as (_, adapted):
        inputs, loss = adapted.torch_inputs()
        assert inputs["tokens"].shape == (1, 8)
        assert inputs["sequence_lengths"].tolist() == [3, 2, 0, 3, 0, 0]
        assert loss["loss_mask"].sum() == 4
        assert loss["target_tokens"][0, 1] == 0
    with bound(setup, layouts, []) as (_, adapted):
        inputs, _ = adapted.torch_inputs()
        assert inputs["sequence_lengths"].tolist() == [4, 4, 0, 0, 0, 0]


@pytest.mark.parametrize("lora", [None, LoRA(2, targets=["blocks.*.attn.wq"])])
def test_packed_cpu_capture_loss_and_all_gradients_match_independent_sequences(lora):
    from shadowspill_workloads.lora import configure_lora

    _, setup, layouts = prepared(lora=lora)
    torch.manual_seed(11)
    model = setup.model.create()
    reference = Reference(Config(**dict(setup.model.config["architecture"])))
    if lora is not None:
        reference = configure_lora(
            reference, rank=2, alpha=2, targets=lora.targets, factor_dtype="float32"
        )
    reference.load_state_dict(model.state_dict())
    expected = torch.zeros(())
    for datum in datums():
        if not datum.inputs["tokens"]:
            continue
        logits = reference(torch.tensor([datum.inputs["tokens"]]))[0]
        fields = datum.loss_inputs
        active = torch.tensor(fields["loss_mask"])
        targets = torch.tensor(
            [
                int(t) if v else 0
                for t, v in zip(
                    fields["target_tokens"], fields["loss_mask"], strict=True
                )
            ]
        )
        weights = torch.tensor(
            [
                w if v else 0
                for w, v in zip(fields["weights"], fields["loss_mask"], strict=True)
            ]
        )
        expected += (
            torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            * weights
            * active
        ).sum()
    with bound(setup, layouts, datums()) as (binding, adapted):
        args = binding.arguments(adapted)
        captured = capture_training_objective(model, binding.objective, args)
        actual, loss_sum, count = captured.exported.exported_program.module()(*args)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(loss_sum, expected.detach(), rtol=1e-5, atol=2e-6)
        assert count == 4
        actual.backward()
        expected.backward()
        for (name, value), (other, target) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == other
            if value.requires_grad:
                torch.testing.assert_close(
                    value.grad, target.grad, rtol=3e-5, atol=3e-6
                )


@pytest.mark.parametrize("lengths", [[40, 40, 40], [38, 32, 19, 13, 13, 13], [1] * 64])
def test_q25_count_three_is_ineligible_and_larger_shape_covers_awkward_lengths(lengths):
    _, setup, layouts = prepared(
        cap=128 * 1024,
        length=64 * 1024,
        count=64,
        sizes=(64 * 1024, 128 * 1024),
        counts=(1, 2, 3),
    )
    key = OperationKey("forward", "logprobs", "evaluation")
    for candidate in layouts.families[key]:
        assert (
            candidate.candidate.schedule.capacities(setup.context.schema)[
                "token_capacity"
            ]
            <= 128 * 1024
        )
    largest = next(
        row
        for row in layouts.families[key]
        if (
            row.candidate.schedule.buffers[0].fields[("inputs", "tokens")]
            == (128 * 1024,)
        )
    )
    assert largest.candidate.schedule.buffers[0].max_datums == 64
    assert any(row.stage == "capacity" for row in layouts.rejected)
    datums = [
        Datum(
            {"tokens": np.ones(n * 1024, dtype=np.int64)},
            {"target_tokens": np.zeros(n * 1024, dtype=np.int64)},
        )
        for n in lengths
    ]
    budget = BufferBudget(16 << 20)
    with snapshot_batch(datums, budget) as snapshot:
        batch = validate_batch(
            snapshot,
            setup.context.schema,
            setup.context.batches[0],
            required_paths=largest.bindings[0, 0].contract.required_paths,
        )
        result = select_schedule(
            batch,
            setup.context.schema,
            batch.config,
            setup.context.parallelism.inputs,
            [row.candidate.schedule for row in layouts.families[key]],
        )
        assert sorted(i for group in result.records for i in group) == list(
            range(len(lengths))
        )
        if lengths == [40, 40, 40]:
            assert len(result.schedule.buffers) == 1
            assert result.schedule.buffers[0].fields[("inputs", "tokens")] == (
                128 * 1024,
            )
    assert budget.used == (0, 0)


def test_random_loader_is_chunk_invariant_and_initializes_derived_rotary(tmp_path):
    _, setup, _ = prepared()
    model = setup.model.create()
    spec = setup.model.describe_initialization(
        InitializationContext(
            setup.model.settings,
            setup.model.settings.initialize,
            setup.context.parallelism,
            0,
        )
    )
    runtime = InitializationRuntime(
        tmp_path, lambda _: pytest.fail("unexpected artifact")
    )
    with spec.open(runtime) as fill:
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            count = tensor.numel()
            whole = torch.empty(count, dtype=tensor.dtype)
            pieces = torch.empty_like(whole)
            fill(TensorChunk(name, tuple(tensor.shape), tensor.dtype, 0, count), whole)
            for start in range(0, count, 7):
                stop = min(start + 7, count)
                fill(
                    TensorChunk(name, tuple(tensor.shape), tensor.dtype, start, count),
                    pieces[start:stop],
                )
            torch.testing.assert_close(whole, pieces, rtol=0, atol=0)
            if name.startswith("rotary."):
                torch.testing.assert_close(whole, tensor.flatten(), rtol=0, atol=0)


def test_forward_capture_accepts_changed_packing_without_rollout_or_loss_fields():
    _, setup, layouts = prepared()
    model = setup.model.create()
    calls = [
        [Datum({"tokens": [1, 2, 3]}, {"target_tokens": [2, 3, 4]})],
        [
            Datum({"tokens": [1, 2]}, {"target_tokens": [3, 4]}),
            Datum({"tokens": [5, 6, 7, 8]}, {"target_tokens": [6, 7, 8, 9]}),
        ],
    ]
    exported = None
    for records in calls:
        with bound(setup, layouts, records, forward=True) as (binding, adapted):
            args = binding.arguments(adapted)

            # Capture the target-logprob computation without a training loss.
            def computation(model, inputs, loss_inputs, controls):
                return setup.objective.computation(binding.contract.key)(
                    model, inputs, loss_inputs, controls
                )["logprobs"]

            if exported is None:

                class Forward(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.model = model

                    def forward(self, inputs, loss_inputs, controls):
                        return computation(self.model, inputs, loss_inputs, controls)

                exported = torch.export.export(Forward(), tuple(args)).module()
            actual = exported(*args)
            # The physical order is size-sorted, independent of original order.
            start = 0
            for record in sorted(records, key=lambda row: -len(row.inputs["tokens"])):
                length = len(record.inputs["tokens"])
                expected = model.target_logprobs(
                    torch.tensor([record.inputs["tokens"]]),
                    torch.tensor([record.loss_inputs["target_tokens"]]),
                )
                torch.testing.assert_close(actual[:, start : start + length], expected)
                start += length


@pytest.mark.parametrize(
    "config",
    [
        {"architecture": {"n_heads": 0}},
        {"architecture": {"n_kv_heads": -1}},
        {"architecture": {"norm_epsilon": float("nan")}},
        {"grad_dtype": "float8"},
        {"implementation": "unknown"},
        {"lora": {"rank": 2, "dropout": 0.1}},
    ],
)
def test_invalid_structural_settings_fail_on_cpu(config):
    from shadowspill_workloads.definitions.llama.configuration import Settings

    with pytest.raises(ValueError):
        Settings(**config)


def test_description_imports_no_framework_or_model_implementation():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name == "torch" or name == "triton" or ".implementations." in name:
            raise AssertionError("unexpected import: " + name)
sys.meta_path.insert(0, Block())
from shadowspill.training.definitions import load_registry
from shadowspill_client import Model
r = load_registry(["shadowspill-workloads"])
m = r.resolve_model("shadowspill/llama3@1", settings=Model("shadowspill/llama3@1"))
assert m.config["grad_dtype"] == "float32"
assert "mlops" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
