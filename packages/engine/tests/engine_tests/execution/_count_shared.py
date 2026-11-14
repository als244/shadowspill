"""Execute newly instantiated counts using only the source compiled task forms."""

from contextlib import ExitStack, contextmanager, nullcontext

import torch

from shadowspill.parallelism import InputFieldPlacement, InputPlacement
from shadowspill.training.inputs.adapters import InputAdapter, InputBinding, InputLayout
from shadowspill.training.inputs.coverage import certify_coverage
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.inputs.shapes import BufferShape, PackingSchedule
from shadowspill.training.inputs.validation import validate_batch
from shadowspill.training.observations import to_host
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch


@contextmanager
def certified_examples(examples):
    paths = (("inputs", "x"), ("loss_inputs", "target"), ("loss_inputs", "mask"))
    schema = InputSchema(
        [TensorField(path, "float32", ("positions", 4), "pack", 0) for path in paths],
        [
            Capacity("positions", "positions", "sum", paths[0], 0, scheduled=True),
            Capacity("length", "positions", "max", paths[0], 0),
            Capacity("records", "records", "count"),
        ],
    )
    limits = BatchConfig(limits={"positions": 6, "length": 2, "records": 3})
    shape = BufferShape(3, dict.fromkeys(paths, (2, 4)))
    schedule = PackingSchedule("count_repeat", [[shape] * 3])
    placement = InputPlacement.independent(
        1, fields=[InputFieldPlacement(path) for path in paths]
    )
    proof = certify_coverage(schema, limits, placement, [schedule])
    assert proof.construction == "slots"
    binding = InputBinding(
        InputLayout(schema, shape, placement, 0), InputAdapter(padding_safe=True)
    )
    budget = BufferBudget(8 << 20, max_requests=8)
    with ExitStack() as stack:
        snapshot = stack.enter_context(
            snapshot_batch(
                [
                    Datum({"x": x}, {"target": target, "mask": mask})
                    for x, target, mask in examples
                ],
                budget,
            )
        )
        batch = validate_batch(snapshot, schema, limits)
        assignment = proof.pack(batch)
        result = []
        for index in range(3):
            buffer = stack.enter_context(
                materialize_buffer(assignment, schema, index, budget)
            )
            adapted = stack.enter_context(
                binding.adapt(route_buffer(buffer, placement, 0), budget)
            )
            inputs, loss_inputs = adapted.torch_inputs()
            result.append((inputs["x"], loss_inputs["target"], loss_inputs["mask"]))
        yield result
    assert budget.used == (0, 0)


def qualify_count_family(executor, plans, oracle, snapshot, fill):
    executor.initialize(fill, chunk_bytes=32)
    for name, expected in zip(
        ("count_repeat", "count_mixed", "count_backward"),
        oracle["count_updates"],
        strict=True,
    ):
        examples = [oracle["examples"][index] for index in expected["indices"]]
        if name == "count_backward":
            first = executor.run(plan=name, args=examples[:2]).result()
            first = to_host((first.losses, first.metrics))
            last = executor.run(
                plan="backward1", args=examples[2:], accumulate_gradients=True
            ).result()
            last = to_host((last.losses, last.metrics))
            losses, metrics = (*first[0], *last[0]), (*first[1], *last[1])
            executor.run(
                plan="optim",
                hyperparams={
                    plans["optim"].program.bindings["gradient_scale_control"]: expected[
                        "scale"
                    ]
                },
            ).result()
        else:
            with (
                certified_examples(examples)
                if name == "count_repeat"
                else nullcontext(examples)
            ) as inputs:
                result = executor.run(
                    plan=name,
                    args=inputs,
                    hyperparams={
                        plans[name].program.bindings[
                            "gradient_scale_control"
                        ]: expected["scale"]
                    },
                ).result()
                losses, metrics = to_host((result.losses, result.metrics))
        for loss, prediction, observed_loss, observed in zip(
            expected["losses"], expected["predictions"], losses, metrics, strict=True
        ):
            torch.testing.assert_close(observed_loss, loss)
            torch.testing.assert_close(observed["prediction"], prediction)
        actual = snapshot(executor)
        for key, value in expected["state"].items():
            torch.testing.assert_close(actual[key].reshape(value.shape), value)
