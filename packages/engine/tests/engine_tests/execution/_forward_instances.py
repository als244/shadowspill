"""Consume saved count Plans without model, capture, compilation or planning."""

import sys
from pathlib import Path

import torch
import triton
from torch.utils._pytree import tree_flatten

from engine_tests.execution._cold import forbid_preparation


def compilation_finished(**event):
    assert event["cache_hit"], f"unexpected compilation: {event['src'].name}"


triton.knobs.compilation.listener = compilation_finished
forbid_preparation()
from shadowspill import Executor, Plan  # noqa: E402
from shadowspill.training.observations import to_host  # noqa: E402

root = Path(sys.argv[1])
plans = {
    name: Plan.load(root / f"{name}.ssp") for name in ("count", "single", "repeated")
}
oracle = torch.load(root / "oracle.pt", weights_only=True)


def fill(request, chunk):
    assert chunk.nbytes <= 32
    value = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(value[request.offset : request.offset + chunk.numel()])


def compare(actual, expected):
    actual_leaves, actual_tree = tree_flatten(to_host(actual))
    expected_leaves, expected_tree = tree_flatten(expected)
    assert actual_tree == expected_tree
    for left, right in zip(actual_leaves, expected_leaves, strict=True):
        if isinstance(right, torch.Tensor):
            torch.testing.assert_close(left, right)
        else:
            assert left == right


with Executor(plans.pop("count"), plans=plans) as executor:
    executor.initialize(fill, chunk_bytes=32)
    retained = []
    for index, name in enumerate((None, "single", "repeated")):
        args = oracle["calls"][index]
        if index == 2:
            result = executor.run(
                plan=name, input_loader=lambda i, values=args: (values[i], {})
            ).result()
        else:
            result = executor.run(plan=name, args=args).result()
        retained.append(result)
        for previous, expected in zip(retained, oracle["expected"], strict=False):
            compare(previous, expected)
        count = {}

        def export(request, chunk, destination=count):
            if request.name == "counter":
                destination["counter"] = chunk.clone()

        executor.export_tensors(export, chunk_bytes=32)
        assert count["counter"].item() == (3, 4, 6)[index]
        outputs = result if isinstance(result, tuple) else (result,)
        assert len(
            {item["input"].untyped_storage().data_ptr() for item in outputs}
        ) == len(outputs)
        for item in outputs:
            assert (
                item["input"].untyped_storage().data_ptr()
                == item["view"].untyped_storage().data_ptr()
            )
    compare(retained[0], oracle["expected"][0])
    del retained, result, previous, outputs, item
    executor.synchronize()
print("Qualified three saved-plan invocations", flush=True)
