"""Unused inputs retain their guards without requiring spill allocations."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch.utils._pytree import treespec_dumps

from shadowspill.artifacts._values import freeze
from shadowspill.execution.inputs import stage_inputs
from shadowspill.pytorch.guards import capture_input_signature


def test_only_consumed_nonempty_inputs_are_staged_but_all_inputs_are_guarded():
    args = (torch.arange(4), torch.ones(7), torch.empty(0))
    signature = capture_input_signature((args, {}))
    target = torch.empty(args[0].nbytes, dtype=torch.uint8)
    calls = []

    def copy(alias, value):
        calls.append(alias)
        assert alias == "a0"
        target.copy_(value)

    state = SimpleNamespace(
        aliases={f"o{i}": f"a{i}" for i in range(3)},
        geometry={f"o{i}": {"device_type": "cuda"} for i in range(3)},
        sizes={f"a{i}": value.nbytes for i, value in enumerate(args)},
        initial={"a0", "a2"},
        spill=SimpleNamespace(copy_from=copy),
        plan=SimpleNamespace(
            program=SimpleNamespace(
                bindings={
                    "kind": "forward",
                    "root_inputs": [
                        {"leaf_index": i, "object_id": f"o{i}"} for i in range(3)
                    ],
                    "root_sources": [
                        {"kind": "USER_INPUT", "user_leaf": i} for i in range(3)
                    ],
                    "input_signature": freeze(
                        {
                            "tree": treespec_dumps(signature.tree_spec),
                            "leaves": [leaf.identity() for leaf in signature.leaves],
                        }
                    ),
                }
            )
        ),
    )
    stage_inputs(state, [args], None, None)
    assert calls == ["a0"]
    assert torch.equal(target.view(torch.int64), args[0])
    with pytest.raises(ValueError, match="input signature"):
        stage_inputs(state, [(args[0], torch.ones(8), args[2])], None, None)
    assert calls == ["a0"]


def test_streamed_example_references_end_before_the_next_loader_call():
    sample = torch.arange(4)
    signature = capture_input_signature(((sample,), {}))
    spec = freeze(
        {
            "tree": treespec_dumps(signature.tree_spec),
            "leaves": [leaf.identity() for leaf in signature.leaves],
        }
    )
    copies, refs = [], []
    state = SimpleNamespace(
        aliases={"o": "a"},
        geometry={"o": {"device_type": "cuda"}},
        sizes={"a": sample.nbytes},
        initial={"a"},
        spill=SimpleNamespace(
            copy_from=lambda alias, value: copies.append(value.clone())
        ),
        plan=SimpleNamespace(
            program=SimpleNamespace(
                bindings={
                    "kind": "step",
                    "num_examples": 2,
                    "root_inputs": [[{"leaf_index": 0, "object_id": "o"}]] * 2,
                    "root_sources": [{"kind": "USER_INPUT", "user_leaf": 0}],
                    "input_signature": spec,
                }
            )
        ),
    )

    def load(index):
        assert all(ref() is None for ref in refs)
        value = torch.arange(4) + index
        refs.append(weakref.ref(value))
        return (value,), {}

    stage_inputs(state, None, None, load)
    assert all(ref() is None for ref in refs)
    assert len(copies) == 2
    torch.testing.assert_close(copies[1].view(torch.int64), torch.arange(4) + 1)
