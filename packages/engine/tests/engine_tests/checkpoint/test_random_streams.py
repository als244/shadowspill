"""Logical stream preservation and stable derivation use real CPU generators."""

import json
import os
import random
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from engine_tests.execution.test_checkpoint import MemoryExecutor
from shadowspill.artifacts.state import StateId
from shadowspill.checkpoint import load_checkpoint, metadata_values, save_checkpoint
from shadowspill.checkpoint.random_streams import (
    MAX_STREAM_BYTES,
    RandomState,
    RandomStream,
    map_random_streams,
)


def stream(name="dropout", rank=0, representation="torch.cpu@1"):
    return RandomStream(StateId("rng", name, ("training",)), representation, rank)


def payload(value):
    return json.dumps(metadata_values.encode(value), separators=(",", ":"))


def decoded(state):
    return metadata_values.decode(json.loads(state.payload))


@pytest.mark.parametrize("backend", ["python", "numpy", "torch"])
def test_preserved_stream_changes_owner_and_replays_the_next_draw(backend):
    if backend == "python":
        generator = random.Random(91)

        def draw():
            return [generator.random() for _ in range(7)]

        get, put = generator.getstate, generator.setstate
    elif backend == "numpy":
        generator = np.random.RandomState(91)

        def draw():
            return generator.random_sample(7)

        get, put = generator.get_state, generator.set_state
    else:
        generator = torch.Generator().manual_seed(91)

        def draw():
            return torch.rand(7, generator=generator).numpy()

        get, put = generator.get_state, generator.set_state
    draw()
    original = RandomState(stream(representation=backend + "@1"), payload(get()))
    expected = draw()
    mapped = map_random_streams(
        [original],
        [replace(original.stream, rank=7)],
        root_seed=91,
        restore_id="resume-1",
        initializers=(),
    )
    assert mapped.states[0].payload == original.payload
    assert mapped.states[0].stream.rank == 7 and mapped.derived == ()
    assert mapped.provenance()["target_streams"][0]["action"] == "preserve"
    put(decoded(mapped.states[0]))
    np.testing.assert_array_equal(draw(), expected)


def test_native_archive_preserves_rng_and_the_next_stochastic_optimizer_update(
    tmp_path,
):
    generator = torch.Generator().manual_seed(512)
    model = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    optimizer = torch.optim.SGD([model], lr=0.1)

    def update():
        optimizer.zero_grad()
        loss = (model * torch.randn(3, generator=generator)).square().sum()
        loss.backward()
        optimizer.step()
        return model.detach().clone()

    update()
    saved = RandomState(stream(), payload(generator.get_state()))
    path = tmp_path / "state.ssp"
    executor = MemoryExecutor()
    executor.values = {"parameter": model.detach()}
    save_checkpoint(
        executor, path, metadata={"random": saved.to_dict()}, chunk_bytes=17
    )
    expected = update()
    with torch.no_grad():
        model.zero_()
    generator.manual_seed(123)
    metadata = load_checkpoint(executor, path, chunk_bytes=19)
    saved = RandomState(saved.stream, metadata["random"]["payload"])
    mapped = map_random_streams(
        [saved],
        [replace(saved.stream, rank=3)],
        root_seed=512,
        restore_id="native-resume",
        initializers=(),
    )
    generator.set_state(decoded(mapped.states[0]))
    torch.testing.assert_close(update(), expected, rtol=0, atol=0)


def test_changed_layout_preserves_known_stream_and_derives_only_new_streams():
    old = RandomState(stream("existing"), payload(torch.Generator().get_state()))
    new = stream("new", rank=3)
    options = dict(
        root_seed=17, restore_id="convert-checkpoint", initializers={new.representation}
    )
    mapped = map_random_streams([old], [new, replace(old.stream, rank=2)], **options)
    assert mapped == map_random_streams(
        [old], list(reversed([new, replace(old.stream, rank=2)])), **options
    )
    by_name = {state.stream.identity.name: state for state in mapped.states}
    assert by_name["existing"].payload == old.payload
    assert by_name["new"].seed is not None and mapped.derived == (new.identity,)
    expected = torch.rand(
        11, generator=torch.Generator().manual_seed(by_name["new"].seed)
    )
    again = map_random_streams([old], [replace(new, rank=99)], **options).states[0]
    torch.testing.assert_close(
        torch.rand(11, generator=torch.Generator().manual_seed(again.seed)),
        expected,
        rtol=0,
        atol=0,
    )
    assert (
        map_random_streams([old], [new], **{**options, "restore_id": "distinct"})
        .states[0]
        .seed
        != again.seed
    )
    assert (
        map_random_streams([old], [new], **{**options, "root_seed": 18}).states[0].seed
        != again.seed
    )


def test_versioned_derivation_is_independent_of_hash_seed_and_framework_imports():
    script = """
import json, sys
from shadowspill.artifacts.state import StateId
from shadowspill.checkpoint.random_streams import RandomStream, map_random_streams
target = RandomStream(StateId('rng', 'dropout', ('training',)), 'torch.cpu@1', 9)
mapped = map_random_streams(
    [], [target], root_seed=17, restore_id='restore-1',
    initializers={'torch.cpu@1'},
)
blocked = {'torch', 'triton', 'pynvml', 'shadowspill_workloads'}
assert not blocked.intersection(sys.modules)
print(json.dumps(mapped.provenance(), sort_keys=True))
"""
    outputs = []
    for seed in ("1", "984"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
    assert outputs[0]["version"] == 1
    assert outputs[0]["source_streams"] == []
    assert outputs[0]["target_streams"][0]["action"] == "derive"
    assert outputs[0]["target_streams"][0]["seed"] == 6957210690759386126


def test_unknown_changed_representation_requires_explicit_initializer():
    old = RandomState(stream(representation="external.generator@1"), "opaque")
    target = replace(old.stream, representation="external.generator@2")
    with pytest.raises(ValueError, match="declared initializer"):
        map_random_streams(
            [old], [target], root_seed=4, restore_id="r", initializers=()
        )
    mapped = map_random_streams(
        [old],
        [target],
        root_seed=4,
        restore_id="r",
        initializers={target.representation},
    )
    assert mapped.states[0].payload is None and mapped.states[0].seed is not None
    assert (
        mapped.provenance()["source_streams"][0]["representation"]
        == old.stream.representation
    )


@pytest.mark.parametrize("role", ["parameter", "optimizer", "buffer", "registered"])
def test_rng_permission_never_initializes_other_state_roles(role):
    with pytest.raises(ValueError, match="RNG state identity"):
        RandomStream(StateId(role, "state"), "external@1", 0)


def test_conflicting_stream_ids_and_unbounded_or_ambiguous_states_fail():
    saved = RandomState(stream(), "opaque")
    for sources, targets in (
        ([saved, saved], [saved.stream]),
        ([saved], [saved.stream, saved.stream]),
    ):
        with pytest.raises(ValueError, match="duplicate"):
            map_random_streams(
                sources, targets, root_seed=0, restore_id="r", initializers=()
            )
    for values in ({}, {"payload": "x", "seed": 3}):
        with pytest.raises(ValueError, match="exactly one"):
            RandomState(stream(), **values)
    for body in ("x" * (MAX_STREAM_BYTES + 1), "ø" * MAX_STREAM_BYTES):
        with pytest.raises(ValueError, match="byte bound"):
            RandomState(stream(), body)
    for seed in (-1, True, 1 << 64):
        with pytest.raises(ValueError):
            RandomState(stream(), seed=seed)
