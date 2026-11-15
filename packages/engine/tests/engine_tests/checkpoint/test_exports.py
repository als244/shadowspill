"""Inference archives preserve declared weight copies and owned model views."""

import io
import tarfile
from dataclasses import replace

import pytest
import torch
from blake3 import blake3

from engine_tests.checkpoint.test_conversion import HostRank, family
from engine_tests.checkpoint.test_logical import model_spec, shaped
from engine_tests.checkpoint.test_state import plan
from shadowspill.checkpoint import load_checkpoint, weights_size, write_weights
from shadowspill.checkpoint.archive import read_manifest
from shadowspill.checkpoint.state import PlanState
from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec


def decode(body):
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:") as archive:
        manifest = read_manifest(archive)
        values = {}
        for row in manifest["tensors"]:
            payload = archive.extractfile(row["member"]).read()
            assert len(payload) == row["bytes"]
            assert blake3(payload).hexdigest() == row["blake3"]
            values[row["name"]] = torch.frombuffer(
                bytearray(payload),
                dtype=getattr(torch, row["dtype"].removeprefix("torch.")),
            )
        return manifest, values


@pytest.mark.parametrize("count,dim", [(1, 0), (2, 0), (4, 1)])
@pytest.mark.parametrize(
    "weights,master", [("master", True), ("compute", True), ("master", False)]
)
@pytest.mark.parametrize("frozen", [False, True])
def test_exact_size_authoritative_weights_owned_views_and_frozen_base(
    count, dim, weights, master, frozen
):
    identity = "exact-base" if frozen else None
    for state in family(count, dim, master=master, frozen=frozen):
        worker = HostRank(state, initialize=True)
        seen = []
        original = worker.export_tensors

        def export(callback, *, chunk_bytes, names, original=original, seen=seen):
            seen.append(names)
            assert not any(
                name.startswith(("optimizer", "gradient", "fixed")) for name in names
            )
            original(callback, chunk_bytes=chunk_bytes, names=names)

        worker.export_tensors = export
        options = dict(
            metadata={"updates": 17}, weights=weights, frozen_state_id=identity
        )
        stream = io.BytesIO()
        returned = write_weights(worker, stream, chunk_bytes=17, **options)
        assert len(stream.getvalue()) == weights_size(state, **options)
        manifest, values = decode(stream.getvalue())
        assert manifest == returned
        assert manifest["format"] == "shadowspill.weights" and manifest["version"] == 1
        assert {
            row["tensor"]["identity"]["role"] for row in manifest["model_tensors"]
        } <= {"parameter", "buffer", "master"}
        expected = (
            "master.linear.weight"
            if weights == "master" and master
            else "linear.weight"
        )
        assert set(values) == {expected} and seen == [frozenset(values)]
        torch.testing.assert_close(
            values[expected], worker.values[expected], rtol=0, atol=0
        )
        assert manifest["omitted_parameters"] == (
            ["base.tied", "base.weight"] if frozen else []
        )
        model = {
            row["tensor"]["identity"]["name"]: row
            for row in manifest["model_tensors"]
            if row["tensor"]["identity"]["role"] == "parameter"
        }
        assert model["linear.weight"]["archive"] == model["head.weight"]["archive"]
        assert model["linear.weight"]["view"]["shape"] == [4, 12]
        assert (
            model["linear.weight"]["view"]["origin"][dim]
            == (count - state.logical_rank - 1) * (4 if dim == 0 else 12) // count
        )


def test_full_inference_export_keeps_frozen_model_state_without_claiming_a_base():
    worker = HostRank(family(1, 0, frozen=True)[0], initialize=True)
    stream = io.BytesIO()
    write_weights(worker, stream, chunk_bytes=16)
    manifest, values = decode(stream.getvalue())
    assert set(values) == {"master.linear.weight", "base.weight"}
    assert not manifest["omitted_parameters"] and manifest["frozen_state_id"] is None


@pytest.mark.parametrize("kind", ["buffer", "mutated", "additional_plan_mutation"])
def test_frozen_identity_keeps_buffers_and_weights_modified_by_any_plan(kind):
    source = family(1, 0, frozen=True)[0].plan.program
    additional = {}
    if kind == "buffer":
        source = replace(
            source,
            bindings={
                **source.bindings,
                "registrations": [
                    {**row, "parameter": False, "persistent": True}
                    if row["name"].startswith("base.")
                    else row
                    for row in source.bindings["registrations"]
                ],
            },
        )
    else:
        mutation = TaskSpec(
            "mutate_base",
            ResourceSpec("rank0", ResourceKind.COMPUTE),
            "boundary",
            inputs=("base.weight",),
            mutations=(MutationSpec("base.weight"),),
            requires_entrypoint=False,
        )
        changed = replace(source, tasks=(*source.tasks, mutation))
        if kind == "mutated":
            source = changed
        else:
            additional["evaluation"] = plan(changed, kind="forward")
    state = PlanState(plan(source), additional, logical_rank=0)
    worker = HostRank(state, initialize=True)
    worker.values["base.weight"].add_(5)
    sink = io.BytesIO()
    write_weights(worker, sink, frozen_state_id="exact-base", chunk_bytes=16)
    manifest, values = decode(sink.getvalue())
    assert not manifest["omitted_parameters"]
    assert set(values) == {"master.linear.weight", "base.weight"}
    torch.testing.assert_close(values["base.weight"], worker.values["base.weight"])


def test_model_export_does_not_require_unrelated_optimizer_logical_placement():
    source = family(2, 0)[0].plan.program
    source = replace(
        source,
        bindings={
            **source.bindings,
            "optimizer_bindings": [
                *source.bindings["optimizer_bindings"],
                dict(
                    name="optimizer.statistics",
                    role="state",
                    object_id=None,
                    device="cpu",
                    parameter_name="linear.weight",
                    shape=(2,),
                    stride=(1,),
                    dtype="torch.float32",
                    initialization={"kind": "constant", "value": 0},
                ),
            ],
        },
    )
    state = PlanState(plan(source), {}, logical_rank=0)
    with pytest.raises(ValueError, match="optimizer state requires its own"):
        tuple(state.logical_tensors())
    worker = HostRank(state)
    sink = io.BytesIO()
    write_weights(worker, sink, chunk_bytes=16)
    manifest, values = decode(sink.getvalue())
    assert len(sink.getvalue()) == weights_size(state)
    assert set(values) == {"master.linear.weight"}
    assert all(
        row["tensor"]["identity"]["role"] != "optimizer"
        for row in manifest["model_tensors"]
    )


def test_weights_archive_cannot_be_mistaken_for_optimizer_resume(tmp_path):
    worker = HostRank(family(1, 0)[0], initialize=True)
    path = tmp_path / "weights.ssp"
    with path.open("wb") as stream:
        write_weights(worker, stream, chunk_bytes=16)
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(worker, path)


@pytest.mark.parametrize(
    "fault", ["missing", "unselected", "geometry", "dtype", "failure"]
)
def test_failed_or_inconsistent_export_never_returns_a_complete_archive(fault):
    worker = HostRank(family(1, 0)[0], initialize=True)
    original = worker.export_tensors

    def export(callback, *, chunk_bytes, names):
        if fault == "missing":
            return

        def emit(chunk, value):
            if fault == "failure":
                raise OSError("export interrupted")
            if fault == "unselected":
                chunk = replace(chunk, name="optimizer.fake")
            elif fault == "geometry":
                chunk = replace(chunk, shape=(1,))
            elif fault == "dtype":
                value = value.to(torch.float64)
            callback(chunk, value)

        original(emit, chunk_bytes=chunk_bytes, names=names)

    worker.export_tensors = export
    with pytest.raises((ValueError, OSError)):
        write_weights(worker, io.BytesIO(), chunk_bytes=16)


def test_short_sink_writes_and_metadata_mutation_preserve_original_manifest():
    worker = HostRank(family(1, 0)[0], initialize=True)
    metadata = {"cursor": [1, 2]}

    class Sink(io.BytesIO):
        def write(self, data):
            metadata["cursor"][:] = [99]
            return super().write(data[:7])

    sink = Sink()
    write_weights(worker, sink, metadata=metadata, chunk_bytes=16)
    manifest, _ = decode(sink.getvalue())
    assert manifest["metadata"] == {"cursor": [1, 2]}


def test_large_model_size_is_described_without_tensor_allocation(monkeypatch):
    size = 9 << 30
    source = shaped((1, size // 4), model_spec((0,), 0))
    state = PlanState(plan(source), {}, logical_rank=0)
    monkeypatch.setattr(
        torch, "empty", lambda *_args, **_kwargs: pytest.fail("allocated payload")
    )
    required = weights_size(state)
    # This fixture has no scalar-cast contract, so both 9 GiB weight copies
    # remain authoritative under the existing explicit retention exception.
    assert 2 * size < required < 2 * size + (64 << 10)


def test_metadata_bound_fails_before_export_or_writing():
    worker = HostRank(family(1, 0)[0])
    worker.export_tensors = lambda *_args, **_kwargs: pytest.fail(
        "exported before metadata admission"
    )
    sink = io.BytesIO()
    with pytest.raises(ValueError, match="16 MiB"):
        write_weights(worker, sink, metadata={"oversized": "x" * (16 << 20)})
    assert not sink.getvalue()
