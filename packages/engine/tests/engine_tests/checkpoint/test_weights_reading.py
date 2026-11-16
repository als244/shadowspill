"""Weights-only sources preserve native ownership and bounded verified reads."""

import json
import os
import tarfile
from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch

from engine_tests.checkpoint.archive_helpers import write_contents
from engine_tests.checkpoint.test_conversion import WEIGHT, HostRank, family
from engine_tests.checkpoint.test_mapping import view
from engine_tests.checkpoint.test_transfer import outputs
from shadowspill.checkpoint import save_checkpoint, write_weights
from shadowspill.checkpoint.transfer import SourceView, transfer_tensor
from shadowspill.checkpoint.weights_reading import read_weights


def save(state, path, *, weights="master", frozen_state_id=None):
    worker = HostRank(state, initialize=True)
    with path.open("wb") as stream:
        write_weights(
            worker,
            stream,
            chunk_bytes=17,
            weights=weights,
            frozen_state_id=frozen_state_id,
            metadata={"cursor": 7},
        )
    return worker


@pytest.mark.parametrize("count,dim", [(1, 0), (2, 0), (4, 1)])
@pytest.mark.parametrize(
    "weights,master", [("master", True), ("compute", True), ("master", False)]
)
@pytest.mark.parametrize("frozen", [False, True])
def test_native_weights_reads_preserve_shards_ties_copies_and_omitted_base(
    tmp_path, monkeypatch, count, dim, weights, master, frozen
):
    identity = "exact-base" if frozen else None
    for state in family(count, dim, master=master, frozen=frozen):
        path = tmp_path / f"rank-{state.logical_rank}.ssp"
        worker = save(state, path, weights=weights, frozen_state_id=identity)
        allocated = []
        empty = torch.empty

        def allocate(*args, empty=empty, allocated=allocated, **kwargs):
            value = empty(*args, **kwargs)
            allocated.append(value.nbytes)
            return value

        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", allocate)
            with read_weights(
                state, path, chunk_bytes=17, frozen_state_id=identity
            ) as source:
                assert allocated == [17]
                assert source.manifest["metadata"]["cursor"] == 7
                assert not any(
                    n.startswith(("optimizer.", "gradient.")) for n in source.chunks
                )
                for name, chunk in source.chunks.items():
                    values = worker.values[name]
                    width = 17 // chunk.dtype.itemsize
                    for start in reversed(range(0, values.numel(), width)):
                        target = torch.empty(
                            min(width, values.numel() - start), dtype=chunk.dtype
                        )
                        source.readinto(
                            replace(chunk, offset=chunk.region_offset + start), target
                        )
                        assert torch.equal(
                            target, values[start : start + target.numel()]
                        )
            assert source.closed and max(allocated) <= 17


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "rank",
        "views",
        "nested_type",
        "derived",
        "state",
        "payload_geometry",
        "extra",
    ],
)
def test_changed_native_weight_metadata_fails_before_yield(tmp_path, fault):
    state = family(1, 0)[0]
    path = tmp_path / "weights.ssp"
    save(state, path)
    with tarfile.open(path) as archive:
        contents = {
            row.name: archive.extractfile(row).read() for row in archive.getmembers()
        }
    manifest = json.loads(contents["manifest.json"])
    if fault == "format":
        manifest["format"] = "shadowspill.checkpoint"
    elif fault == "rank":
        manifest["logical_rank"] += 1
    elif fault == "views":
        manifest["model_tensors"][0]["view"]["origin"][0] += 1
    elif fault == "nested_type":
        manifest["model_tensors"][0]["view"]["origin"][0] = False
    elif fault == "derived":
        manifest["derived_tensors"] = {}
    elif fault == "state":
        manifest["state"] = "0" * 64
    elif fault == "payload_geometry":
        manifest["tensors"][0]["shape"] = [1]
    else:
        manifest["unrecognized"] = True
    contents["manifest.json"] = json.dumps(manifest).encode()
    write_contents(path, contents)
    with pytest.raises(ValueError), read_weights(state, path, chunk_bytes=17):
        pytest.fail("changed weight geometry became readable")


def test_corrupt_unrequested_weights_fail_before_yield(tmp_path):
    state = family(1, 0, frozen=True)[0]
    path = tmp_path / "weights.ssp"
    save(state, path)
    with tarfile.open(path) as archive:
        offset = archive.getmember("tensors/1").offset_data
    with path.open("r+b") as stream:
        stream.seek(offset)
        value = stream.read(1)[0]
        stream.seek(offset)
        stream.write(bytes([value ^ 255]))
    with (
        pytest.raises(ValueError, match="checksum differs"),
        read_weights(state, path, chunk_bytes=17),
    ):
        pytest.fail("corrupt weights became readable")


def test_checkpoint_cannot_be_mistaken_for_fresh_weight_initialization(tmp_path):
    state = family(1, 0)[0]
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(HostRank(state, initialize=True), path, chunk_bytes=17)
    with (
        pytest.raises(ValueError, match="weights archive differs"),
        read_weights(state, path, chunk_bytes=17),
    ):
        pytest.fail("optimizer checkpoint accepted as weights")


@pytest.mark.parametrize("identity", [None, "different-base"])
def test_omitted_weights_require_the_exact_base(tmp_path, identity):
    state = family(1, 0, frozen=True)[0]
    path = tmp_path / "weights.ssp"
    save(state, path, frozen_state_id="exact-base")
    with (
        pytest.raises(ValueError, match="frozen base"),
        read_weights(state, path, chunk_bytes=17, frozen_state_id=identity),
    ):
        pytest.fail("missing frozen base accepted")


def test_borrowed_stream_position_and_mutation_checks_are_preserved(tmp_path):
    state = family(1, 0)[0]
    path = tmp_path / "weights.ssp"
    save(state, path)
    with path.open("rb") as stream:
        stream.seek(37)
        with (
            pytest.raises(ValueError, match="changed during conversion"),
            read_weights(state, stream, chunk_bytes=17) as source,
            path.open("r+b") as changed,
        ):
            changed.seek(512)
            changed.write(b"\xff")
            changed.flush()
            stamp = path.stat()
            # Some filesystems reuse one clock tick for these tiny operations.
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
        assert source.closed and stream.tell() == 37 and not stream.closed


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("weights", ["master", "compute"])
def test_existing_transfer_maps_weight_rows_to_three_column_owners(
    tmp_path, count, weights
):
    targets = [
        view((4, 12), local=(4, 4), origin=(0, column * 4)) for column in range(3)
    ]
    values, destinations = outputs(targets, torch.float32)
    with ExitStack() as stack:
        sources = []
        for state in family(count, 0):
            path = tmp_path / f"rank-{state.logical_rank}.ssp"
            save(state, path, weights=weights)
            reader = stack.enter_context(read_weights(state, path, chunk_bytes=17))
            logical = next(
                row
                for row in state.logical_tensors()
                if row.tensor.identity.role
                == ("master" if weights == "master" else "parameter")
                and row.tensor.identity.name == "linear.weight"
            )
            sources.append(
                SourceView(reader, reader.chunks[logical.archive.name], logical.view)
            )
        transfer_tensor(
            sources, destinations, scratch_bytes=60, transformation="scalar_cast"
        )
    expected = WEIGHT if weights == "master" else WEIGHT.bfloat16().float()
    for column, actual in enumerate(values):
        torch.testing.assert_close(
            actual.reshape(4, 4),
            expected[:, column * 4 : (column + 1) * 4],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("budget", [0, -1, True, 3.0, 3])
def test_invalid_weight_read_budget_is_rejected(tmp_path, budget):
    state = family(1, 0)[0]
    path = tmp_path / "weights.ssp"
    save(state, path)
    with (
        pytest.raises(ValueError, match=r"positive integer|one element"),
        read_weights(state, path, chunk_bytes=budget),
    ):
        pytest.fail("invalid read budget accepted")


def test_weight_metadata_limit_precedes_decoding(tmp_path, monkeypatch):
    from shadowspill.checkpoint import weights_reading

    state = family(1, 0)[0]
    path = tmp_path / "weights.ssp"
    save(state, path)
    monkeypatch.setattr(
        weights_reading,
        "read_manifest",
        lambda *a: pytest.fail("decoded oversized metadata"),
    )
    with (
        pytest.raises(ValueError, match="configured byte limit"),
        read_weights(state, path, chunk_bytes=17, max_manifest_bytes=1),
    ):
        pytest.fail("oversized metadata admitted")
