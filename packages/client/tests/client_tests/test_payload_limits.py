"""Prepared upload bounds reserve decoding before header or tensor allocation."""

from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest

from shadowspill_client import Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch
from shadowspill_client.protocol.payload_io import read_payload
from shadowspill_client.protocol.payload_sizes import PayloadSize
from shadowspill_client.protocol.requests import ArtifactRef


def payload():
    with snapshot_batch(
        [Datum({"x": np.array([1, 2], dtype=np.float32)})], BufferBudget(1 << 20)
    ) as snapshot:
        return (
            b"".join(snapshot.chunks()),
            ArtifactRef("input", snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE),
            PayloadSize(8, len(snapshot.header), 1024, 8),
        )


def test_large_sequence_batch_manifest_roundtrips_with_budget_accounting():
    data = np.array([1, 2], dtype=np.int64)
    batch = [
        Datum(
            {"tokens": data},
            {
                "target_tokens": data,
                "loss_mask": [True, True],
                "weights": [0.5, 0.5],
                "router_mask": [True, True],
            },
        )
        for _ in range(8192)
    ]
    budget = BufferBudget(128 << 20)
    with snapshot_batch(batch, budget) as snapshot:
        wire = b"".join(snapshot.chunks())
        reference = ArtifactRef("input", snapshot.sha256, len(wire), MEDIA_TYPE)
        assert len(snapshot.header) > 1 << 20
        with read_payload(
            BytesIO(wire), reference, budget, datum_count=8192
        ) as decoded:
            assert len(decoded.datums) == 8192
            np.testing.assert_array_equal(
                decoded.datums[-1].inputs["tokens"].array(), data
            )
    assert budget.used == (0, 0)


def test_decode_peak_is_owned_before_first_read_and_shrinks_after_validation():
    data, reference, limits = payload()
    budget = BufferBudget(limits.staging)
    reads = []

    class Reader(BytesIO):
        def readinto(self, buffer):
            reads.append(len(buffer))
            assert budget.used == (limits.staging, 1)
            return super().readinto(buffer)

    with read_payload(
        Reader(data), reference, budget, limits=limits, datum_count=1
    ) as snapshot:
        assert snapshot.datums[0].inputs["x"].array().tolist() == [1, 2]
        assert budget.used[0] < limits.staging
    assert reads and budget.used == (0, 0)


def test_insufficient_decode_reservation_or_oversized_frame_reads_no_bytes():
    data, reference, limits = payload()
    stream = BytesIO(data)
    with pytest.raises(ValueError, match="limit"):
        read_payload(stream, reference, BufferBudget(limits.staging - 1), limits=limits)
    assert stream.tell() == 0
    with pytest.raises(ValueError, match="prepared decode envelope"):
        read_payload(
            stream,
            replace(reference, size_bytes=limits.wire + 1),
            BufferBudget(limits.staging),
            limits=limits,
        )
    assert stream.tell() == 0


@pytest.mark.parametrize(
    "part", ["header", "tree", "tensors", "largest_tensor", "count"]
)
def test_bad_manifest_is_rejected_before_reading_any_tensor_bytes(part):
    data, reference, limits = payload()
    if part == "header":
        limits = replace(limits, header=limits.header - 1, tensors=108)
    elif part == "tensors":
        limits = replace(limits, tensors=7, header=limits.header + 1)
    elif part in ("tree", "largest_tensor"):
        limits = replace(limits, **{part: 0})
    stream, budget = BytesIO(data), BufferBudget(limits.staging)
    with pytest.raises(ValueError):
        read_payload(
            stream,
            reference,
            budget,
            limits=limits,
            datum_count=2 if part == "count" else 1,
        )
    assert stream.tell() <= len(data) - 8
    assert budget.used == (0, 0)


def test_checksum_failure_releases_preowned_decode_reservation():
    data, reference, limits = payload()
    budget = BufferBudget(limits.staging)
    with pytest.raises(ValueError, match="checksum"):
        read_payload(
            BytesIO(data), replace(reference, sha256="0" * 64), budget, limits=limits
        )
    assert budget.used == (0, 0)
