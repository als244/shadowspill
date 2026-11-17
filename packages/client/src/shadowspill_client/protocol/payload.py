"""Owned Datum snapshots and deterministic, chunked binary transfer framing."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Self

from ..buffers import BufferBudget, BufferLease
from ..datum import Datum
from .payload_sizes import (
    MAX_INPUT_METADATA_BYTES,
    MAX_TREE_DEPTH,
    TREE_NODE_BYTES,
    PayloadSize,
    manifest_json,
)
from .tensors import TensorPayload, TensorSource

MAGIC = b"SSDT\x01\0\0\0"
MEDIA_TYPE = "application/vnd.shadowspill.datum+binary;version=1"


def owned_datums(
    trees: Sequence[Any], tensors: Sequence[TensorPayload]
) -> tuple[Datum, ...]:
    """Construct immutable trees from an already validated tensor-index manifest."""

    def own(tree: Any) -> Any:
        if isinstance(tree, dict):
            return MappingProxyType({key: own(child) for key, child in tree.items()})
        return tensors[tree]

    return tuple(
        Datum(inputs=own(tree["inputs"]), loss_inputs=own(tree["loss_inputs"]))
        for tree in trees
    )


class BatchSnapshot:
    """Owned immutable payload; a submission holds it until upload/local handoff.

    The lease covers tensor bytes, bounded metadata accounting and temporary
    normalization space. Close after the last transfer consumer, not merely
    after returning a future. This is memory ownership, not durable acceptance.
    """

    def __init__(
        self,
        header: bytes,
        tensors: Sequence[TensorPayload],
        datums: Sequence[Datum],
        sha256: str,
        lease: BufferLease,
    ) -> None:
        self.header = header
        self.tensors = tuple(tensors)
        self.datums = tuple(datums)
        self.sha256 = sha256
        self.size_bytes = (
            len(MAGIC) + 8 + len(header) + sum(len(t.data) for t in tensors)
        )
        self._lease = lease
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def chunks(self, chunk_bytes: int = 1 << 20) -> Iterator[memoryview]:
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ValueError("chunk_bytes must be a positive integer")
        if self._closed:
            raise RuntimeError("input snapshot is closed")
        parts = (MAGIC + struct.pack("<Q", len(self.header)), self.header)
        for data in (*parts, *(tensor.data for tensor in self.tensors)):
            for start in range(0, len(data), chunk_bytes):
                if self._closed:
                    raise RuntimeError("input snapshot closed during transfer")
                yield memoryview(data)[start : start + chunk_bytes]

    def close(self) -> None:
        if self._closed:
            return
        self.header, self.tensors, self.datums = b"", (), ()
        self._closed = True
        self._lease.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def snapshot_batch(
    batch: Sequence[Datum],
    budget: BufferBudget,
    *,
    timeout: float | None = None,
    workspace_bytes: int = 0,
) -> BatchSnapshot:
    """Copy exactly one logical call before its future may be returned.

    Metadata inspection and reservation precede array normalization. Array/list
    payloads never enter JSON. Callers may reuse all original inputs after this
    function returns; concurrent mutation during the copy is not supported.
    """
    if not isinstance(batch, (list, tuple)) or any(
        not isinstance(d, Datum) for d in batch
    ):
        raise TypeError("global_batch must be a list or tuple of Datum records")
    sources: list[TensorSource] = []
    metadata_size = 0

    def inspect(value: Any, depth: int = 0) -> Any:
        nonlocal metadata_size
        metadata_size += TREE_NODE_BYTES
        # Leave room for the outer manifest/Datum containers under the shared
        # protocol's 64-level nesting limit.
        if depth > MAX_TREE_DEPTH or metadata_size > MAX_INPUT_METADATA_BYTES:
            raise ValueError("input tree metadata exceeds its bounded size/depth")
        if isinstance(value, Mapping):
            result = {}
            if any(type(key) is not str or not key for key in value):
                raise ValueError("input tree keys must be nonempty strings")
            for key, child in sorted(value.items()):
                metadata_size += len(key.encode("utf-8"))
                result[key] = inspect(child, depth + 1)
            return result
        source = TensorSource.inspect(value)
        sources.append(source)
        return len(sources) - 1

    trees = [
        {"inputs": inspect(d.inputs), "loss_inputs": inspect(d.loss_inputs)}
        for d in batch
    ]
    header = manifest_json(
        trees, [{"dtype": s.dtype, "shape": s.shape} for s in sources]
    )
    if len(header) > MAX_INPUT_METADATA_BYTES:
        raise ValueError("input tensor metadata exceeds its byte limit")
    # Arrays are immutable bytes, so NumPy decoding is a zero-copy view. Reserve
    # one extra leaf for list/endian/optional Torch normalization while copying.
    # Metadata has a bounded allowance for decoded containers as well as JSON.
    size = PayloadSize(
        sum(s.nbytes for s in sources),
        len(header),
        metadata_size,
        max((s.nbytes for s in sources), default=0),
    )
    if type(workspace_bytes) is not int or workspace_bytes < 0:
        raise ValueError("snapshot workspace must be nonnegative bytes")
    lease = budget.reserve(size.staging + workspace_bytes, timeout=timeout)
    try:
        digest = hashlib.sha256(MAGIC + struct.pack("<Q", len(header)) + header)
        tensors = []
        for source in sources:
            tensor = source.copy()
            digest.update(tensor.data)
            tensors.append(tensor)

        result = BatchSnapshot(
            header, tensors, owned_datums(trees, tensors), digest.hexdigest(), lease
        )
        lease.shrink(size.retained + workspace_bytes)
        return result
    except BaseException:
        lease.close()
        raise
