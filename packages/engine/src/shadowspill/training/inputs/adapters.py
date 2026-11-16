"""Definition-owned fixed input views and bounded generated CPU metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from shadowspill.parallelism import InputPlacement
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.inputs.shapes import BufferShape
from shadowspill_client.inputs.validation import Path
from shadowspill_client.inputs.values import (
    VALIDATION_SCRATCH_BYTES,
    ValidatedTensor,
    validate_values,
)
from shadowspill_client.protocol.inputs import InputSchema, TensorField
from shadowspill_client.protocol.payload import BatchSnapshot
from shadowspill_client.protocol.tensors import (
    TensorPayload,
    storage_dtype,
    tensor_bytes,
)

from .frames import InputFrame
from .routing import RoutedBuffer


@dataclass(frozen=True)
class InputView:
    """Rename/reshape one routed field without copying or changing its values."""

    path: Path
    source: Path
    shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for key in ("path", "source"):
            value = tuple(getattr(self, key))
            if (
                len(value) < 2
                or value[0] not in ("inputs", "loss_inputs")
                or any(not isinstance(part, str) or not part for part in value)
            ):
                raise ValueError("adapter paths must name model or loss tensor leaves")
            object.__setattr__(self, key, value)
        if self.shape is not None:
            shape = tuple(self.shape)
            if any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(
                    "adapter view shapes must be fixed nonnegative integers"
                )
            object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class InputLayout:
    schema: InputSchema
    buffer: BufferShape
    placement: InputPlacement
    logical_rank: int

    def local_shape(self, path: Path) -> tuple[int, ...]:
        shape = self.buffer.fields[path]
        regions = self.placement.region(path, shape, self.logical_rank)
        return tuple(
            len(range(*region.indices(size)))
            for size, region in zip(shape, regions, strict=True)
        )


@dataclass(frozen=True)
class AdapterContext:
    layout: InputLayout
    routed: RoutedBuffer


type MetadataWriter = Callable[[AdapterContext, Mapping[Path, NDArray[Any]]], None]


@dataclass(frozen=True)
class InputAdapter:
    """Small functions and descriptions, without an adapter base class.

    Views borrow routed inputs. Generated tensors have fixed, declared host
    storage allocated before ``fill`` runs. The installed definition owns
    sequence/group/media semantics and declares whether padding preserves its
    model computation; objective masking is a separate preparation requirement.
    """

    views: Sequence[InputView] | None = None
    metadata: Sequence[TensorField] = ()
    fill: MetadataWriter | None = None
    padding_safe: bool = False

    def __post_init__(self) -> None:
        views = None if self.views is None else tuple(self.views)
        if views is not None and any(not isinstance(view, InputView) for view in views):
            raise TypeError("adapter views must be InputView records")
        fields = tuple(self.metadata)
        if any(
            not isinstance(field, TensorField)
            or any(type(n) is not int for n in field.shape)
            or len(field.path) < 2
            or field.path[0] not in ("inputs", "loss_inputs")
            for field in fields
        ):
            raise ValueError("adapter metadata requires fixed tensor-leaf declarations")
        if self.fill is not None and not callable(self.fill):
            raise TypeError("metadata fill must be callable")
        if type(self.padding_safe) is not bool:
            raise TypeError("padding_safe must be boolean")
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "metadata", fields)


class AdaptedInputs:
    def __init__(
        self,
        owner: RoutedBuffer,
        fields: Mapping[Path, NDArray[Any]],
        dtypes: Mapping[Path, str],
        lease: BufferLease,
    ) -> None:
        self.owner = owner
        self.fields = MappingProxyType(dict(fields))
        self.dtypes = dtypes
        self._lease = lease
        self.closed = False

    def torch_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Borrow CPU tensor views through existing capture/input staging only."""
        import torch

        if self.closed or self.owner.owner.closed:
            raise RuntimeError("adapted input storage is closed")
        trees: dict[str, dict[str, Any]] = {"inputs": {}, "loss_inputs": {}}
        for path, array in self.fields.items():
            # Engine-owned storage arrays remain writable; exposed NumPy views
            # are read-only. Staging consumes this private Torch view without
            # mutation and copies into admitted worker input storage.
            writable = array.view()
            writable.setflags(write=True)
            tensor = torch.from_numpy(writable)
            if self.dtypes[path] == "bfloat16":
                tensor = tensor.view(torch.bfloat16)
            tree = trees[path[0]]
            for name in path[1:-1]:
                tree = tree.setdefault(name, {})
            tree[path[-1]] = tensor
        return trees["inputs"], trees["loss_inputs"]

    def close(self) -> None:
        self.fields = MappingProxyType({})
        self.closed = True
        self._lease.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class InputBinding:
    """Resolved adapter ABI, inspectable before model construction or GPU work."""

    def __init__(self, layout: InputLayout, adapter: InputAdapter) -> None:
        if not isinstance(layout, InputLayout) or not isinstance(adapter, InputAdapter):
            raise TypeError("bind an InputLayout and InputAdapter")
        self.layout, self.adapter = layout, adapter
        layout.placement.partition_index(layout.logical_rank)
        definitions = {tuple(field.path): field for field in layout.schema.fields}
        layout.buffer.validate(definitions)
        self.views = (
            tuple(InputView(path, path) for path in layout.buffer.fields)
            if adapter.views is None
            else tuple(adapter.views)
        )
        shapes, dtypes = {}, {}
        for view in self.views:
            if view.source not in layout.buffer.fields:
                raise ValueError(
                    f"adapter source is absent from the physical buffer: {view.source}"
                )
            source_shape = layout.local_shape(view.source)
            target = source_shape if view.shape is None else view.shape
            if prod(source_shape) != prod(target):
                raise ValueError("adapter view must preserve every source element")
            shapes[view.path], dtypes[view.path] = (
                target,
                definitions[view.source].dtype,
            )
        for field in adapter.metadata:
            shapes[tuple(field.path)] = tuple(int(n) for n in field.shape)
            dtypes[tuple(field.path)] = field.dtype
        paths = [view.path for view in self.views] + [
            tuple(field.path) for field in adapter.metadata
        ]
        if len(set(paths)) != len(paths) or any(
            p != q and q[: len(p)] == p for p in paths for q in paths
        ):
            raise ValueError("adapter output paths overlap")
        self.shapes = MappingProxyType(shapes)
        self.dtypes = MappingProxyType(dtypes)
        # Store each referenced physical storage array and generated field once.
        # Reconstructing the declared views then preserves the original strides,
        # empty-slice geometry and storage extents used by capture guards.
        self.frame_sources = {
            ("inputs", "buffers", str(i)): path
            for i, path in enumerate(sorted({view.source for view in self.views}))
        }
        self.frame_metadata = {
            ("inputs", "metadata", str(i)): tuple(field.path)
            for i, field in enumerate(adapter.metadata)
        }
        self.frame = InputFrame(
            {
                **{
                    key: tuple(layout.buffer.fields[path])
                    for key, path in self.frame_sources.items()
                },
                **{key: shapes[path] for key, path in self.frame_metadata.items()},
            },
            {
                **{
                    key: definitions[path].dtype
                    for key, path in self.frame_sources.items()
                },
                **{key: dtypes[path] for key, path in self.frame_metadata.items()},
            },
        )
        self.metadata_bytes = sum(
            tensor_bytes(f.dtype, self.shapes[tuple(f.path)]) for f in adapter.metadata
        )
        scratch = max(
            (
                tensor_bytes(f.dtype, self.shapes[tuple(f.path)])
                for f in adapter.metadata
            ),
            default=0,
        )
        self.retained_bytes = self.metadata_bytes + 512 * (
            sum(len(path) + 1 for path in paths) + 1
        )
        self.staging_bytes = (
            self.retained_bytes
            + scratch
            + (VALIDATION_SCRATCH_BYTES if adapter.metadata else 0)
        )

    def frame_fields(self, adapted: AdaptedInputs) -> Mapping[Path, NDArray[Any]]:
        if adapted.closed or adapted.owner.owner.closed:
            raise RuntimeError("adapted input storage is closed")
        return {
            **{
                key: adapted.owner.owner.fields[path]
                for key, path in self.frame_sources.items()
            },
            **{key: adapted.fields[path] for key, path in self.frame_metadata.items()},
        }

    def frame_arrays(self, snapshot: BatchSnapshot) -> Mapping[Path, NDArray[Any]]:
        """Restore prepared views from ready storage without invoking any hook."""
        if snapshot.closed or snapshot.header != self.frame.header:
            raise ValueError("ready input frame differs from the prepared binding")

        def array(key: Path) -> NDArray[Any]:
            payload = snapshot.datums[0].inputs[key[1]][key[2]]
            assert isinstance(payload, TensorPayload)
            if not prod(payload.shape):
                # np.empty (used by materialization/metadata) gives empty
                # storage arrays zero strides; frombuffer.reshape need not.
                return np.ndarray(
                    payload.shape,
                    dtype=storage_dtype(payload.dtype),
                    buffer=payload.data,
                    strides=(0,) * len(payload.shape),
                )
            return payload.array()

        physical = {path: array(key) for key, path in self.frame_sources.items()}
        result = {}
        for view in self.views:
            source = physical[view.source]
            regions = self.layout.placement.region(
                view.source, source.shape, self.layout.logical_rank
            )
            source = source[regions] if source.ndim else source.view()
            target = self.shapes[view.path]
            if source.shape == target:
                result[view.path] = source.view()
            elif source.flags.c_contiguous:
                result[view.path] = source.reshape(target)
            else:
                raise ValueError("adapter reshape requires a contiguous source view")
        result.update({path: array(key) for key, path in self.frame_metadata.items()})
        return MappingProxyType(result)

    def signature(self) -> dict[str, Any]:
        return {
            "version": 1,
            "fields": [
                {"path": list(path), "dtype": self.dtypes[path], "shape": list(shape)}
                for path, shape in sorted(self.shapes.items())
            ],
            "views": [
                {
                    "path": list(view.path),
                    "source": list(view.source),
                    "shape": None if view.shape is None else list(view.shape),
                }
                for view in self.views
            ],
            "metadata": [field.to_dict() for field in self.adapter.metadata],
            "padding_safe": self.adapter.padding_safe,
            "metadata_bytes": self.metadata_bytes,
            "staging_bytes": self.staging_bytes,
        }

    def adapt(
        self,
        routed: RoutedBuffer,
        budget: BufferBudget,
        *,
        timeout: float | None = None,
    ) -> AdaptedInputs:
        if routed.owner.closed:
            raise RuntimeError("packed buffer is closed")
        if (
            routed.logical_rank != self.layout.logical_rank
            or routed.owner.fields.keys() != self.layout.buffer.fields.keys()
            or routed.fields.keys() != self.layout.buffer.fields.keys()
            or routed.owner.location[0]
            != self.layout.placement.partition_index(routed.logical_rank)
            or any(
                tuple(routed.owner.fields[path].shape) != shape
                for path, shape in self.layout.buffer.fields.items()
            )
            or routed.contributes_loss
            != self.layout.placement.contributes_loss(routed.logical_rank)
        ):
            raise ValueError("routed buffer differs from the prepared adapter layout")
        fields = {}
        for view in self.views:
            source = routed.fields[view.source]
            if tuple(source.shape) != self.layout.local_shape(
                view.source
            ) or source.dtype != storage_dtype(self.dtypes[view.path]):
                raise ValueError("routed field shape differs from its prepared slice")
            target = self.shapes[view.path]
            if source.shape == target:
                value = source.view()
            elif source.flags.c_contiguous:
                # Contiguity proves reshape is a view on every supported NumPy
                # version; do not attempt an allocating reshape and check later.
                value = source.reshape(target)
            else:
                raise ValueError("adapter reshape requires a contiguous source view")
            fields[view.path] = value
        lease = budget.reserve(self.staging_bytes, timeout=timeout)
        try:
            generated = {}
            for field in self.adapter.metadata:
                path = tuple(field.path)
                array = np.empty(self.shapes[path], dtype=storage_dtype(field.dtype))
                assert isinstance(field.padding, (bool, int, float))
                ValidatedTensor(
                    field.dtype, self.shapes[path], default=field.padding
                ).copy_into(array)
                generated[path] = array
            if self.adapter.fill is not None:
                self.adapter.fill(
                    AdapterContext(self.layout, routed),
                    MappingProxyType(
                        {path: value.view() for path, value in generated.items()}
                    ),
                )
            for field in self.adapter.metadata:
                path = tuple(field.path)
                array = generated[path]
                if array.shape != self.shapes[path] or array.dtype != storage_dtype(
                    field.dtype
                ):
                    raise ValueError(
                        "metadata writer changed its declared tensor layout"
                    )
                validate_values(
                    TensorPayload(field.dtype, array.shape, array.tobytes()),
                    field,
                    f"metadata {path}",
                )
                readonly = array.view()
                readonly.setflags(write=False)
                fields[path] = readonly
            lease.shrink(self.retained_bytes)
            return AdaptedInputs(routed, fields, self.dtypes, lease)
        except BaseException:
            lease.close()
            raise
