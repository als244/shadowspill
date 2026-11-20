"""Tensor geometry and semantic input provenance, without live ownership."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from shadowspill.task.inputs import TaskInputRole


@dataclass(frozen=True, slots=True)
class TensorGeometry:
    """The geometry of one tensor, without its storage.

    Chunk policies receive one for each tensor input in `StageInfo.inputs`,
    and an `ExecutionLifetimes` ``select`` callable one for each tensor
    argument and result in `LifetimeOperation.inputs` and
    `LifetimeOperation.outputs`.

    Attributes
    ----------
    shape : tuple of int
        The tensor's shape.
    stride : tuple of int
        Its strides, in elements.
    storage_offset : int
        Its first element's offset into its storage, in elements.
    dtype : torch.dtype
        Its element type.
    device_type : str
        Its device's type, such as ``"cuda"``.
    requires_grad : bool
        Whether it requires gradients.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    device_type: str
    requires_grad: bool

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorGeometry:
        """Return the geometry of ``tensor``.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor. Only its metadata is read.

        Returns
        -------
        TensorGeometry
            Its geometry.
        """
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            dtype=tensor.dtype,
            device_type=tensor.device.type,
            requires_grad=bool(tensor.requires_grad),
        )

    def identity(self) -> dict[str, object]:
        """Return the geometry as a JSON-compatible mapping.

        Returns
        -------
        dict
            The six attributes by name, with ``dtype`` as its string, such as
            ``"torch.float32"``.
        """
        return {
            "shape": self.shape,
            "stride": self.stride,
            "storage_offset": self.storage_offset,
            "dtype": str(self.dtype),
            "device_type": self.device_type,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True, slots=True)
class TaskInputProvenance:
    """Offline provenance and optional authentic value for one task input.

    ``representative_value`` is deliberately excluded from equality and contract
    identity. It is an occurrence-local view of initialized/user-owned CPU
    storage and never becomes semantic identity evidence.
    """

    role: TaskInputRole
    source: str | None = None
    consumer_targets: tuple[str, ...] = ()
    representative_value: torch.Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    #: Where the producer put this value when it really ran, if that is
    #: known. A captured contract is traced over fake values, and a fake
    #: kernel is free to disagree with the real one about which of an
    #: operator's results are device memory and which are host scalars
    #: describing it. Where a value really lives is observed rather than
    #: traced, so it is carried beside the value rather than in the
    #: contract, and excluded from identity for the same reason the value
    #: is: it is an occurrence-local observation, not semantic evidence.
    produced_device_type: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, TaskInputRole):
            raise TypeError("task input role has an invalid type")
        if self.source is not None and not self.source:
            raise ValueError("task input provenance source must be non-empty")
        if any(not item for item in self.consumer_targets):
            raise ValueError("task input consumer targets must be non-empty")
        value = self.representative_value
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError("representative task input must be a Tensor")
