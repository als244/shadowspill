"""Inspectable extension identities, output ordering and participation rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

from .._records import Record, names, record, text
from .inputs import DTYPES, shape


@dataclass(frozen=True)
class DefinitionSource(Record):
    """The installed package and source digest of a definition.

    A run's resolved configuration records one for every definition it uses,
    and for the engine and the client.

    Parameters
    ----------
    distribution : str
        Name of the installed distribution that provides the definition.
    version : str
        The distribution's version.
    digest : str
        SHA-256 digest of the definition's source files and dependencies, as
        64 lowercase hexadecimal digits.

    Raises
    ------
    ValueError
        ``distribution`` or ``version`` is empty, or ``digest`` is not 64
        lowercase hexadecimal digits.
    """

    distribution: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        text(self.distribution, "distribution")
        text(self.version, "version")
        if len(self.digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.digest
        ):
            raise ValueError("definition source requires a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TensorOutput(Record):
    """A tensor result an `OutputSet` declares.

    Parameters
    ----------
    dtype : str
        Element type, one of those `TensorField` accepts.
    shape : sequence of int or str
        For ``"datum"`` and ``"aligned"`` ordering, the shape of one example's
        output, with symbolic dimensions sized from that example's inputs.
        For ``"global"``, the fixed shape of the whole output.
    ordering : str
        How the output maps back to the batch: ``"datum"`` gives one value per
        example, ``"aligned"`` follows the positions of an input field and is
        split back per example, and ``"global"`` gives one value for the whole
        batch.
    alignment : str, optional
        For ``"aligned"`` only, and required there: the symbolic dimension of
        ``shape`` along which the output follows its input field.
    source : sequence of str, optional
        Path of the input field the output follows, starting with
        ``"inputs"`` or ``"loss_inputs"``. Needed when more than one field
        could be followed; only for an output with a symbolic dimension that
        is not ``"global"``.
    reduction : str, optional
        For ``"global"`` only, and required there: how the values of the
        batch's physical buffers combine: ``"sum"``, ``"min"``, ``"max"`` or
        ``"identical"``, which requires them to be equal.

    Raises
    ------
    ValueError
        The dtype or ordering is unknown, a ``"global"`` shape has a symbolic
        dimension, or ``alignment``, ``source`` or ``reduction`` does not suit
        the ordering.
    """

    dtype: str
    shape: Sequence[int | str]
    ordering: Literal["datum", "aligned", "global"]
    alignment: str | None = None
    source: Sequence[str] | None = None
    reduction: Literal["sum", "min", "max", "identical"] | None = None

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError("unsupported output dtype")
        object.__setattr__(self, "shape", shape(self.shape))
        if self.ordering not in ("datum", "aligned", "global"):
            raise ValueError("output ordering must be datum, aligned or global")
        if self.ordering == "aligned":
            text(self.alignment, "output alignment")
            if self.alignment not in self.shape:
                raise ValueError("aligned output must name a declared dimension")
        elif self.alignment is not None:
            raise ValueError("only aligned outputs declare an alignment")
        if self.source is not None:
            if (
                not isinstance(self.source, (list, tuple))
                or len(self.source) < 2
                or self.source[0] not in ("inputs", "loss_inputs")
                or not any(isinstance(d, str) for d in self.shape)
                or self.ordering == "global"
            ):
                raise ValueError("symbolic output source must name a model/loss field")
            object.__setattr__(
                self, "source", tuple(text(p, "output source") for p in self.source)
            )
        if self.ordering == "global":
            if self.reduction not in ("sum", "min", "max", "identical"):
                raise ValueError("declare a reduction for every global tensor output")
            if any(type(d) is not int for d in self.shape):
                raise ValueError("global tensor shapes must have fixed dimensions")
        elif self.reduction is not None:
            raise ValueError("only global tensor outputs declare a reduction")


@dataclass(frozen=True)
class OutputSet(Record):
    """The inputs, controls and results of one named output selection.

    An objective declares one for each name a call can pass as ``outputs``.
    A call checks the fields listed here, filling omitted optional ones with
    their defaults; other declared fields may be supplied and are checked
    too. Paths are tuples of keys within `Datum.inputs` or
    `Datum.loss_inputs`.

    Parameters
    ----------
    model_inputs : sequence of sequence of str, default ()
        Model input fields the computation reads.
    loss_inputs : sequence of sequence of str, default ()
        Objective input fields the computation reads.
    controls : sequence of str, default ()
        Names of the objective controls it reads.
    tensors : mapping of str to TensorOutput, default {}
        Tensor results, by name.
    metrics : mapping of str to str, default {}
        Scalar results, by name, each with the reduction that combines the
        values of the batch's physical buffers: ``"sum"``, ``"min"``,
        ``"max"`` or ``"identical"``, which requires them to be equal.
    forward : OutputSet, optional
        A separate selection that ``forward`` calls use under the same name,
        without a training loss. It cannot have its own ``forward``.

    Raises
    ------
    ValueError
        A path is empty or repeated, a control is repeated, a tensor and a
        metric share a name, a metric's reduction is unknown, or ``forward``
        has its own ``forward``.
    """

    model_inputs: Sequence[Sequence[str]] = ()
    loss_inputs: Sequence[Sequence[str]] = ()
    controls: Sequence[str] = ()
    tensors: Mapping[str, TensorOutput] = field(default_factory=dict)
    metrics: Mapping[str, Literal["sum", "min", "max", "identical"]] = field(
        default_factory=dict
    )
    forward: OutputSet | None = None

    def __post_init__(self) -> None:
        for name in ("model_inputs", "loss_inputs"):
            paths = []
            for path in getattr(self, name):
                if not isinstance(path, (list, tuple)) or not path:
                    raise ValueError("output input paths must be nonempty sequences")
                paths.append(tuple(text(p, name) for p in path))
            if len(set(paths)) != len(paths):
                raise ValueError(f"duplicate {name} path")
            object.__setattr__(self, name, tuple(paths))
        object.__setattr__(self, "controls", names(self.controls, "output controls"))
        tensors = {
            text(k, "output name"): record(TensorOutput, v, k)
            for k, v in self.tensors.items()
        }
        metrics = {text(k, "metric name"): v for k, v in self.metrics.items()}
        if set(tensors) & set(metrics):
            raise ValueError("tensor outputs and scalar metrics require distinct names")
        if any(v not in ("sum", "min", "max", "identical") for v in metrics.values()):
            raise ValueError(
                "unsupported metric reduction; report means from declared sums"
            )
        object.__setattr__(self, "tensors", MappingProxyType(tensors))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        if self.forward is not None:
            forward = record(OutputSet, self.forward, "forward outputs")
            if forward.forward is not None:
                raise ValueError("forward output contracts cannot be nested")
            object.__setattr__(self, "forward", forward)

    def select(self, *, forward: bool) -> OutputSet:
        """Return the selection one prepared operation uses.

        Parameters
        ----------
        forward : bool
            Whether the operation is a ``forward`` call.

        Returns
        -------
        OutputSet
            For a ``forward`` call, the declared ``forward`` selection when
            there is one; otherwise this selection, without its ``forward``.
        """
        if forward and self.forward is not None:
            return self.forward
        return self if self.forward is None else replace(self, forward=None)


@dataclass(frozen=True)
class Participation(Record):
    """Which inputs take part in a training call's loss.

    An objective declares it. A ``step`` or ``forward_backward`` call whose
    batch has no participating position is refused. Participation counts
    validity only, whatever the loss values, weights, advantages or gradient
    scale, and padding never participates.

    Parameters
    ----------
    path : sequence of str, default ()
        Path of a boolean field in `Datum.loss_inputs` whose true entries
        participate. Empty means every example participates.

    Raises
    ------
    TypeError
        ``path`` is not a list or tuple.
    ValueError
        A path component is empty or has surrounding spaces.
    """

    path: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, (list, tuple)):
            raise TypeError("participation path must be a sequence")
        object.__setattr__(
            self, "path", tuple(text(p, "participation") for p in self.path)
        )
