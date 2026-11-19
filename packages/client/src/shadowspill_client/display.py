"""Optional, CPU-only command tables; observing a future never submits work."""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from shutil import get_terminal_size
from statistics import mean, median
from threading import RLock
from typing import TYPE_CHECKING, TextIO

from ._records import integer, number
from .protocol.operations import ComputeOperation

if TYPE_CHECKING:
    from .results import OperationFuture, TrainingResult


@dataclass(frozen=True)
class LengthStatistics:
    """Summary statistics of a set of sequence lengths.

    Parameters
    ----------
    count : int
        Number of sequences.
    total : int
        Sum of the lengths.
    minimum : int
        Shortest length.
    maximum : int
        Longest length.
    average : float
        Mean length.
    median : float
        Median length.
    """

    count: int
    total: int
    minimum: int
    maximum: int
    average: float
    median: float

    @classmethod
    def from_lengths(cls, lengths: Sequence[int]) -> LengthStatistics:
        """Compute the statistics of ``lengths``.

        Parameters
        ----------
        lengths : sequence of int
            One length per sequence.

        Returns
        -------
        LengthStatistics
            Their count, total, extremes, mean and median.

        Raises
        ------
        ValueError
            ``lengths`` is empty or holds a value that is not a nonnegative
            Python int.
        """
        values = tuple(integer(value, "sequence length") for value in lengths)
        if not values:
            raise ValueError("length statistics require at least one sequence")
        return cls(
            len(values),
            sum(values),
            min(values),
            max(values),
            float(mean(values)),
            float(median(values)),
        )

    def summary(self) -> str:
        """Format the statistics as ``"min/max mean/median"``.

        Returns
        -------
        str
            The minimum and maximum, then the mean and median rounded to whole
            numbers, with thousands separators, for example
            ``"345/3,507 1,126/1,013"``.
        """
        return (
            f"{self.minimum:,}/{self.maximum:,} {self.average:,.0f}/{self.median:,.0f}"
        )


@dataclass(frozen=True)
class BatchStatistics:
    """Lengths of a batch's examples before packing, for `CommandTable`.

    Lengths count the examples' own positions, along an axis the caller
    chooses: no packing padding, and no copies on cooperating workers. No
    inputs are retained.

    Parameters
    ----------
    lengths : LengthStatistics
        Input length of each example.
    supervised : LengthStatistics, optional
        Trained positions of each example. None when they are not counted.
    """

    lengths: LengthStatistics
    supervised: LengthStatistics | None = None

    @classmethod
    def from_lengths(
        cls,
        lengths: Sequence[int],
        *,
        supervised: Sequence[int] | None = None,
    ) -> BatchStatistics:
        """Compute batch statistics from per-example lengths.

        Parameters
        ----------
        lengths : sequence of int
            Input length of each example.
        supervised : sequence of int, optional
            Trained positions of each example, in the same order, each at most
            its input length. By default they are not counted.

        Returns
        -------
        BatchStatistics
            The statistics of both.

        Raises
        ------
        ValueError
            ``lengths`` or ``supervised`` is empty or holds a value that is not
            a nonnegative Python int, or ``supervised`` differs from
            ``lengths`` in count or exceeds a length.
        """
        total = LengthStatistics.from_lengths(lengths)
        if supervised is None:
            return cls(total)
        trained = LengthStatistics.from_lengths(supervised)
        if trained.count != total.count or any(
            active > size for active, size in zip(supervised, lengths, strict=True)
        ):
            raise ValueError("supervised lengths must fit the input sequences")
        return cls(total, trained)


@dataclass
class _Row:
    call: int
    operation: ComputeOperation
    submitted: datetime
    batch: BatchStatistics | None
    planned_seconds: float | None
    completed: datetime | None = None
    status: str = "pending"
    loss: float | int | None = None
    seconds: float | None = None


class CommandTable:
    """A live table of compute calls, one row per tracked call.

    `track` adds a row when a call is submitted and fills it in when the call
    completes. The table never waits, submits, validates or changes training.
    On a terminal it repaints its recent rows in place; on any other stream
    each submission and each completion appends a line, with the same call
    number. Submission and completion times use the client's local clock;
    durations and tokens per second use the service's execution time.

    Parameters
    ----------
    stream : file-like, optional
        Text stream the table writes to. Standard output by default.
    max_rows : int, default 20
        Number of recent rows kept and repainted on a terminal.

    Attributes
    ----------
    error : Exception or None
        The `OSError` or `ValueError` that stopped the table writing to its
        stream, or None. Once it is set the table writes nothing more; the
        tracked calls and their futures are unaffected.

    Raises
    ------
    ValueError
        ``max_rows`` is below one.
    """

    def __init__(self, *, stream: TextIO | None = None, max_rows: int = 20) -> None:
        self._stream = sys.stdout if stream is None else stream
        self._interactive = self._stream.isatty()
        self._rows: deque[_Row] = deque(maxlen=integer(max_rows, "max_rows", minimum=1))
        self._lock = RLock()
        self._next = 0
        self._started = False
        self._frame: list[str] = []
        self._widths = (4, 18, 9, 9, 7, 8, 8, 8, 9, 8, 5, 23, 23)
        self._heading = self._line(
            (
                "Call",
                "Operation",
                "Submitted",
                "Completed",
                "Status",
                "Loss",
                "Time",
                "Planned",
                "Tokens/s",
                "Tokens",
                "Seqs",
                "Total",
                "Supervised",
            )
        )
        self.error: Exception | None = None

    def track(
        self,
        operation: ComputeOperation,
        future: OperationFuture[TrainingResult],
        *,
        batch: BatchStatistics | None = None,
        planned_seconds: float | None = None,
    ) -> OperationFuture[TrainingResult]:
        """Add a row for a submitted compute call and fill it in on completion.

        The row shows ``pending`` at once. When ``future`` completes, it shows
        ``done``, ``skipped`` or ``failed``, the loss, and the result's
        `TrainingResult.execution_seconds`; tokens per second divides
        ``batch.lengths.total`` by that time. A value that is absent shows as
        a dash.

        Parameters
        ----------
        operation : str
            The call: ``"step"``, ``"forward_backward"``, ``"optim_step"`` or
            ``"forward"``.
        future : OperationFuture of TrainingResult
            The future the call returned.
        batch : BatchStatistics, optional
            Lengths of the submitted batch, for example from
            `text_batch_statistics`. None for ``"optim_step"``.
        planned_seconds : float, optional
            Estimated execution time, in seconds, shown beside the measured
            one, for example `BatchValidation.estimated_seconds`.

        Returns
        -------
        OperationFuture of TrainingResult
            ``future`` itself.

        Raises
        ------
        ValueError
            ``operation`` is not a compute call, ``batch`` is given for
            ``"optim_step"``, or ``planned_seconds`` is negative or not a
            finite number.
        TypeError
            ``batch`` is not a `BatchStatistics`.
        """
        if operation not in ("step", "forward_backward", "optim_step", "forward"):
            raise ValueError("the command table accepts compute operations only")
        if batch is not None and not isinstance(batch, BatchStatistics):
            raise TypeError("batch requires BatchStatistics")
        if operation == "optim_step" and batch is not None:
            raise ValueError("optim_step has no batch")
        if planned_seconds is not None:
            planned_seconds = number(planned_seconds, "planned seconds", minimum=0)
        with self._lock:
            self._next += 1
            row = _Row(
                self._next,
                operation,
                datetime.now().astimezone(),
                batch,
                planned_seconds,
            )
            self._rows.append(row)
            self._render(row)

        def complete(source: OperationFuture[TrainingResult]) -> None:
            completed = datetime.now().astimezone()
            with self._lock:
                row.completed = completed
                if source.exception(0) is not None:
                    row.status = "failed"
                else:
                    result = source.result(0)
                    row.status = "done" if result.outcome == "completed" else "skipped"
                    row.loss, row.seconds = result.loss, result.execution_seconds
                self._render(row)

        future.add_done_callback(complete)
        return future

    def _line(self, fields: Sequence[object]) -> str:
        return "  ".join(
            str(value).rjust(width)
            for value, width in zip(fields, self._widths, strict=True)
        )

    def _row(self, row: _Row) -> str:
        batch, seconds = row.batch, row.seconds
        return self._line(
            (
                row.call,
                row.operation + "()",
                row.submitted.strftime("%H:%M:%S"),
                "—" if row.completed is None else row.completed.strftime("%H:%M:%S"),
                row.status,
                "—" if row.loss is None else f"{row.loss:.4f}",
                "—" if seconds is None else f"{seconds:.2f}s",
                "—" if row.planned_seconds is None else f"{row.planned_seconds:.2f}s",
                "—"
                if not seconds or batch is None
                else f"{batch.lengths.total / seconds:,.0f}",
                "—" if batch is None else f"{batch.lengths.total:,}",
                "—" if batch is None else batch.lengths.count,
                "—" if batch is None else batch.lengths.summary(),
                "—"
                if batch is None or batch.supervised is None
                else batch.supervised.summary(),
            )
        )

    def _render(self, changed: _Row) -> None:
        # Terminal failure must not affect the operation or escape its callback.
        if self.error is not None:
            return
        try:
            self._write(changed)
        except (OSError, ValueError) as error:
            self.error = error

    def _write(self, changed: _Row) -> None:
        header = [self._heading, "─" * len(self._heading)]
        if not self._started:
            self._stream.write(
                f"Client clock: {changed.submitted:%Z (%z)}. "
                "Time: server execution. Lengths: min/max avg/med.\n"
            )
        if not self._interactive:
            if not self._started:
                self._stream.write("\n".join(header) + "\n")
            self._stream.write(self._row(changed) + "\n")
        else:
            width, height = get_terminal_size(fallback=(180, 40))
            # Account for line wrapping at the current width, including resize.
            rows = [self._row(row) for row in self._rows]

            def physical(line: str) -> int:
                return max(1, ceil(len(line) / max(1, width)))

            available = max(1, height - sum(map(physical, header)) - 2)
            while len(rows) > 1 and sum(map(physical, rows)) > available:
                rows.pop(0)
            previous = sum(map(physical, self._frame))
            if previous:
                self._stream.write(f"\x1b[{previous}A\r\x1b[J")
            self._frame = header + rows
            self._stream.write("\n".join(self._frame) + "\n")
        self._started = True
        self._stream.flush()
