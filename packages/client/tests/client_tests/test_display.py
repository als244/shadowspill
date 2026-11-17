"""A display observes compute futures, without changing submission or timing."""

import gc
import io
import weakref
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from shadowspill_client import Datum
from shadowspill_client import display as module
from shadowspill_client.display import BatchStatistics, CommandTable
from shadowspill_client.protocol.requests import RequestIdentity
from shadowspill_client.protocol.results import OperationResult
from shadowspill_client.results import OperationFuture, TrainingResult
from shadowspill_client.text.statistics import text_batch_statistics


def future(name):
    return OperationFuture(RequestIdentity(name, "run", 1, "writer", 1, "server"))


def result(pending, operation, *, loss=None, seconds=2):
    return TrainingResult(
        OperationResult(
            pending.identity,
            operation,
            "history",
            "completed",
            1,
            loss=loss,
            execution_seconds=seconds,
            grad_scale=1 if operation in ("step", "optim_step") else None,
        ),
        SimpleNamespace(),
    )


def test_all_compute_commands_show_pending_facts_and_server_only_times(monkeypatch):
    times = iter(
        datetime(2026, 11, 17, 1, 2, second, tzinfo=UTC) for second in range(8)
    )
    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda: next(times)))
    stream = io.StringIO()
    table = CommandTable(stream=stream)
    batch = BatchStatistics.from_lengths([2, 8], supervised=[1, 3])
    futures = []
    for operation in ("step", "forward_backward", "optim_step", "forward"):
        pending = future(operation)
        assert (
            table.track(
                operation,
                pending,
                batch=None if operation == "optim_step" else batch,
                planned_seconds=1.5,
            )
            is pending
        )
        futures.append((operation, pending))
    initial = stream.getvalue()
    assert initial.count("pending") == 4 and "2/8 5/5" in initial
    assert "1/3 2/2" in initial and "min/max avg/med" in initial
    assert all(not pending.done() for _, pending in futures)
    for operation, pending in reversed(futures):
        loss = None if operation in ("forward", "optim_step") else 1.25
        pending._set_result(result(pending, operation, loss=loss))
    output = stream.getvalue()
    assert output.count("done") == 4 and output.count("2.00s") == 4
    assert output.count("1.2500") == 2
    lines = [line for line in output.splitlines() if "done" in line]
    assert all("5" in line.split() for line in lines if "optim_step" not in line)
    optimizer = next(line for line in lines if "optim_step" in line)
    assert optimizer.split()[-5:] == ["—"] * 5


def test_completed_future_forward_loss_failure_and_broken_stream():
    stream = io.StringIO()
    table = CommandTable(stream=stream)
    pending = future("forward")
    value = result(pending, "forward", loss=0.75)
    pending._set_result(value)
    assert table.track("forward", pending).result() is value
    assert "0.7500" in stream.getvalue()
    failed = future("failed")
    table.track("optim_step", failed)
    failure = RuntimeError("do not retry")
    failed._set_exception(failure)
    assert "failed" in stream.getvalue() and failed.exception() is failure
    stream.close()
    last = future("closed-output")
    assert table.track("step", last) is last
    last._set_result(result(last, "step", loss=1))
    assert table.error is not None and last.result().loss == 1


def test_interactive_rows_are_bounded_and_do_not_keep_results(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    stream = Terminal()
    monkeypatch.setattr(module, "get_terminal_size", lambda **_: (220, 20))
    table = CommandTable(stream=stream, max_rows=3)
    for index in range(10):
        pending = future(str(index))
        table.track("step", pending)
        value = result(pending, "step", loss=index / 10)
        reference = weakref.ref(value)
        pending._set_result(value)
        del pending, value
    gc.collect()
    assert reference() is None and len(table._rows) == 3
    assert [row.call for row in table._rows] == [8, 9, 10]
    assert "\x1b[" in stream.getvalue()
    # Resize before completing the next command; repaint uses the new width.
    monkeypatch.setattr(module, "get_terminal_size", lambda **_: (110, 20))
    pending = future("last")
    table.track("forward", pending)
    assert table.error is None


def test_statistics_count_shifted_real_tokens_and_unmasked_targets_without_retaining():
    tokens = np.arange(8)
    reference = weakref.ref(tokens)
    summary = text_batch_statistics(
        [
            Datum(
                {"tokens": tokens},
                {"target_tokens": tokens, "loss_mask": tokens % 2 == 0},
            ),
            Datum({"tokens": [1, 2]}, {"target_tokens": [2, 3]}),
        ]
    )
    del tokens
    assert reference() is None
    assert summary.lengths.total == 10 and summary.lengths.summary() == "2/8 5/5"
    assert summary.supervised.total == 6
    assert text_batch_statistics([Datum({"tokens": [1, 2]})]).supervised is None


@pytest.mark.parametrize(
    "lengths,supervised", [([], None), ([1], [2]), ([1, 2], [1]), ([-1], None)]
)
def test_invalid_lengths_are_not_reported_as_batch_facts(lengths, supervised):
    with pytest.raises(ValueError):
        BatchStatistics.from_lengths(lengths, supervised=supervised)
