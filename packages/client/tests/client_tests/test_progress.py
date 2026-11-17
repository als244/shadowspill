"""Creation progress is bounded, optional, and cannot affect accepted computation."""

from base64 import b64encode
from types import SimpleNamespace

import pytest

from shadowspill_client._progress import ProgressDisplay
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.progress import (
    LOG_CHUNK_BYTES,
    LogChunk,
    ProgressPage,
    ProgressRequest,
)
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.runs import RunInfo


def info():
    return RunInfo(
        "owner",
        "session",
        "run",
        "server",
        "a" * 64,
        ArtifactRef("config", "b" * 64, 1),
    )


def chunk(data, *, offset=0, complete=False, stream="build.stdout"):
    return LogChunk(stream, offset, b64encode(data).decode("ascii"), complete)


def test_final_drain_preserves_split_unicode_lines_and_exact_byte_cursors(capsys):
    text = "  ✓ Compile — 1.23s\n"
    data = text.encode()
    cut = data.index("✓".encode()) + 1
    pages = iter(
        [
            ProgressPage("session", "run", [chunk(data[:cut])]),
            ProgressPage(
                "session", "run", [chunk(data[cut:], offset=cut, complete=True)]
            ),
        ]
    )
    requests = []

    def request(method, route, kind, value):
        requests.append(value)
        return next(pages)

    service = SimpleNamespace(
        _request=request, _work=SimpleNamespace(check=lambda: None)
    )
    display = ProgressDisplay(True)
    display.update(service, info(), final=True)
    output = capsys.readouterr().out
    assert output.endswith(text) and "�" not in output
    assert [dict(row.offsets) for row in requests] == [{}, {"build.stdout": cut}]
    assert display._offsets == {"build.stdout": len(data)}


def test_silent_default_makes_no_log_request_or_output(capsys):
    ProgressDisplay(False).update(object(), info(), final=True)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_progress_requires_boolean(value):
    with pytest.raises(TypeError, match="boolean"):
        ProgressDisplay(value)


def test_progress_failure_disables_only_display(capsys):
    def request(*args):
        raise OSError("connection lost")

    service = SimpleNamespace(
        _request=request, _work=SimpleNamespace(check=lambda: None)
    )
    display = ProgressDisplay(True)
    display.update(service, info(), final=True)
    assert not display.enabled
    assert "creation continues" in capsys.readouterr().out
    display.update(object(), info(), final=True)


def test_protocol_limits_and_scope_survive_roundtrip():
    page = ProgressPage("session", "run", [chunk(b"x" * LOG_CHUNK_BYTES)])
    assert decode(encode(page)) == page
    request = ProgressRequest({"build.stdout": 42})
    assert decode(encode(request)) == request
    for offsets in ({"../secret": 0}, {"build.stdout": -1}, {"build.stdout": True}):
        with pytest.raises((ValueError, TypeError)):
            ProgressRequest(offsets)
    with pytest.raises(ValueError, match="bound"):
        chunk(b"x" * (LOG_CHUNK_BYTES + 1))
    with pytest.raises(ValueError, match="repeats"):
        ProgressPage("session", "run", [chunk(b"x"), chunk(b"y")])


def test_broken_output_does_not_escape_to_creation():
    display = ProgressDisplay(True)
    display._stream = SimpleNamespace(
        write=lambda _: (_ for _ in ()).throw(BrokenPipeError())
    )
    display.update(object(), info(), final=True)
    assert not display.enabled


def test_worker_progress_has_independent_bounded_cursors():
    page = ProgressPage(
        "session",
        "run",
        [
            chunk(b"rank 0 ready\n", stream="worker-0.stdout"),
            chunk(b"rank 7 loading\n", stream="worker-7.stderr"),
        ],
    )
    assert decode(encode(page)) == page
    request = ProgressRequest({"startup.stdout": 7, "worker-7.stderr": 4})
    assert decode(encode(request)) == request
    for name in ("worker--1.stdout", "worker-01.stdout", "worker-0/file.stdout"):
        with pytest.raises(ValueError):
            ProgressRequest({name: 0})
