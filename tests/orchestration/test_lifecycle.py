"""Host journals preserve real resource cleanup and exception suppression."""

import json
from contextlib import contextmanager

from orchestration.lifecycle import Lifecycle


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_timed_context_preserves_suppression_and_observes_real_cleanup(tmp_path):
    received = []

    @contextmanager
    def manager():
        try:
            yield 17
        except ValueError as error:
            received.append(str(error))
        finally:
            received.append("closed")

    path = tmp_path / "journal.jsonl"
    with Lifecycle(path) as timing, timing.managed("resource", manager()) as value:
        assert value == 17
        raise ValueError("original failure")
    assert received == ["original failure", "closed"]
    events = _events(path)
    assert [(e["name"], e["boundary"]) for e in events] == [
        ("resource.open", "begin"),
        ("resource.open", "end"),
        ("resource.close", "begin"),
        ("resource.close", "end"),
    ]
    assert all("error" not in e for e in events)
    assert all(e["timestamp_ns"] > 0 and e["monotonic_ns"] > 0 for e in events)
