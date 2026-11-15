"""Native report presence and completed-window evidence are distinct facts."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.captures import (
    NSIGHT_MEDIA_TYPE,
    NsightManifest,
    NsightWindow,
)
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.requests import ArtifactRef


def test_nsight_manifest_requires_complete_window_history_and_report_bytes():
    window = NsightWindow(2, 2, "complete", 2, "history", "backward-1", "step-3")
    manifest = NsightManifest(
        "session",
        "run",
        0,
        ArtifactRef("plans", "a" * 64, 12),
        window,
        "ready",
        ArtifactRef("report", "b" * 64, 42, NSIGHT_MEDIA_TYPE),
    )
    assert decode(encode(manifest)) == manifest
    for options in (
        {"report": None},
        {"report": replace(manifest.report, size_bytes=0)},
        {"report": replace(manifest.report, media_type="application/octet-stream")},
        {"window": replace(window, history_id=None)},
        {"window": replace(window, state="running", completed_updates=1)},
        {"rank": 65536},
        {"error": "error"},
        {"status": "pending"},
    ):
        with pytest.raises(ValueError):
            replace(manifest, **options)
    incomplete = replace(
        window, state="incomplete", completed_updates=1, error="worker exited"
    )
    failed = replace(
        manifest, status="failed", window=incomplete, error="partial capture"
    )
    assert decode(encode(failed)) == failed  # Partial native reports can be inspected.
    with pytest.raises(ValueError):
        replace(window, completed_updates=1)
    with pytest.raises(ValueError):
        replace(window, first_request=None)
    with pytest.raises(ValueError):
        replace(window, last_request=None)
    with pytest.raises(ValueError):
        replace(window, state="incomplete", error="x" * 1025)


def test_pending_and_missing_reports_do_not_fabricate_update_evidence():
    pending = NsightManifest(
        "session",
        "run",
        0,
        ArtifactRef("plans", "a" * 64, 12),
        NsightWindow(2, 2),
    )
    assert decode(encode(pending)) == pending
    missing = replace(
        pending,
        window=replace(pending.window, state="missing", error="window never started"),
        status="missing",
        error="collector produced no report",
    )
    assert decode(encode(missing)) == missing
    for options in (
        {"first_request": "step-2"},
        {"completed_updates": 1},
        {"error": None},
    ):
        with pytest.raises(ValueError):
            replace(missing.window, **options)
