"""Live and retired creation logs through actual HTTP and child processes."""

import io
import sys
from dataclasses import replace

import pytest

from engine_tests.server import test_local_creation as local_fixture
from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from shadowspill.orchestration.launch.phases import NodePhases
from shadowspill_client import ServiceClient
from shadowspill_client._async import ConnectionClosed
from shadowspill_client.creation import CreationError
from shadowspill_client.protocol.progress import ProgressRequest
from shadowspill_client.transport.http import RemoteError


@pytest.mark.parametrize("fail", [False, True])
def test_live_creation_reconnect_failure_and_retained_logs(tmp_path, monkeypatch, fail):
    release = tmp_path / "release-build"
    original = NodePhases.__call__
    code = """
import sys, time
from pathlib import Path
print('  → Live build ✓', flush=True)
while not Path(sys.argv[1]).exists():
    time.sleep(.01)
print('  ✓ Build finished', flush=True)
if sys.argv[2] == 'True':
    print('Intentional build failure', file=sys.stderr, flush=True)
    sys.exit(7)
"""

    def launch(owner, invocation):
        if invocation.phase == "build":
            invocation = replace(
                invocation,
                command=(sys.executable, "-c", code, str(release), str(fail)),
            )
        return original(owner, invocation)

    output = io.StringIO()
    monkeypatch.setattr(NodePhases, "__call__", launch)
    monkeypatch.setattr(sys, "stdout", output)
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (transport, other):
        service = ServiceClient(address(transport), token=TOKEN, session_id="session")
        try:
            pending = service.create_training_client(config=ctx.config, progress=True)
            until(lambda: "Live build ✓" in output.getvalue())
            assert not pending.done()
            route = f"/v1/runs/session/{pending.run_id}/progress"
            with pytest.raises(RemoteError) as denied:
                other.request("POST", route, ProgressRequest())
            assert denied.value.status == 404
            with pytest.raises(RemoteError) as invalid:
                transport.request(
                    "POST", route, ProgressRequest({"build.stdout": 999999})
                )
            assert invalid.value.status == 400
            service.close()
            with pytest.raises(ConnectionClosed):
                pending.result(5)
            reconnected = ServiceClient(
                address(transport), token=TOKEN, session_id="session"
            )
            try:
                resumed = reconnected.get_training_client(pending.run_id, progress=True)
                until(lambda: output.getvalue().count("Live build ✓") == 2)
                release.touch()
                if fail:
                    with pytest.raises(CreationError):
                        resumed.result(10)
                else:
                    training = resumed.result(10)
                    training.close().result(5)
                until(lambda: ctx.jobs.used == (0, 0))
                page = transport.request("POST", route, ProgressRequest()).value
                streams = {row.stream: row for row in page.chunks}
                assert streams["startup.stdout"].complete
                assert (
                    b"Resolve installed training components"
                    in streams["startup.stdout"].bytes()
                )
                if not fail:
                    assert streams["worker-0.stdout"].complete
                    assert b"CPU worker exited" in streams["worker-0.stdout"].bytes()
                assert streams["build.stdout"].complete
                assert b"Build finished" in streams["build.stdout"].bytes()
                if fail:
                    assert (
                        b"Intentional build failure" in streams["build.stderr"].bytes()
                    )
                    assert "Intentional build failure" in output.getvalue()
                assert len(ctx.owners) == 1
            finally:
                reconnected.close()
        finally:
            release.touch()
            service.close()


def test_silent_creation_does_not_request_logs(tmp_path, monkeypatch):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (transport, _),
        ServiceClient(address(transport), token=TOKEN) as service,
    ):
        original = service._transport.request

        def request(method, route, *args, **kwargs):
            assert not route.endswith("/progress")
            return original(method, route, *args, **kwargs)

        monkeypatch.setattr(service._transport, "request", request)
        training = service.create_training_client(config=ctx.config).result(10)
        training.close().result(5)


def test_initializing_worker_logs_are_visible_before_ready(tmp_path, monkeypatch):
    release = tmp_path / "release-worker"
    child = local_fixture.CHILD.replace(
        "        report_ready(launch.startup, stream)",
        "        import time\n        from pathlib import Path\n"
        "        print('Loading model values', flush=True)\n"
        f"        while not Path({str(release)!r}).exists(): time.sleep(.01)\n"
        "        report_ready(launch.startup, stream)",
    )
    monkeypatch.setattr(local_fixture, "CHILD", child)
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    with (
        configured(tmp_path, monkeypatch, saved=True) as ctx,
        endpoint(ctx) as (transport, _),
    ):
        service = ServiceClient(address(transport), token=TOKEN, session_id="session")
        try:
            pending = service.create_training_client(
                config=ctx.config, plans=ctx.saved, progress=True
            )
            until(lambda: "Loading model values" in output.getvalue())
            assert not pending.done()
            assert "Start training workers" in output.getvalue()
            release.touch()
            training = pending.result(10)
            assert "Start training workers —" in output.getvalue()
            training.close().result(5)
        finally:
            release.touch()
            service.close()


def test_controller_log_caps_bytes_and_preserves_cursors():
    from shadowspill.server.startup.progress import StartupLog

    log = StartupLog()
    log.append("x" * (2 * log.limit_bytes))
    first, complete, capped = log.read("stdout", 0)
    assert len(first) == 16 << 10 and capped and not complete
    log.finish()
    log.append("must not extend a finished stream")
    assert log.read("stdout", log.limit_bytes) == (b"", True, True)
    with pytest.raises(ValueError, match="offset"):
        log.read("stdout", log.limit_bytes + 1)
