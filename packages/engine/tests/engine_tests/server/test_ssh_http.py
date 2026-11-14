"""The same accepted queue survives closing and reopening an actual SSH forward."""

import pytest

from client_tests.test_ssh import ssh_server as ssh_server
from engine_tests.server.test_http import ROOT, TOKEN, endpoint
from engine_tests.training.test_controller import configured
from engine_tests.training.test_prepared_requests import upload
from shadowspill_client import SSH
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.transport.http import HTTPTransport, RemoteError
from shadowspill_client.transport.ssh import SSH_BYTES, SSHTunnel


def test_ssh_disconnect_preserves_accepted_work_and_reconnect_reads_without_replay(
    tmp_path, ssh_server
):
    with (
        configured(tmp_path, blocked=True) as run,
        endpoint(tmp_path, run=run) as service,
    ):
        settings = SSH(
            "fixture", service.service.server_port, config_file=ssh_server.config
        )
        tunnel_memory, http_memory = (
            BufferBudget(SSH_BYTES, 1),
            BufferBudget(32 << 20, 2),
        )
        with SSHTunnel(settings, memory=tunnel_memory, timeout=5) as first:
            transport = HTTPTransport(
                first.start(), token=TOKEN, memory=http_memory, timeout=5
            )
            command, wire = upload()
            transport.request("POST", ROOT + "/requests", command)
            accepted = transport.upload(
                ROOT + "/requests/a/input", command.payload, [wire]
            ).value
            assert accepted.acceptance is not None and run.workers.entered.wait(5)
            first.close()
            assert not run.workers.closed and run.workers.calls == ["a"]
        assert tunnel_memory.used == (0, 0)
        with SSHTunnel(settings, memory=tunnel_memory, timeout=5) as second:
            transport = HTTPTransport(
                second.start(), token=TOKEN, memory=http_memory, timeout=5
            )
            run.workers.release.set()
            completed = transport.request(
                "GET", ROOT + "/requests/a?wait=completed&timeout=5"
            ).value
            assert completed.result.completed_updates == 1
            assert (
                transport.request("POST", ROOT + "/requests", command).value
                == completed
            )
            assert run.workers.calls == ["a"]
            unauthorized = HTTPTransport(
                second.base_url,
                token="invalid-" + "b" * 32,
                memory=http_memory,
                timeout=5,
            )
            with pytest.raises(RemoteError) as error:
                unauthorized.request("GET", "/v1/info")
            assert error.value.status == 403
        assert tunnel_memory.used == http_memory.used == (0, 0)
