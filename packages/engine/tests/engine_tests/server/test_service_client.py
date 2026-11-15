"""Public NumPy client against the actual authenticated CPU service."""

import hashlib
import io
import json
import os
import subprocess
from dataclasses import replace

import pytest

from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_local_creation import configured
from engine_tests.training.test_setup_resolution import arguments
from shadowspill_client import ServiceClient
from shadowspill_client._async import ConnectionClosed
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.setup import TrainingSetup
from shadowspill_client.transport.http import RemoteError


def address(server):
    return f"http://127.0.0.1:{server._port}"


def test_public_preview_retains_exact_configuration_and_reports_no_fake_measurement(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (server, _):
        with ServiceClient(address(server), token=TOKEN) as client:
            assert client.info.owner_id == "owner" and client.session_id
            args = arguments()
            args["model"] = replace(args["model"], initialize=None)
            preview = client.describe_training(**args).result(10)
            assert preview.config.model.initialize is None
            assert preview.config.optimizer.defaults == {"lr": 0.01}
            assert client.describe_training(config=preview.config).result(10) == preview
            assert decode(encode(preview)) == preview
            (estimate,) = preview.preparation
            assert estimate.geometry_cases == 3
            assert (
                estimate.candidate_schedules == 5 and estimate.rejected_schedules == 1
            )
            assert "no measured timings or resource reservation" in preview.summary()
            with pytest.raises(ValueError, match="overlapping"):
                client.describe_training(config=preview.config, **args)
            assert ctx.jobs.used == ctx.job_memory.used == (0, 0)
            assert not ctx.phases and not ctx.owners
        assert client._memory.used == client._work.pending.used == (0, 0)
        with pytest.raises(ConnectionClosed):
            client.describe_training(**args)


def test_credentials_from_connection_environment_and_failed_authentication_cleanup(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (server, _):
        monkeypatch.setenv("SHADOWSPILL_TOKEN", TOKEN)
        with ServiceClient(address(server), session_id="saved-session") as client:
            assert client.session_id == "saved-session"
        with pytest.raises(RemoteError) as error:
            ServiceClient(address(server), token="not-authorized" * 4)
        assert error.value.status == 403 and TOKEN not in str(error.value)
        with pytest.raises(ValueError, match="exactly one"):
            ServiceClient()
        with pytest.raises(ValueError, match="opaque identifier"):
            ServiceClient(address(server), session_id="../server-path")


def test_explicit_streamed_download_preserves_existing_files_and_removes_partial_output(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (server, _):
        payload = b"actual immutable artifact" * 10000
        reference = ArtifactRef(
            "client-download", hashlib.sha256(payload).hexdigest(), len(payload)
        )
        with ctx.service.artifacts("owner") as store:
            store.put(reference, lambda _: io.BytesIO(payload))
        with ServiceClient(address(server), token=TOKEN) as client:
            artifact = client.artifact(reference)
            destination = tmp_path / "download.bin"
            assert artifact.download(destination).result(5) == destination
            assert destination.read_bytes() == payload
            with pytest.raises(FileExistsError):
                artifact.download(destination).result(5)
            assert destination.read_bytes() == payload

            def failed(path, reference, stream):
                stream.write(b"partial")
                raise OSError("destination or transfer failed")

            monkeypatch.setattr(client._transport, "download", failed)
            partial = tmp_path / "partial.bin"
            with pytest.raises(OSError, match="transfer failed"):
                artifact.download(partial).result(5)
            assert not partial.exists()
        assert client._memory.used == client._work.pending.used == (0, 0)


@pytest.mark.skipif(
    not os.environ.get("SHADOWSPILL_DEMO_CLIENT_INSPECT_SCRIPT"),
    reason="requires the prepared separate NumPy-only demo client",
)
def test_separate_demo_client_uses_public_api_for_cpu_inspection(tmp_path, monkeypatch):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (server, _):
        args = arguments()
        args["model"] = replace(args["model"], initialize=None)
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                os.environ["SHADOWSPILL_DEMO_CLIENT_HOST"],
                os.environ["SHADOWSPILL_DEMO_CLIENT_PYTHON"],
                "-I",
                os.environ["SHADOWSPILL_DEMO_CLIENT_INSPECT_SCRIPT"],
            ],
            input=json.dumps(
                {
                    "ssh_host": "chicago",
                    "port": server._port,
                    "token": TOKEN,
                    "setup": TrainingSetup(**args).to_dict(),
                }
            ),
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        receipt = json.loads(result.stdout)
        assert receipt["client_resources_released"] and all(receipt["absent"].values())
        assert ctx.jobs.used == ctx.job_memory.used == (0, 0)
        assert not ctx.phases and not ctx.owners
        (tmp_path / "demo-inspection.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
