"""The public inventory/preview and startup use one placement/admission contract."""

from dataclasses import replace

import pytest

from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from engine_tests.training.test_setup_resolution import arguments
from shadowspill.orchestration.launch.admission import ResourceRequest
from shadowspill_client import Resources, ServiceClient
from shadowspill_client.creation import CreationError
from shadowspill_client.transport.http import RemoteError


def test_public_hardware_inspection_tracks_claims_without_reserving(
    tmp_path, monkeypatch
):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as client,
    ):
        before = ctx.node.used
        inventory = client.list_hardware().result(5)
        (node,) = inventory.nodes
        assert node.id == "local" and node.pool == "default"
        assert node.devices[0].available
        assert (
            node.host_available_bytes
            == ctx.node.capacity.host_bytes - before.host_bytes
        )
        claim = ctx.node.reserve(ResourceRequest({"gpu": 1}, 1024))
        try:
            busy = client.list_hardware().result(5).nodes[0]
            assert not busy.devices[0].available
            assert busy.host_available_bytes == node.host_available_bytes - 1024
            usage = ctx.node.used
            preview = client.describe_training(config=ctx.config).result(5)
            assert not preview.hardware[0].device.available
            assert preview.resources == Resources(pool="default")
            assert ctx.node.used == usage  # Preview is not a reservation or a wait.
        finally:
            claim.close()
        assert client.list_hardware().result(5) == inventory
        assert ctx.node.used == before and not ctx.owners and not ctx.phases


def test_public_preview_selects_exact_order_and_creation_uses_it(tmp_path, monkeypatch):
    with (
        configured(tmp_path, monkeypatch, ranks=2) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as client,
    ):
        preview = client.describe_training(
            config=ctx.config, resources=Resources(node="local", devices=[1, 0])
        ).result(5)
        assert preview.config == ctx.config
        assert preview.resources.devices == ("gpu1", "gpu")
        assert [row.device.id for row in preview.hardware] == ["gpu1", "gpu"]
        training = client.create_training_client(
            config=preview.config,
            resources=preview.resources,
        ).result(10)
        assert ctx.owners[0]._node.devices == {0: "gpu1", 1: "gpu"}
        assert not any(
            d.available for d in client.list_hardware().result(5).nodes[0].devices
        )
        training.close().result(5)
        until(lambda: ctx.jobs.used == (0, 0))
        assert all(
            d.available for d in client.list_hardware().result(5).nodes[0].devices
        )


def test_invalid_selection_cannot_reach_a_preparation_phase(tmp_path, monkeypatch):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as client,
    ):
        invalid = Resources(node="missing")
        with pytest.raises(RemoteError, match="no assigned hardware"):
            client.describe_training(config=ctx.config, resources=invalid).result(5)
        # Raw setup follows the same preview before a run is accepted.
        with pytest.raises(RemoteError, match="no assigned hardware"):
            client.create_training_client(resources=invalid, **arguments()).result(5)
        assert not ctx.owners and not ctx.phases
        # Resolved callers are checked again by the creation owner, before launch.
        with pytest.raises(CreationError, match="no assigned hardware"):
            client.plan_training(config=ctx.config, resources=invalid).result(5)
        until(lambda: ctx.jobs.used == (0, 0))
        assert len(ctx.owners) == 1 and not ctx.phases


def test_plan_then_create_keeps_selection_outside_portable_plans(tmp_path, monkeypatch):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as client,
    ):
        preview = client.describe_training(
            config=ctx.config, resources=Resources(node="local", devices=[0])
        ).result(5)
        plans = client.plan_training(
            config=preview.config,
            resources=preview.resources,
        ).result(10)
        assert ctx.jobs.used == (0, 0)
        assert client.list_hardware().result(5).nodes[0].devices[0].available
        assert "resources" not in preview.config.to_dict()
        training = client.create_training_client(
            config=preview.config,
            plans=plans,
            resources=replace(preview.resources, devices=None),
        ).result(10)
        training.close().result(5)
        until(lambda: ctx.jobs.used == (0, 0))
        assert ctx.phases == ["build", "profile", "communicate", "plan"]
