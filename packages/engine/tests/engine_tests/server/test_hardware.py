"""The public inventory/preview and startup use one placement/admission contract."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from engine_tests.training.test_setup_resolution import arguments
from shadowspill.orchestration.launch.admission import ResourceRequest
from shadowspill.server.startup import creation as creation_module
from shadowspill.server.startup.memory import resolve_memory
from shadowspill_client import Memory, Resources, ServiceClient
from shadowspill_client.creation import CreationError
from shadowspill_client.protocol.hardware import (
    HardwareDevice,
    HardwareInventory,
    HardwareNode,
    WorkerPlacement,
)
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


def test_a_registered_node_is_listed_but_a_preview_places_only_where_runs_start(
    tmp_path, monkeypatch
):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as client,
    ):
        (local,) = client.list_hardware().result(5).nodes
        # Another node registers with a free GPU while this node's is claimed:
        # automatic selection over both would prefer the registered one.
        registered = replace(local, id="registered")
        ctx.factory.nodes = SimpleNamespace(
            inventory=lambda: HardwareInventory((registered,))
        )
        claim = ctx.node.reserve(ResourceRequest({"gpu": 1}, 1024))
        try:
            listed = client.list_hardware().result(5).nodes
            assert [node.id for node in listed] == ["local", "registered"]
            assert not listed[0].devices[0].available
            assert listed[1].devices[0].available
            preview = client.describe_training(config=ctx.config).result(5)
            assert [row.node for row in preview.hardware] == ["local"]
            # Naming the registered node is refused, as creation refuses it.
            with pytest.raises(RemoteError, match="no assigned hardware"):
                client.describe_training(
                    config=ctx.config, resources=Resources(node="registered")
                ).result(5)
        finally:
            claim.close()


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


def test_automatic_budgets_are_sized_on_the_selected_workers_and_recorded(
    tmp_path, monkeypatch
):
    # 90% of a 4 GiB GPU, and of the 16 GiB node's share for each of its two
    # GPUs, in whole MiB.
    sized = Memory(device_gib=3686 / 1024, spill_gib=7372 / 1024)
    requested = {}

    def change(config):
        # The fixture's prepared Plans hold the settings creation must record.
        requested["config"] = config
        return replace(
            config,
            memory=sized,
            origins={
                **config.origins,
                "memory.device_gib": "engine_default",
                "memory.spill_gib": "engine_default",
            },
        )

    with (
        configured(tmp_path, monkeypatch, ranks=2, change=change) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as client,
    ):
        prepared = []
        prepare = creation_module.prepare_job
        monkeypatch.setattr(
            creation_module,
            "prepare_job",
            lambda root, **options: (
                prepared.append(options["config"]),
                prepare(root, **options),
            )[1],
        )
        automatic = replace(requested["config"], memory=Memory(0, 0))
        preview = client.describe_training(config=automatic).result(5)
        assert preview.config == ctx.config
        assert "3.59961 GiB device, 7.19922 GiB spill" in preview.summary()
        # A set budget is kept.
        partial = client.describe_training(
            config=replace(requested["config"], memory=Memory(2, 0))
        ).result(5)
        assert partial.config.memory == Memory(device_gib=2, spill_gib=sized.spill_gib)
        assert "memory.device_gib" not in partial.config.origins
        assert partial.config.origins["memory.spill_gib"] == "engine_default"
        # Created from automatic settings, the run records and plans the sizes.
        training = client.create_training_client(config=automatic).result(10)
        assert training.get_config() == ctx.config
        assert prepared == [ctx.config]
        training.close().result(5)
        until(lambda: ctx.jobs.used == (0, 0))


def test_an_automatic_budget_takes_the_smallest_workers_share():
    def node(name, host_bytes, devices):
        return HardwareNode(
            name,
            "default",
            tuple(
                HardwareDevice(f"{name}-{index}", index, None, 8 << 30, True)
                for index in range(devices)
            ),
            host_bytes,
            host_bytes,
        )

    small, large = node("small", 8 << 30, 4), node("large", 64 << 30, 2)
    workers = (
        WorkerPlacement(0, "large", large.devices[0]),
        WorkerPlacement(1, "small", small.devices[0]),
    )
    inventory = HardwareInventory((small, large))
    resolved = resolve_memory(Memory(device_gib=1, spill_gib=0), workers, inventory)
    # The small node's 2 GiB per GPU, not the large node's 32 GiB.
    assert resolved == Memory(device_gib=1, spill_gib=1843 / 1024)
    with pytest.raises(ValueError, match="no memory for a spill budget"):
        resolve_memory(
            Memory(device_gib=1, spill_gib=0),
            workers,
            HardwareInventory((node("small", 1 << 20, 4), large)),
        )
