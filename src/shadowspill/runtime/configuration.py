"""Validate runtime pools, transfer routes, and execution-device bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from shadowspill.frontend import RuntimeFrontend
from shadowspill.libraries import resolve_library
from shadowspill.memory import (
    DevicePool,
    MemoryPoolConfig,
    SpillPool,
)
from shadowspill.memory import (
    TransferRoute as TransferRouteConfig,
)

from .bootstrap import PoolBootstrap, RouteBootstrap
from .topology import MemoryPool, RuntimeRoute


class RuntimeConfigurationError(RuntimeError):
    """Raised when a runtime or plan asks for incompatible pool resources."""


def validate_topology(
    pools: Mapping[str, MemoryPoolConfig],
    routes: Mapping[str, TransferRouteConfig],
) -> tuple[dict[str, MemoryPoolConfig], dict[str, TransferRouteConfig]]:
    """The pools and routes a runtime may be built from, or why not."""

    if not isinstance(pools, Mapping):
        raise TypeError("pools must be a mapping from names to pool configurations")
    normalized = dict(pools)
    if len(normalized) < 2:
        raise RuntimeConfigurationError("a runtime requires at least two memory pools")
    for name, config in normalized.items():
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise RuntimeConfigurationError(
                f"pool name {name!r} must be a non-empty identifier"
            )
        if not isinstance(config, (DevicePool, SpillPool)):
            raise TypeError(f"unsupported pool configuration for {name!r}")
    if sum(isinstance(value, DevicePool) for value in normalized.values()) != 1:
        raise RuntimeConfigurationError(
            "the installed allocator requires exactly one device pool"
        )
    # Nothing asks which kind the spill pools are. Two pools, exactly one of
    # them the device pool, and every pool one of the two kinds of thing above
    # already means at least one spill pool -- and which kind of spill pool is
    # the loaded libraries' business, not this function's.

    if not isinstance(routes, Mapping):
        raise TypeError("routes must be a mapping from names to route configurations")
    normalized_routes = dict(routes)
    if not normalized_routes:
        raise RuntimeConfigurationError(
            "a runtime requires at least one transfer route"
        )
    endpoint_pairs: set[tuple[str, str]] = set()
    for name, route in normalized_routes.items():
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise RuntimeConfigurationError(
                f"route name {name!r} must be a non-empty identifier"
            )
        if not isinstance(route, TransferRouteConfig):
            raise TypeError(f"unsupported route configuration for {name!r}")
        for endpoint in (route.source, route.destination):
            if endpoint not in normalized:
                raise RuntimeConfigurationError(
                    f"route {name!r} references unknown pool {endpoint!r}"
                )
        pair = (route.source, route.destination)
        if pair in endpoint_pairs:
            raise RuntimeConfigurationError(
                "runtime route endpoint pairs must be unique; duplicate "
                f"{route.source!r} -> {route.destination!r}"
            )
        endpoint_pairs.add(pair)
        source = normalized[route.source]
        destination = normalized[route.destination]
        if isinstance(source, DevicePool) == isinstance(destination, DevicePool):
            raise RuntimeConfigurationError(
                "a transfer route runs between the device pool and a spill "
                "pool; exactly one endpoint must be the device pool"
            )
    return normalized, normalized_routes


@dataclass(frozen=True, slots=True)
class Topology:
    """The pools and routes a runtime is built from, validated and numbered.

    Pool and route ids are their positions in the caller's mappings, which is
    what the bootstrap records hand the allocator and what the registries the
    runtime publishes repeat.
    """

    pool_configs: Mapping[str, MemoryPoolConfig]
    route_configs: Mapping[str, TransferRouteConfig]

    @property
    def pool_names(self) -> tuple[str, ...]:
        return tuple(self.pool_configs)

    @property
    def route_names(self) -> tuple[str, ...]:
        return tuple(self.route_configs)

    @property
    def device(self) -> DevicePool:
        """The one device pool, which the allocator serves."""

        return next(
            config
            for config in self.pool_configs.values()
            if isinstance(config, DevicePool)
        )

    @property
    def allocator_pool_id(self) -> int:
        return self.pool_names.index(
            next(
                name
                for name, config in self.pool_configs.items()
                if config is self.device
            )
        )

    @property
    def pool_bootstrap(self) -> tuple[PoolBootstrap, ...]:
        return tuple(
            PoolBootstrap(
                pool_id=index,
                kind=0 if isinstance(config, DevicePool) else config.kind,
                capacity_bytes=0 if isinstance(config, DevicePool) else config.capacity,
                configuration=(
                    None if isinstance(config, DevicePool) else config.configuration()
                ),
                library=(None if isinstance(config, DevicePool) else config.library),
            )
            for index, config in enumerate(self.pool_configs.values())
        )

    @property
    def route_bootstrap(self) -> tuple[RouteBootstrap, ...]:
        names = self.pool_names
        return tuple(
            RouteBootstrap(
                route_id=index,
                name=name,
                source_pool_id=names.index(route.source),
                destination_pool_id=names.index(route.destination),
            )
            for index, (name, route) in enumerate(self.route_configs.items())
        )

    def pools(self, allocator_pool_bytes: int) -> Mapping[str, MemoryPool]:
        """The pool registry the runtime publishes, once the device pool's
        suballocatable capacity is known."""

        return MappingProxyType(
            {
                name: MemoryPool(
                    name=name,
                    pool_id=index,
                    kind=(
                        "device" if isinstance(config, DevicePool) else config.kind_name
                    ),
                    capacity=(
                        allocator_pool_bytes
                        if isinstance(config, DevicePool)
                        else config.capacity
                    ),
                    physical_capacity=(
                        config.physical_capacity
                        if isinstance(config, DevicePool)
                        else config.capacity
                    ),
                    device_ordinal=(
                        config.device if isinstance(config, DevicePool) else None
                    ),
                    # The execution pool is device memory, which the framework
                    # addresses through the backend rather than directly; every
                    # other kind answers for itself.
                    addressable=(
                        True if isinstance(config, DevicePool) else config.addressable
                    ),
                    image_bytes=(
                        0
                        if isinstance(config, DevicePool)
                        else int(getattr(config, "image_bytes", 0) or 0)
                    ),
                )
                for index, (name, config) in enumerate(self.pool_configs.items())
            }
        )

    def routes(self) -> Mapping[str, RuntimeRoute]:
        """The route registry the runtime publishes."""

        names = self.pool_names
        return MappingProxyType(
            {
                name: RuntimeRoute(
                    name=name,
                    route_id=index,
                    source=route.source,
                    destination=route.destination,
                    source_pool_id=names.index(route.source),
                    destination_pool_id=names.index(route.destination),
                )
                for index, (name, route) in enumerate(self.route_configs.items())
            }
        )


def configure_topology(
    pools: Mapping[str, MemoryPoolConfig],
    routes: Mapping[str, TransferRouteConfig],
) -> Topology:
    """The pools and routes a runtime may be built from, or why not."""

    normalized, normalized_routes = validate_topology(pools, routes)
    return Topology(normalized, normalized_routes)


def resolve_execution_device(
    frontend: RuntimeFrontend, value: object | None, pool: MemoryPool
) -> int:
    """Resolve, and when explicit select, the device a plan will execute on.

    What a device argument may be is the frontend's question, so the ordinal
    comes from the frontend; that it matches the pool the runtime was opened for
    is the runtime's.
    """

    pool_device = pool.device_ordinal
    if pool_device is None:
        raise RuntimeConfigurationError(
            f"execution pool {pool.name!r} has no accelerator device"
        )
    if value is None:
        resolved = frontend.current_device_ordinal()
    else:
        resolved = frontend.device_ordinal(value)
    if resolved != pool_device:
        raise RuntimeConfigurationError(
            f"execution_device={resolved} does not match execution pool "
            f"{pool.name!r} device={pool_device}"
        )
    if value is not None:
        frontend.select_device(resolved)
    return resolved


def adapter_path(configured: str | Path | None) -> Path:
    """The framework adapter library to load: the configured one, else the one
    installed beside the package."""

    if configured is not None:
        path = Path(configured).expanduser().resolve()
    else:
        discovered = resolve_library("libshadowspill_pytorch.so")
        if discovered is None:
            raise RuntimeConfigurationError(
                "ShadowSpill's framework adapter was not found; install "
                "ShadowSpill or build the editable checkout at its configured "
                "build location"
            )
        path = discovered
    if not path.is_file():
        raise RuntimeConfigurationError(
            f"ShadowSpill's framework adapter was not found: {path}"
        )
    return path


__all__ = [
    "RuntimeConfigurationError",
    "Topology",
    "adapter_path",
    "configure_topology",
    "resolve_execution_device",
    "validate_topology",
]
