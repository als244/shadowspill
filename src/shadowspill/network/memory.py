"""A spill pool whose memory is held by a daemon on another machine."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool

#: The ``ShadowSpillPoolKind`` value the network library registers.
REMOTE_POOL_KIND = 2

_LIBRARY = "libshadowspill_network.so"

#: What the daemon is asked for when a pool does not say. It is the only
#: selector a daemon serves today, but the word stays in the protocol: it is
#: what keeps one remote pool kind from becoming two the day a daemon can serve
#: device memory.
DEFAULT_SELECTOR = "host"


class RemoteConfiguration(ctypes.Structure):
    """``ShadowSpillRemotePoolConfiguration`` -- what this pool's ``acquire``
    is told about *this* pool.

    The runtime passes it through untouched as a ``void *``; only the network
    library reads it, which is why the layout is agreed here and in
    ``csrc/network/internal.h`` and nowhere in between.
    """

    _fields_ = [
        ("host", ctypes.c_char_p),
        ("port", ctypes.c_char_p),
        ("selector", ctypes.c_char_p),
    ]


@dataclass(frozen=True, slots=True)
class RemotePool(SpillPool):
    """Configuration for a spill pool held by a daemon on another machine.

    ``capacity`` is declared rather than discovered: the daemon is asked for
    exactly this many bytes and bringing the runtime up fails if it cannot
    serve them. That is what keeps a plan reproducible from its configuration
    alone -- a pool that quietly came up smaller would admit a plan that cannot
    run.

    The whole capacity is taken once, at create. Leases are carved from it by
    the same suballocator that serves a local pool, because nothing in the
    memory subsystem reads through a pool's address.

    Two things outside it do, and both go through this kind rather than around
    it: a lane, for the pools it was made to connect, and the object registry,
    when state is imported or exported -- which is what ``write`` and ``read``
    on the kind's ``pool_memory`` entry are for. The framework is told none of
    this beyond :attr:`addressable`, and is never handed an address here.
    """

    host: str = ""
    port: int = 0
    selector: str = DEFAULT_SELECTOR
    kind: int = REMOTE_POOL_KIND
    kind_name: str = "remote"

    #: The region is a local reservation standing in for memory on another
    #: machine, so no address in it may be dereferenced here. The kind's
    #: ``write`` and ``read`` are what move bytes across that edge.
    addressable: ClassVar[bool] = False

    #: Built in ``__post_init__`` and held for this configuration's life: the
    #: runtime borrows a pointer to it for the whole of bootstrap, so it must
    #: not be a temporary.
    _configuration: RemoteConfiguration = field(
        default_factory=RemoteConfiguration, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Named rather than `super()`: @dataclass(slots=True) rebuilds the
        # class, so a zero-argument super() closes over the class that was
        # replaced and raises at runtime.
        SpillPool.__post_init__(self)
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty address or hostname")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise TypeError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be a TCP port number")
        if not isinstance(self.selector, str) or not self.selector:
            raise ValueError("selector must be a non-empty string")
        if any(character.isspace() for character in self.selector):
            # The control channel is one request per line, so a selector with
            # whitespace in it would arrive as two arguments.
            raise ValueError("selector must not contain whitespace")
        self._configuration.host = self.host.encode()
        self._configuration.port = str(self.port).encode()
        self._configuration.selector = self.selector.encode()

    @property
    def library(self) -> Path | None:
        path = resolve_library(_LIBRARY)
        if path is None:
            raise RuntimeError(
                f"{_LIBRARY} was not found; a remote pool needs the network "
                "library, which this build did not produce"
            )
        return path

    def configuration(self) -> ctypes.Structure | None:
        return self._configuration


def remote(
    *,
    capacity: int,
    host: str,
    port: int,
    selector: str = DEFAULT_SELECTOR,
) -> RemotePool:
    """Return a remote spill-pool configuration served by one daemon."""

    return RemotePool(
        capacity=capacity, host=host, port=port, selector=selector
    )


__all__ = [
    "DEFAULT_SELECTOR",
    "REMOTE_POOL_KIND",
    "RemoteConfiguration",
    "RemotePool",
    "remote",
]
