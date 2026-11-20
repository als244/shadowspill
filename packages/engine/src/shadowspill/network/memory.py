"""A spill pool whose memory is held by a daemon on another machine."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool

#: The ``ShadowSpillPoolKind`` value the network library registers.
REMOTE_POOL_KIND = 2

_LIBRARY = "libshadowspill_network.so"

#: What the daemon is asked for when a pool does not say, and the only
#: selector a daemon serves. The word stays in the protocol even so: it is
#: what keeps one remote pool kind from becoming two.
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
    """A spill pool in memory that a daemon on another machine holds.

    `remote` returns one. When the runtime opens, it connects to the daemon
    at ``host`` and ``port`` and asks for exactly ``capacity`` bytes, all at
    once; opening fails if the daemon cannot serve them. Each execution
    worker opens its own pool and connection, so a daemon serving several
    workers needs memory for all their capacities. It needs the network library,
    ``libshadowspill_network.so``.

    Parameters
    ----------
    capacity : int
        Bytes to ask the daemon for. Positive.
    kind : int, default 2
        The runtime's ``ShadowSpillPoolKind`` value for remote memory.
    kind_name : str, default "remote"
        The kind's name, as the runtime's pool registry reports it.
    host : str
        The daemon's address or host name. Required: the empty default is
        refused.
    port : int
        The daemon's TCP port, from 1 to 65535. Required: the default, 0, is
        refused.
    selector : str, default "host"
        Which of the daemon's memory to ask for, without whitespace.
        ``shadowspill_memory_daemon`` serves only ``"host"``.

    Raises
    ------
    TypeError
        If ``capacity`` or ``port`` is not an integer.
    ValueError
        If ``capacity`` is not positive, ``host`` or ``selector`` is empty,
        ``port`` is out of range, or ``selector`` contains whitespace.
    """

    host: str = ""
    port: int = 0
    selector: str = DEFAULT_SELECTOR
    kind: int = REMOTE_POOL_KIND
    kind_name: str = "remote"

    #: Always False: a lease's address lies in a local reservation that stands
    #: for the daemon's memory and maps none of it. The kind's ``write`` and
    #: ``read``, and the pool's lanes, move its bytes.
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
        """The network library, ``libshadowspill_network.so``.

        Raises
        ------
        RuntimeError
            If this build does not include it.
        """
        path = resolve_library(_LIBRARY)
        if path is None:
            raise RuntimeError(
                f"{_LIBRARY} was not found; a remote pool needs the network "
                "library, which this build did not produce"
            )
        return path

    def configuration(self) -> ctypes.Structure | None:
        """Return what the remote kind is told about this pool.

        Returns
        -------
        ctypes.Structure
            The ``ShadowSpillRemotePoolConfiguration`` holding ``host``,
            ``port`` and ``selector`` as text. It lives as long as the pool.
        """
        return self._configuration

    def calibration_identity(self) -> dict[str, Any]:
        """Return the pool's configuration as stable JSON, for pool memory evidence.

        Returns
        -------
        dict
            ``host``, ``port`` and ``selector``.
        """
        return {"host": self.host, "port": self.port, "selector": self.selector}


def remote(
    *,
    capacity: int,
    host: str,
    port: int,
    selector: str = DEFAULT_SELECTOR,
) -> RemotePool:
    """Return a configuration for a spill pool that a daemon serves.

    Parameters
    ----------
    capacity : int
        Bytes to ask the daemon for. Positive.
    host : str
        The daemon's address or host name.
    port : int
        The daemon's TCP port, from 1 to 65535.
    selector : str, default "host"
        Which of the daemon's memory to ask for, as in `RemotePool`.

    Returns
    -------
    RemotePool
        The configuration.

    Raises
    ------
    TypeError
        If ``capacity`` or ``port`` is not an integer.
    ValueError
        If an argument is empty or out of range, as `RemotePool` describes.
    """

    return RemotePool(capacity=capacity, host=host, port=port, selector=selector)


__all__ = [
    "DEFAULT_SELECTOR",
    "REMOTE_POOL_KIND",
    "RemoteConfiguration",
    "RemotePool",
    "remote",
]
