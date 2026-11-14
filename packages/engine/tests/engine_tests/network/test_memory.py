"""A spill pool whose memory a daemon on another machine holds."""

from __future__ import annotations

import pytest

from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool
from shadowspill.network import RemotePool, remote
from shadowspill.network.memory import REMOTE_POOL_KIND


def test_remote_is_a_spill_pool_carrying_its_own_kind() -> None:
    pool = remote(capacity=1 << 30, host="10.0.0.2", port=7654)

    assert isinstance(pool, SpillPool)
    assert pool.kind == REMOTE_POOL_KIND
    assert pool.kind_name == "remote"
    assert pool.capacity == 1 << 30


def test_remote_configuration_is_what_the_kind_reads() -> None:
    pool = remote(capacity=1 << 20, host="host.example", port=9, selector="host")
    configuration = pool.configuration()

    assert configuration is not None
    assert configuration.host == b"host.example"
    # The port travels as text: the control channel is line-oriented, and
    # getaddrinfo takes a service name either way.
    assert configuration.port == b"9"
    assert configuration.selector == b"host"
    # One object for the pool's life. The runtime borrows a pointer to it, so a
    # fresh structure per call would hand it something already collected.
    assert pool.configuration() is configuration


def test_remote_refuses_what_the_control_channel_could_not_carry() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        remote(capacity=0, host="h", port=1)
    with pytest.raises(ValueError, match="non-empty address"):
        remote(capacity=1, host="", port=1)
    with pytest.raises(TypeError, match="port must be an integer"):
        remote(capacity=1, host="h", port="9")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="TCP port number"):
        remote(capacity=1, host="h", port=70000)
    # A selector with a space in it would arrive at the daemon as two
    # arguments, since the protocol is one request per line.
    with pytest.raises(ValueError, match="must not contain whitespace"):
        remote(capacity=1, host="h", port=1, selector="cuda 0")


def test_remote_names_the_library_that_serves_it() -> None:
    pool = remote(capacity=1 << 20, host="h", port=1)
    if resolve_library("libshadowspill_network.so") is None:
        # A build without the network library: asking for the pool is where the
        # absence is reported, rather than at runtime create.
        with pytest.raises(RuntimeError, match="was not found"):
            _ = pool.library
        return
    assert pool.library is not None
    assert pool.library.name.startswith("libshadowspill_network.so")


def test_remote_pool_is_frozen() -> None:
    pool = remote(capacity=1 << 20, host="h", port=1)
    with pytest.raises(AttributeError):
        pool.host = "other"  # type: ignore[misc]
    assert isinstance(pool, RemotePool)
