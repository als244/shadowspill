"""Walking the heap for tensors survives whatever else lives on it."""

import weakref

from shadowspill.pytorch.bindings import occupants


class _Referent:
    pass


def test_a_dead_weak_proxy_does_not_stop_the_walk() -> None:
    # Graph capture leaves proxies like this one behind. Asked for its class
    # once its referent is gone, a proxy raises rather than answering.
    referent = _Referent()
    kept = [weakref.proxy(referent)]
    del referent

    assert occupants((), lambda address: None) == {}
    assert type(kept[0]) is weakref.ProxyType
