"""Preparation messages can exceed TCPStore's per-value payload limit."""

import random
from concurrent.futures import ThreadPoolExecutor

import torch.distributed as dist

from shadowspill.pytorch.distributed._control import Control


def test_large_control_messages_and_mailbox_reuse():
    class BoundedStore:
        def __init__(self):
            self.inner = dist.HashStore()
            self.sizes = {}

        def set(self, key, value):
            assert len(value) < 8 << 20
            self.sizes[key] = len(value)
            self.inner.set(key, value)

        def get(self, key):
            return self.inner.get(key)

        def check(self, keys):
            return self.inner.check(keys)

        def delete_key(self, key):
            self.sizes.pop(key, None)
            return self.inner.delete_key(key)

    store = BoundedStore()
    controls = []
    for rank in range(2):
        control = object.__new__(Control)
        control.store = store
        control.rank = rank
        control.members = (0, 1)
        control.timeout = 30
        control.sequence = 0
        control._chunk_counts = {}
        controls.append(control)
    # Incompressible enough that compressed transport still needs multiple chunks.
    large = random.Random(37).randbytes(9 << 20).hex()
    with ThreadPoolExecutor(2) as executor:
        for round_number in range(4):
            values = [large if round_number == 0 else "small", round_number]
            futures = [
                executor.submit(c.exchange, "test", v)
                for c, v in zip(controls, values, strict=True)
            ]
            for future in futures:
                assert future.result() == tuple(values)
    assert all(size < 1024 for size in store.sizes.values())
