"""Portable writer fencing metadata needs no engine or transport implementation."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.requests import RequestIdentity, WriterLease


def test_lease_roundtrip_and_exact_fencing_fields():
    lease = WriterLease("run", "writer", 3, "server", 7)
    identity = RequestIdentity("request", "run", 7, "writer", 3, "server")
    assert decode(encode(lease)) == lease
    assert lease.permits(identity)
    assert lease.permits(replace(identity, sequence=8))
    for field, value in (
        ("run_id", "other"),
        ("writer_id", "other"),
        ("writer_epoch", 2),
        ("server_incarnation", "other"),
        ("sequence", 6),
    ):
        assert not lease.permits(replace(identity, **{field: value}))


@pytest.mark.parametrize(
    "change", [{"writer_epoch": 0}, {"next_sequence": True}, {"writer_id": "../path"}]
)
def test_invalid_lease_metadata_is_rejected(change):
    with pytest.raises(ValueError):
        WriterLease(
            **{**WriterLease("run", "writer", 1, "server", 1).to_dict(), **change}
        )
