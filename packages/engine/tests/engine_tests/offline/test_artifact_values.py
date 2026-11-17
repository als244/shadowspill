"""Canonical artifact identities survive immutable metadata and JSON fast paths."""

import hashlib
from collections import UserDict

import pytest

from shadowspill.artifacts._values import digest, encoded, freeze, plain


def test_canonical_bytes_match_for_owned_and_frozen_metadata():
    source = UserDict(
        z=[True, None, 2.5, [1, 2]], a=UserDict(text="é", values=[-3, 0])
    )
    frozen = freeze(source)
    expected = (
        b'{"a":{"text":"\\u00e9","values":[-3,0]},'
        b'"z":[true,null,2.5,[1,2]]}'
    )
    for value in (source, frozen, plain(frozen)):
        assert encoded(value) == expected
        assert digest(value) == hashlib.sha256(expected).hexdigest()
    source["z"][3][0] = 99
    assert encoded(frozen) == expected
    with pytest.raises(TypeError):
        frozen["a"]["values"][0] = 99


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_invalid_metadata_is_not_admitted_by_scalar_fast_paths(value):
    with pytest.raises((TypeError, ValueError)):
        freeze({"nested": [value]})
    with pytest.raises((TypeError, ValueError)):
        encoded({"nested": [value]})
