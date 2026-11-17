"""Canonical artifact identities survive immutable metadata and JSON fast paths."""

import hashlib
from collections import UserDict
from dataclasses import replace

import pytest

from shadowspill.artifacts._values import digest, encoded, freeze, plain

from ._example import profiled


def test_canonical_bytes_match_for_owned_and_frozen_metadata():
    source = UserDict(z=[True, None, 2.5, [1, 2]], a=UserDict(text="é", values=[-3, 0]))
    frozen = freeze(source)
    expected = b'{"a":{"text":"\\u00e9","values":[-3,0]},"z":[true,null,2.5,[1,2]]}'
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


def test_program_hashes_preserve_canonical_json_and_own_their_metadata():
    original = {"nested": {"values": [False, 0, -0.0, 1.0, "é", None]}}
    source = profiled()
    source = replace(
        source,
        program=replace(
            source.program,
            bindings=original,
            setup=original,
            executables={"update_impl": b"immutable executable"},
        ),
        setup=original,
        metadata=original,
        task_profiles={
            key: replace(record, metadata=original)
            for key, record in source.task_profiles.items()
        },
    )
    expected = [digest(artifact.to_dict()) for artifact in (source.program, source)]
    original["nested"]["values"].append("external mutation")
    for artifact, identity in zip((source.program, source), expected, strict=True):
        assert artifact.digest == identity
        editable = artifact.to_dict()
        metadata = editable.get("profiles", editable)
        metadata["setup"]["nested"]["values"].append("public-copy mutation")
        assert artifact.digest == digest(artifact.to_dict()) == identity
    changed = replace(source, metadata={"nested": {"values": [True]}})
    assert changed.digest == digest(changed.to_dict()) != source.digest
