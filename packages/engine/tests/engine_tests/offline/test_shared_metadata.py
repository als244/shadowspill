"""Portable typed tables preserve values and reject corrupt graph structure."""

import json
import math
import struct
from collections import UserDict
from enum import StrEnum
from types import MappingProxyType

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from shadowspill.artifacts._values import encoded, freeze, plain
from shadowspill.ir import ObjectSpec
from shadowspill.serialization.sharing import (
    read_record,
    shared_metadata,
    writing_metadata,
)
from shadowspill.serialization.tables import (
    HEADER,
    MAGIC,
    NODE,
    OBJECT,
    decode,
    encode,
)


def restored(value):
    return decode(encode(value), expanded_limit=1 << 20)


scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.text()
    | st.floats(allow_nan=False, allow_infinity=False)
)
values = st.recursive(
    scalars,
    lambda children: (
        st.lists(children, max_size=6)
        | st.dictionaries(st.text(), children, max_size=6)
    ),
    max_leaves=30,
)


@given(st.dictionaries(st.text(), values, max_size=6))
def test_canonical_values_survive_binary_tables(value):
    actual, _ = restored(value)
    assert encoded(actual) == encoded(value)


def test_scalar_types_signed_zero_unicode_and_enum_values_remain_exact():
    class Name(StrEnum):
        VALUE = "name"

    value = {
        "values": [0, False, 0.0, -0.0, 1, True, 1.0, -(1 << 100), 1 << 100],
        "text": ["é", "\ud800", "\x00", Name.VALUE],
    }
    actual, _ = restored(value)
    assert encoded(actual) == encoded(value)
    assert [type(v) for v in actual["values"]] == [type(v) for v in value["values"]]
    assert math.copysign(1, actual["values"][3]) == -1


def test_shared_values_decode_once_and_records_do_not_share_mutable_metadata():
    item = {
        "object_id": "weight",
        "alias_group_id": "weights",
        "offset_bytes": 0,
        "size_bytes": 16,
        "role": "parameter",
        "persistence": "run",
    }
    source = {"rows": [item, dict(item)], "metadata": {"values": [1, 2]}}
    value, nodes = restored(source)
    first, second = value["rows"]
    assert first is second
    with shared_metadata(nodes):
        one = read_record(ObjectSpec.from_value, first, "one")
        two = read_record(ObjectSpec.from_value, second, "two")
        assert one is two
        frozen = freeze(value["metadata"])
        assert freeze(value["metadata"]) is frozen
    value["metadata"]["values"][0] = 99
    assert frozen["values"] == (1, 2)
    # No identity cache outlives its owned decode, even for the same input object.
    assert read_record(ObjectSpec.from_value, first, "again") is not one


def test_deriving_artifacts_reuses_only_owned_immutable_metadata():
    original = {"nested": {"values": [1, 2]}}
    frozen = freeze(MappingProxyType(original))
    original["nested"]["values"].append(3)
    assert plain(frozen) == {"nested": {"values": [1, 2]}}
    assert freeze(frozen) is frozen
    derived = freeze({**frozen, "extra": True})
    assert derived["nested"] is frozen["nested"]
    with pytest.raises(TypeError):
        frozen["nested"]["values"] = ()
    assert encoded(derived) == encoded({"nested": {"values": [1, 2]}, "extra": True})


def test_shared_containers_are_visited_once_and_keep_canonical_bytes():
    class Counted(UserDict):
        reads = 0

        def items(self):
            self.reads += 1
            return super().items()

    item = Counted({"data": [1, "value", None]})
    value = {"rows": [item] * 32}
    data = encode(value)
    assert item.reads == 1
    assert data == encode({"rows": [dict(item) for _ in range(32)]})


def test_deeply_shared_values_do_not_expand_during_encoding():
    value = [0]
    for _ in range(50):
        value = [value, value]
    data = encode({"value": value})
    assert len(data) < 4096
    with pytest.raises(ValueError, match="expanded"):
        decode(data, expanded_limit=4096)


def test_reusing_a_subtree_at_greater_depth_still_checks_the_full_depth():
    shared = {"leaf": [1]}
    nested = shared
    for _ in range(127):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        encode({"first": shared, "second": nested})


def test_archive_write_sharing_is_scoped_and_public_copies_remain_independent():
    value = freeze({"rows": [1, 2]})
    with writing_metadata():
        first = plain(value)
        assert first is plain(value)
    second = plain(value)
    second["rows"].append(3)
    assert first == {"rows": [1, 2]}
    with writing_metadata():
        assert plain(value) is not first


@pytest.mark.parametrize("cut", [0, 7, HEADER.size - 1, -1])
def test_truncated_tables_fail_before_artifact_construction(cut):
    data = encode({"value": [1, 2]})
    with pytest.raises(ValueError, match=r"truncated|sizes"):
        decode(data[:cut], expanded_limit=1 << 20)


@pytest.mark.parametrize(
    "change", ["version", "count", "root", "flags", "kind", "edge", "range"]
)
def test_invalid_headers_and_references_are_rejected(change):
    data = bytearray(encode({"value": [1, 2]}))
    _, count, edges, size, root = HEADER.unpack_from(data)
    nodes = np.frombuffer(data, dtype=NODE, count=count, offset=HEADER.size)
    if change == "version":
        data[7] = 99
    elif change == "count":
        HEADER.pack_into(data, 0, MAGIC, 1 << 63, edges, size, root)
    elif change == "root":
        HEADER.pack_into(data, 0, MAGIC, count, edges, size, count)
    elif change == "flags":
        nodes[0]["flags"] = 1
    elif change == "kind":
        nodes[0]["kind"] = 255
    elif change == "range":
        nodes[-1]["offset"] = edges + 1
    else:
        edge_start = HEADER.size + count * NODE.itemsize
        struct.pack_into("<I", data, edge_start, root)
    with pytest.raises(ValueError):
        decode(bytes(data), expanded_limit=1 << 20)


def test_duplicate_keys_are_rejected_even_with_valid_edge_geometry():
    data = bytearray(encode({"first": 1, "second": 2}))
    _, count, edge_count, _, _ = HEADER.unpack_from(data)
    nodes = np.frombuffer(data, dtype=NODE, count=count, offset=HEADER.size)
    edges = np.frombuffer(
        data, dtype="<u4", count=edge_count, offset=HEADER.size + count * NODE.itemsize
    )
    assert nodes[-1]["kind"] == OBJECT
    offset = int(nodes[-1]["offset"])
    edges[offset + 2] = edges[offset]
    with pytest.raises(ValueError, match="duplicate"):
        decode(bytes(data), expanded_limit=1 << 20)


def test_a_small_shared_table_cannot_expand_past_the_metadata_budget():
    value = [0]
    for _ in range(16):
        value = [value, value]
    data = encode({"value": value})
    assert len(data) < 4096
    with pytest.raises(ValueError, match="expanded"):
        decode(data, expanded_limit=4096)


def test_binary_metadata_is_data_only_and_has_bounded_nesting():
    with pytest.raises(TypeError):
        encode({"function": json.loads})
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="nesting"):
        encode({"cycle": value})
    for nonfinite in (math.inf, -math.inf, math.nan):
        with pytest.raises(TypeError):
            encode({"nonfinite": nonfinite})
