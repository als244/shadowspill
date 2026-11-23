"""A cached plan's key follows the layout of what the plan records."""

from __future__ import annotations

from dataclasses import make_dataclass

from shadowspill.planning.plan_store.records import record_layout


def layout(leaf_fields):
    leaf = make_dataclass("Leaf", leaf_fields, module=__name__)
    root = make_dataclass(
        "Root", [("leaves", tuple[leaf, ...]), ("name", str)], module=__name__
    )
    return record_layout(root)


def test_a_field_changed_anywhere_in_a_record_changes_its_layout():
    assert layout([("size", int)]) == layout([("size", int)])
    assert layout([("size", int)]) != layout([("size", int), ("end", int)])
    assert layout([("size", int)]) != layout([("bytes", int)])
