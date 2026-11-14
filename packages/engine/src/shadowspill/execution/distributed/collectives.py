"""Resolve a saved collective task against the worker's ordered groups."""

from collections.abc import Mapping
from typing import Any

from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.communication.requests import CollectiveRequest
from shadowspill.communication.task import PackedCollective
from shadowspill.parallelism import GroupRef


class CollectiveTask(PackedCollective):
    def __init__(self, description: Mapping[str, Any], groups: GroupBindings):
        reference = GroupRef("collective", description["groups"])
        super().__init__(
            CollectiveRequest(
                description["operation"],
                reference.members(groups.logical_rank),
                description["tensor_bytes"],
                description["dtype"],
                inplace=description.get("inplace"),
            ),
            groups.resolve(reference),
        )
