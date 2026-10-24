"""Bind saved weight shards to native transfer descriptors without calibration."""

from typing import TYPE_CHECKING

import torch

from shadowspill.parallelism import GroupRef

if TYPE_CHECKING:
    from shadowspill.artifacts.program import Program
    from shadowspill.communication.binding.groups import GroupBindings


class WeightTransfers:
    """Own descriptor references; registered runtime objects retain their own."""

    def __init__(self, program: "Program", groups: "GroupBindings", device: int):
        self.descriptors: dict[str, int] = {}
        cache: dict[tuple[str, int, int], int] = {}
        try:
            for alias, description in program.bindings.get("weight_shards", {}).items():
                group = groups.resolve(GroupRef(alias, description["groups"]))
                key = (
                    group.group_name,
                    description["padded_bytes"],
                    description["logical_bytes"],
                )
                if key not in cache:
                    cache[key] = int(
                        torch.ops.shadowspill._create_sharded_transfer(
                            group.group_name,
                            device,
                            description["padded_bytes"],
                            None,
                            None,
                            description["logical_bytes"],
                        )
                    )
                self.descriptors[alias] = cache[key]
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for descriptor in set(self.descriptors.values()):
            torch.ops.shadowspill._close_sharded_transfer(descriptor)
        self.descriptors.clear()
