"""Optional semantic stages for the example model's sequential experts."""

import re
from dataclasses import dataclass

from shadowspill.pytorch import PartitionPolicy


@dataclass(frozen=True)
class GLMStages(PartitionPolicy):
    """A workload-local PartitionPolicy, shared by forward and training capture.

    Attention/routing form a stage before the experts; each expert is separate;
    residual work after the experts forms a suffix. Each vision block is bounded
    separately. Stage IDs identify occurrences, not unique compiled contracts.
    """

    def assign_stages(self, graph_module, module):
        result = {}
        current, previous = -1, None
        for node in graph_module.graph.nodes:
            if node.op in {"placeholder", "get_attr", "output"}:
                continue
            paths = [
                entry[0]
                for entry in node.meta.get("nn_module_stack", {}).values()
                if isinstance(entry, tuple) and isinstance(entry[0], str)
            ]
            layer = next(
                (
                    m.group(1)
                    for p in reversed(paths)
                    if (m := re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", p))
                ),
                None,
            )
            expert = next(
                (
                    m.group(1)
                    for p in reversed(paths)
                    if (m := re.search(r"\.backend\.experts\.(\d+)(?:\.|$)", p))
                ),
                None,
            )
            vision_block = next(
                (
                    m.group(1)
                    for p in reversed(paths)
                    if (m := re.search(r"(?:^|\.)visual\.blocks\.(\d+)(?:\.|$)", p))
                ),
                None,
            )
            is_vision = any(re.search(r"(?:^|\.)visual(?:\.|$)", p) for p in paths)
            key = ("vision", vision_block) if is_vision else ("text", layer, expert)
            if key != previous:
                current += 1
                previous = key
            result[node.name] = current
        return result
