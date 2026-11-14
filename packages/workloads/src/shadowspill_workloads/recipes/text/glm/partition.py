"""Optional per-layer GLM partitioning; automatic build partitioning is default."""

import re
from dataclasses import dataclass

from shadowspill.pytorch import PartitionPolicy


@dataclass(frozen=True)
class GLMStages(PartitionPolicy):
    """A workload-local PartitionPolicy, shared by forward and training capture.

    One stage per text decoder layer -- attention, mHC/norm work and the MLP,
    the MoE layer's one ``mlops.moe`` / ``expert_lora`` call included -- and
    one per vision block. Modules laid out as ``...backend.experts.N`` would
    each be a stage of their own, the work after them joining the next
    layer's stage (a backward recomputes from its stage's inputs, so that
    keeps one copy of the four-stream state per layer rather than two); GLM's
    stacked experts have no such modules. Stage IDs identify occurrences, not
    unique compiled contracts.
    """

    def assign_stages(self, graph_module, module):
        result = {}
        current, previous = -1, None
        routed = set()  # text layers whose experts have begun
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
            if is_vision:
                key = ("vision", vision_block)
            elif expert is not None:
                routed.add(layer)
                key = ("text", layer, expert)
            elif layer in routed:
                # After a layer's experts, its residual combination belongs
                # with the next layer's attention.
                key = ("text", str(int(layer) + 1), None)
            else:
                key = ("text", layer, None)
            if key != previous:
                current += 1
                previous = key
            result[node.name] = current
        return result
