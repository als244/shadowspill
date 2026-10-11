"""Optional semantic stages for the example model's sequential experts."""

import re
from dataclasses import dataclass

from shadowspill.pytorch import PartitionPolicy


@dataclass(frozen=True)
class GLMStages(PartitionPolicy):
    """A workload-local PartitionPolicy, shared by forward and training capture.

    Attention/routing form a stage before the experts; each expert is separate;
    the residual combination after a layer's experts joins the next layer's
    attention stage, and the last layer's forms its own. A backward recomputes
    from its stage's inputs, so every object a boundary hands to a later
    backward is retained from the forward; placing the combination with the
    next attention means one copy of the four-stream state per layer is
    retained rather than two. Each vision block is bounded separately. Stage
    IDs identify occurrences, not unique compiled contracts.
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
