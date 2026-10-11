# Custom stage partitioning

Automatic partitioning follows repeated module structure. Use a custom
`PartitionPolicy` when the application has a better semantic boundary or must
control stage granularity explicitly. The [architecture contract](../architecture/partitioning.md)
defines stage occurrences, derived object contracts and scheduled tasks; the
[API reference](../python/api/frontend.md#partitionpolicy) documents the argument.

This policy assigns a new stage after a fixed number of executable FX nodes:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.fx import GraphModule
from shadowspill.pytorch import PartitionPolicy, plan_step


@dataclass(frozen=True)
class EveryNNodes(PartitionPolicy):
    nodes_per_stage: int

    def assign_stages(
        self,
        graph_module: GraphModule,
        module: nn.Module,
    ) -> Mapping[str, int]:
        del module
        if self.nodes_per_stage <= 0:
            raise ValueError("nodes_per_stage must be positive")
        executable = [
            node
            for node in graph_module.graph.nodes
            if node.op not in {"placeholder", "output", "get_attr"}
        ]
        return {
            node.name: index // self.nodes_per_stage
            for index, node in enumerate(executable)
        }


train_step = plan_step(
    model,
    objective=objective,
    optimizer=torch.optim.AdamW,
    hyperparams=("lr",),
    example_inputs=example_inputs,
    runtime=runtime,
    execution="execution",
    spill="spill",
    partition=EveryNNodes(nodes_per_stage=12),
    artifact_store=artifact_store,
)
```

The returned mapping must:

- contain every executable node exactly once and no placeholder, output, or
  `get_attr` nodes;
- use nonnegative integer labels;
- assign each label to one contiguous topological interval;
- leave the supplied graph unchanged.

Labels need not be contiguous; ShadowSpill normalizes them by first appearance.
Missing/extra nodes, noncontiguous reuse of a label, invalid values, graph
mutation, or a policy exception becomes a `CaptureError` before compilation.

Node-count partitioning is intentionally simple and is not necessarily a good
performance policy. A production policy can inspect `node.meta`, module paths,
operator targets, and the source module, while preserving the same complete
and contiguous contract. Partitioning only defines stages; graph-pair
construction, graph-pair selection, the plan search, and physical admission
remain unchanged.

## A model-scale example

[GLMStages](../../workloads/mlops/glm53_flash/partition.py) implements this same
protocol using exported module paths. Its [workload factory](../../workloads/mlops/glm53_flash/quickstart.py)
passes the policy through `plan_options`, so quickstart uses it for both geometry
search and the chosen training plan. Model operators contain no partition API
calls. Other architectures can supply their own policy in exactly the same way.
