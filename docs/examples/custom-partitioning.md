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
from shadowspill import build_step
from shadowspill.pytorch.partition import PartitionPolicy


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


program = build_step(
    model,
    objective=objective,
    optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False),
    example_args=(features,),
    example_kwargs={"targets": targets},
    partition=EveryNNodes(nodes_per_stage=12),
)
```

The model, objective and CPU input tensors are defined as in the
[training example](generic-training.md). Partitioning happens during build;
profiling and planning consume the resulting Program without calling the policy.

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

[GLMStages](../../packages/workloads/src/shadowspill_workloads/recipes/text/glm/partition.py) implements this same
protocol using exported module paths. The GLM [workload factory](../../packages/workloads/src/shadowspill_workloads/recipes/text/glm/quickstart.py)
passes the policy to `build_step` through its build options when asked
(`partition="glm_stages"`). Quickstart saves the resulting Program, profiles it
and searches its budgets/orderings without rebuilding the partition.
Model operators contain no partition API calls. Other architectures can supply
their own policy in exactly the same way.

## Checkpoint boundaries and execution regions

The builder's `partition=` controls checkpoint stages. Its independent
`chunking=` argument can split execution inside those stages without creating
additional saved checkpoint boundaries:

```python
from shadowspill.pytorch import ChunkRegions, IsolateOperations

execution_policy = ChunkRegions(
    partition=IsolateOperations(sequence_operator_targets, base="whole"),
    chunking=axis_policy,
    variants=("save", "recompute"),
)

# Supply these alongside the model, objective, optimizer and example inputs.
build_options = {
    "partition": checkpoint_policy,
    "chunking": execution_policy,
}
```

Here `sequence_operator_targets` names the application's sequence operations,
`checkpoint_policy` implements `PartitionPolicy`, and `axis_policy` implements
[`ChunkPolicy`](../python/api/frontend.md#axis-chunking-policies). Its
`configure_stage` callback receives each explicit forward or backward region.
Return `None` for an intact sequence operation, or `AxisChunks` with the region's
input-axis mappings and chunk size for independent work. Projection and expert
regions can use different sizes. There is no assumption that the selected axis
is the leading dimension or represents tokens.

Separate execution regions can still exchange full-axis intermediates. To keep
intermediates bounded while using different inner chunk sizes, compose the
inner regions through `AxisChunks.regions`; see
[bounded composition](../architecture/axis-chunking.md). Optional execution
lifetime rules control movement and explicitly permitted local replay.

[`GraphPairPolicy`](../python/api/frontend.md#graph-pair-policies) separately
controls the SAVE and RECOMPUTE cuts. Retention does not pin a tensor to a device:
the planner still chooses variants, placement and transfers. Quickstart can
[search each retention policy's chunk candidates](../../benchmarking/quickstart.md#searching-chunk-and-retention-policies),
profile every resulting Program, and choose the admitted Plan with the lowest
simulated time across the candidates.
