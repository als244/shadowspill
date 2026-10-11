# Model and task partitioning

Partitioning divides a captured forward computation into ordered **stages**.
It is a PyTorch frontend operation, before backward construction, compilation,
profiling, and memory planning. It has the same contract for every model.

## Terms and responsibilities

| Term | Meaning |
|---|---|
| Partition policy | A caller-supplied rule assigning captured operations to stages. |
| Stage occurrence | One contiguous interval of forward operations at a particular model position and microbatch. |
| Stage contract | The inputs, outputs, explicit mutations, and storage/alias relationships derived for that occurrence. Equivalent contracts may share compiled artifacts without sharing parameter values. |
| Graph pair | A compiled forward/backward alternative for a training stage, such as save or recompute. |
| Task | One scheduled entrypoint invocation. A training stage normally contributes forward and backward tasks; optimizer and transfer actions are planned separately. |

A stage is not a named layer type, a device kernel, or a memory reservation. Ten
identically structured experts can be ten stage occurrences and share a small
number of compiled contracts. A policy cannot demand a particular number of
unique contracts: geometry, constants, mutations, and boundary roles also matter.

The policy chooses **where boundaries go**. ShadowSpill derives which objects
cross each boundary, constructs backward dependencies, compiles the alternatives,
and measures their workspace. The planner then chooses alternatives, microbatch
ordering, placement and transfers under the supplied budgets.

## Public contract

`PartitionSpec` is `"auto"`, `"whole"`, or an object implementing
`shadowspill.pytorch.PartitionPolicy`:

```python
def assign_stages(self, graph_module, module) -> Mapping[str, int]:
    ...
```

The policy receives a captured FX graph and its source module. It returns a
mapping from **FX node name** to stage label.

1. Assign every executable node exactly once. Exclude `placeholder`,
   `get_attr` and `output` nodes.
2. Labels are nonnegative integers, excluding booleans. They need not be
   consecutive or numerically sorted; first appearance determines stage order.
3. Each label occupies one contiguous interval in graph order. A label cannot
   reappear after another stage.
4. Do not modify the graph, model, parameters, or captured metadata. The callback
   describes boundaries; it must not execute model kernels or inspect live tensor
   values.
5. Make the rule deterministic for a given captured structure and configuration.
   Use structural information such as operator targets, tensor metadata and
   `nn_module_stack` module paths. Account for prefixes introduced by an
   objective or selected-forward wrapper.
6. Use a stable configuration representation for persisted request identity.
   A frozen dataclass is a useful ordinary Python implementation.

The frontend checks mapping coverage, label validity and contiguity, rejects
detected graph edits, and wraps callback failures in `CaptureError`. This
validation cannot prove that arbitrary Python callback code is free of side
effects; that remains part of the caller's contract.

A policy is applied during capture for each relevant input structure. Labels
are local to that graph. Reusing a policy across a geometry sweep does not
require exactly the same number of operations at every geometry.

## What a boundary means

The frontend derives live tensor inputs and outputs from graph dependencies.
Parameters, buffers, differentiable outputs, mutation versions, CPU control
values and aliases use the same generic capture/lowering rules as automatic
partitioning. A policy does not manually enumerate these objects.

A boundary does not copy a tensor, move it to a device, or imply that it can be
freed immediately. Objects with later users stay live according to the resulting
program. Captured CPU controls stay CPU controls; partitioning does not choose
their residency.

An opaque custom operator is one captured operation. A policy cannot split its
internal loop or GEMMs into multiple stages. A model that needs independently
schedulable experts must expose independently captured expert operations.
Similarly, internal asynchronous work must satisfy the task completion contract;
adding a stage boundary does not repair hidden stream dependencies.

For training, stages must support the derived backward dependencies. A detached
control-only stage with no path to the objective gradient may be rejected; group
that work with its differentiable consumer. Aliasing, mutation and differentiable
output requirements are checked by capture/lowering. Mapping validity alone does
not guarantee that an arbitrary boundary can be compiled.

## Where application policy belongs

Keep model-specific choices beside the model or workload, separate from its
mathematical forward. Pass the policy through the generic public API:

```text
workload PartitionPolicy
  -> plan_forward / plan_step / build_step_programs / plan_step_search(partition=...)
  -> FX stage partitioning
  -> ordinary contracts, graph pairs, profiling and planning
```

Quickstart factories return the same object in
`experiment["plan_options"]["partition"]`. The generic runner forwards these
options both to candidate search and to execution planning. It does not recognize
architecture names or substitute a model-specific policy.

The [custom partitioning example](../examples/custom-partitioning.md) shows a
small standalone policy. The [GLM workload](../../workloads/mlops/glm53_flash/README.md#partitioning)
is a model-scale example: its workload-local policy bounds individual experts
while grouping ordinary attention/routing and residual work. No GLM condition
belongs in the planner, simulator, runtime, or MLOps operators.

See also [frontend API](../python/api/frontend.md#partitionpolicy),
[graph-pair construction](graph-pair-construction.md), and
[task boundaries](task-boundaries.md).
