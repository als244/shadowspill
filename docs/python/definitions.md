# Installed model and objective definitions

The trainer refactor adds public definition hooks under
`shadowspill.training.definitions` and portable schema records under
`shadowspill_client.protocol`. Install the engine's `training` extra to use these
hooks. They currently support CPU inspection and the existing objective capture
path. Service execution and automatic input packing are still under construction.

A model definition describes an ordinary factory, validated configuration and
input schema. An objective receives `(model, inputs, loss_inputs, loss_params)`
and returns the existing `ObjectiveResult`, a scalar loss or `(loss, metrics)`.
The objective owns its reduction. `simple_loss` adapts an ordinary
`loss(predictions, targets)` function without changing its reduction.

Register both with a `Registry` using `add_model` and `add_objective`. Each
registration supplies a `DefinitionSource`: distribution name, version and
SHA-256 digest of the definition's source closure, including model and objective
dependencies. IDs contain an explicit version, for example
`my_project/image_classifier@1`. Re-registering an ID is an error. Lookups do not
import Python modules named by requests. Supplying an expected source rejects
changed definitions, including changes in editable installations.

The small [external classifier fixture](../../packages/engine/tests/engine_tests/fixtures/external_definition.py)
shows the full interface. The supplied
[regression definition](../../packages/workloads/src/shadowspill_workloads/definitions/regression.py)
uses the same registry and its existing model. Importing either description
requires no Torch import; model construction is explicit. Text and media catalog
definitions will migrate as their preparation paths become available.

`resolve_model` and `resolve_objective` validate configuration and return immutable
portable settings, input schemas and provenance. Configuration factories return
dataclasses or mappings. The resolved model's `create()` is a separate preparation
action. Objectives declare required model capabilities; resolution rejects
incompatible bindings. An objective with several named output sets supplies a
`bind(config, outputs)` function selecting the computation for that set.

Each `TensorField` describes one path in a Datum's input or loss-input tree:
dtype, per-example shape, padding and optional default, and stacking, packing or
shared-value semantics. Repeated symbolic dimensions explicitly declare aligned
positions. Matching integer sizes alone carry no alignment meaning. Capacities
declare their units, counted field axis and global sum/maximum or example count.
Scheduled capacities include padding. These descriptions do not establish
packing coverage or admit memory; that work belongs to preparation.

`Participation` identifies a boolean validity field with false padding, or
declares that each real example contributes. It does not inspect weights,
advantages, observed gradients or gradient scaling. `ControlSchema` validates
finite scalar controls and resolves setup defaults plus call-local overrides
without retaining overrides. Output sets declare needed fields/controls, tensor
ordering and scalar reductions; unknown fields and capabilities are errors.

The extension tests capture and execute a real CPU classifier, compare masked,
signed-weight gradients with an independent formula, change a runtime coefficient
without recapture, and check the external path with catalog imports blocked.
These are capture/interface checks, not qualification of the future service.

`OptimizerDefinition` registers a factory for an ordinary capturable optimizer.
The factory receives `(model, config, controls)`, where the build worker provides
tensor-valued runtime controls. Each advertised control explicitly names its
captured scalar bindings. After existing `capture_optimizer` runs,
`validate_capture` checks that those bindings are read by the captured update.
Static options and unused tensors cannot be advertised as runtime overrides.
The registry resolves nonsticky defaults before construction; it does not run
optimizer steps or introduce another optimizer implementation.

The phase engine also provides `artifacts.state_inventory.program_state(program,
logical_rank=0)`. It describes captured parameters, buffers, masters, persistent
gradients, optimizer tensors and host counters using role/name/component IDs.
Saved geometry, parameter ownership and physical storage bindings remain distinct.
The inventory serializes without importing Torch or the SDK. It detects layout,
optimizer-family and alias conflicts when matching state between Programs, and
refuses to alias different logical workers. This is the shared description for
the upcoming operation-state and checkpoint-conversion work; it does not yet make
additional mutable Plans executable through the current Trainer.
