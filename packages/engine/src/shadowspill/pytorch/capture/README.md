# PyTorch capture internals

The public entry points are `build_forward`, `build_forward_backward` and
`build_step`. This package implements their PyTorch capture passes.

| Module | Responsibility |
| --- | --- |
| `export.py` | Strict export, functionalization and explicit mutation signatures. |
| `objective.py` | Loss/metric structure and binding model inputs to an exported objective. |
| `aot.py` | Differentiate one functional graph into a forward/backward pair. |
| `provenance.py` | Track logical inputs and saved values across differentiation. |
| `gradients/` | Gradient precision, accumulation, output layout and shard padding. |
| `padding.py` | Equal-shard storage padding of task outputs, keeping their logical geometry. |
| `retention.py` | Operator recomputation decisions during AOT partitioning. |
| `artifacts/`, `schema.py`, `storage/` | Graph, input/output and aliasing descriptions. |
| `serialized_graph.py` | Lossless persistence of explicit FX task graphs. |
| `fake.py`, `inputs.py` | Prepare capture inputs and registered tensors. |
| `live_storage.py` | Storage identity of user-owned inputs, parameters, buffers and their views. |
| `forward.py` | Select a model callback while retaining registered tensor names. |
| `effects.py`, `functional.py` | Preserve ordered effects and logical tensor operations. |
| `torch_deprecations.py` | Copy graph modules and export without PyTorch's internal `LeafSpec` deprecation warning. |

Stage partitioning and compiled task construction live in the sibling
`partition/` and `graph_pairs/` packages. GPU timing, planning and runtime pools
are separate phases; capture does not perform those jobs.
