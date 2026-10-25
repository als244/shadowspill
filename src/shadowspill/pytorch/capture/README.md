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
| `retention.py` | Operator recomputation decisions during AOT partitioning. |
| `artifacts.py`, `schema.py`, `storage/` | Graph, input/output and aliasing descriptions. |
| `fake.py`, `inputs.py`, `materialize.py` | Prepare capture inputs and registered tensors. |
| `forward.py` | Select a model callback while retaining registered tensor names. |
| `effects.py`, `functional.py` | Preserve ordered effects and logical tensor operations. |

Stage partitioning and compiled task construction live in the sibling
`partition/` and `graph_pairs/` packages. GPU timing, planning and runtime pools
are separate phases; capture does not perform those jobs.
