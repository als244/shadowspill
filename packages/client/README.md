# ShadowSpill client records

Portable configuration and logical examples for the training service under
construction. This package currently supplies validated records, schemas and text helpers;
service connections and training execution will follow in later refactor stages.

```python
from shadowspill_client import BatchConfig, Datum, Model, Objective, Weights

model = Model("my_project/classifier@1", initialize=Weights.random(seed=123))
objective = Objective("my_project/classification@1")
batch = BatchConfig(limits={"max_examples": 64})
example = Datum(inputs={"pixels": pixels}, loss_inputs={"class_id": 7})
```

Configuration records own immutable copies of their JSON-compatible settings.
Datum records hold caller data; they do not themselves copy arrays. The submission
boundary will provide bounded owned snapshots before returning a request future.
The separate `shadowspill_client.text.text_datum` helper constructs shifted,
aligned next-token inputs without a tokenizer or model import.

Install `./packages/client` for these records. The package has no dependency on
the engine, model catalog or a GPU framework. Its optional `planning` extra adds
the existing engine; use a qualified CPU-only engine wheel for offline planning.

Portable input/control/output schemas live under `shadowspill_client.protocol`.
The engine uses these same records for its public
[definition hooks](../../docs/python/definitions.md).

Preparation settings expose fixed `BuildOptions`, explicit `BuildCandidates`,
`ProfilingOptions` and `PlanningOptions` under `Preparation`. `Policy` identifies
an installed server policy and its portable configuration. Omitted choices stay
distinct from explicit `None` (unchunked); candidate axes and complete coupled
`BuildCase` lists are mutually exclusive. These records preserve user choices;
server preset resolution, coverage and memory admission remain later work.

The protocol modules also define compute requests, acceptance, results and
failures. `protocol.codec` encodes a versioned, bounded canonical JSON envelope.
Tensor bytes use separate artifact references; Datum is deliberately not a wire
record. Failures say whether execution never started, completed, or had unknown
effects. Results include effective controls and immutable artifact references.
These records do not yet provide HTTP connections or request futures.
