# Advanced service client

Start with [basic training](../../examples/client-training.md). This reference covers
setup controls, saved Plans, recovery, artifact transfers and diagnostics.
Hardware inventory, optional resource selection and planning before initialization
are covered in [Compute resources](compute-resources.md).
Operator setup, storage and the node-agent configuration contract are in
[Service deployment](deployment.md).

## Contents

- [Connection and setup](#connection-and-setup)
- [Saved Plans and reconnect](#saved-plans-and-reconnect)
- [Preparation and local planning](#preparation-and-local-planning)
- [Runtime operations](#runtime-operations)
- [Metrics and reports](#metrics-and-reports)
- [Checkpoints and weight exports](#checkpoints)
- [Uploads](#uploads)
- [Tracing and Nsight](#tracing-and-nsight)

## Connection and setup

Connect to a running endpoint with a client credential in `SHADOWSPILL_TOKEN`.
For listener setup, storage, definition packages and local checkpoint lookup, see
[Service deployment](deployment.md). Those operator settings are separate from
the training configuration below.

```python
from shadowspill_client import SSH, ServiceClient

# Read the deployment credential from SHADOWSPILL_TOKEN, or pass token explicitly.
from shadowspill_client import BatchConfig, Memory, Model, Objective, Optimizer, Weights

service = ServiceClient(ssh=SSH(host="training-host", server_port=8000))
preview = service.describe_training(
    model=Model("regression", initialize=Weights.random(seed=0)),
    objective=Objective("weighted_squared_error"),
    optimizer=Optimizer("adamw"),
    batch=BatchConfig(limits={"max_examples": 8}),
    memory=Memory(4, 1),
).result()
print(preview.summary())

training = service.create_training_client(
    config=preview.config,
    resources=preview.resources,
).result()
pending = training.step(global_batch, optim={"lr": 0.001})
request_id = pending.request_id
pending.accepted().result()  # Server acceptance, beyond the client input copy.
result = pending.result()
print(result.completed_updates, result.loss)
training.close().result()
service.close()
```

Use `pending.add_done_callback(callback)` to observe completion without waiting.
The callback receives the completed future, including on failure, and should
return promptly. It can record a client completion timestamp independently of
when the training loop consumes the result. `result.execution_seconds` measures
server worker execution; client submission/completion timestamps also reflect
upload, queueing and result delivery.

The setup keywords are documented in the [client guide](../client.md#create-a-training-run).
They describe the model, objective, optimizer, global batch limits and memory budget. Definitions must already
be installed on the server. Omitted initialization remains omitted; a preview
does not choose fresh weights. `config=` rechecks an exact earlier configuration
and cannot be mixed with setup keywords. The preview reports physical geometry
cases and eligible/rejected schedules; capture expansion and reuse can change
the actual build work. It contains no fabricated benchmark or task count.

Use `ServiceClient(base_url="https://service.example", ...)` for a direct
connection. Loopback HTTP supports an existing or SDK-managed SSH forward.
`session_id` identifies the caller's record session and defaults to a new opaque
ID. SSH uses ordinary OpenSSH aliases, keys, host verification and jump settings.
Neither connection path starts a server or requests a GPU allocation.

Futures support `.result(timeout=...)`, `.exception(timeout=...)` and `.done()`.
A wait timeout does not cancel work or retry a mutation. `cancel()` returns false.
`close()` or leaving the context ends connection work and releases the owned SSH
forward; it does not stop accepted server work or close a training run.

`Execution(idle_timeout_seconds=...)` optionally releases idle workers. The default
is `None`, with no idle expiry. The timer runs only while the request queue and
input staging are empty, after worker initialization and after work finishes.
Inspection, duplicate queries and writer reattachment do not extend that timer.
Expiry records `idle_timeout` on the stopped run and preserves saved results and
artifacts. It does not save a checkpoint or apply pending gradients; resume from
a saved checkpoint in a new run.

`Execution(reject_overbudget=False)` is the default: external or whole-process
GPU memory overruns are reported. Set it to `True` to reject those overruns.
Pool bounds, preparation memory admission and the scheduled token cap remain
enforced in either mode.

`Execution(max_staging_bytes=2 << 30)` sets the default 2 GiB bound for host input
snapshots, validation and packing workspace. This is separate from GPU and spill
budgets. It is an accounting limit; small requests do not allocate 2 GiB. Explicit
smaller bounds are checked during preparation. Binary batch metadata permits up
to 64 MiB, with actual storage still charged to the staging budget.

## Saved Plans and reconnect

See [download and reuse saved Plans](saved-plans.md) for single-GPU and distributed
examples, including selecting different physical GPUs on the target service.

Creation accepts raw setup keywords directly, or `config=preview.config`.
`plans=training.plans` with the same computation settings skips preparation.
The explicit initializer may select different random weights or another compatible
pretrained/artifact source. The server rechecks the captured programs, schedules
and resource limits, then retains a new immutable bundle recording that source;
the original Plans remain available. This performs no build, profiling or search.
To resume, supply
`service.create_training_client(plans=training.plans, checkpoint=checkpoint)`.
The client reads the saved configuration as bounded metadata; it does not download
or decode executable Plans. The server validates every rank before initialization
and loads the checkpoint directly. A partial checkpoint opens only its recorded
frozen-base source; a full checkpoint does not need the old fresh-weight source.
The new run starts at the saved update count with a new history and a durable
parent-checkpoint reference. Setup defaults remain the saved configuration's
defaults; per-call overrides in the checkpoint do not replace them.
The creation future exposes its
`session_id` and `run_id` before readiness. Reconnect using the same session ID
and `service.get_training_client(run_id).result()`. Writable reattachment fences
the earlier writer; `read_only=True` permits inspection without takeover.

## Preparation and local planning

Pass `progress=True` to `create_training_client`, `plan_training` or `prepare_training` to display
the existing worker logs as preparation runs. The same option on
`get_training_client`, `get_planning` or `get_preparation` displays retained logs after reconnect.
Each worker records UTC timestamps, rank/process identity, nested operation starts,
completions, failures and elapsed host durations. Build output identifies candidate
shapes and policies, CPU/compiler thread settings, graph capture, stage construction,
task compilation and kernel finalization. Profiling shows task names, cache reuse,
measured GPU durations and workspace. Planning shows each layout and ordering,
estimated execution time, retained variants and shared memory requirements,
followed by validation and archive-writing phases. Operation durations are host
wall times, separate from the GPU timings used for planning. No periodic messages
or estimated completion percentages are emitted.

Display is optional and defaults to silent. Logs remain in the service after a
disconnect or creation failure. Reads use bounded byte cursors, including across
UTF-8 boundaries; the display reports when a stream reaches its configured log
limit. A display failure does not cancel or recreate accepted server work.

Defaults come from the installed workload's preparation preset. For built-in
packed text, the microbatch token capacities are 1, 2 and 4 times the maximum
sequence length, capped by the per-worker share of the global token limit.
Duplicate sizes are removed. An 8K sequence limit and 128K global limit on one
GPU therefore prepare 8K, 16K and 32K shapes. Each shape can pack multiple shorter
sequences; a microbatch size is a token capacity, not a sequence count.

The preset tries counts that fill each worker's token allowance, rounded down,
plus one microbatch for smaller calls. Over-cap combinations are rejected before
building; scheduled padding counts toward the limit. For each eligible count,
planning compares every depth/breadth factorization, from microbatch-major to
layer-major execution. Counts and orderings reuse the same shape captures.
Actual batch packing must fit an available prepared layout.

Chunking is disabled by default. The standard `cost-input` retention policy
provides saved-state and recomputation alternatives; planning selects among them
using measured costs. It does not search other chunking or retention policies
unless requested. Inspect all resolved choices before building with
`service.describe_training(...).result().config.preparation`.

Override the candidate sizes and counts explicitly when needed:

```python
from shadowspill_client import BuildCandidates, PlanningOptions, Preparation

prepare = Preparation(
    candidates=BuildCandidates(microbatch_sizes=[8192, 16384, 32768]),
    planning=PlanningOptions(microbatch_counts=[1, 4, 8, 16]),
)
# Include prepare=prepare in describe_training or create_training_client.
```

`BuildOptions` fixes a chunking or retention policy; `BuildCandidates` can compare
several. Shape-dependent policies use complete `BuildCase` records so each policy
names its matching microbatch geometry. These are advanced controls, independent
of the batch's sequence boundaries and input placement.

For text models, the helper keeps chunk sizes paired with their input shapes and
uses chunk-aware retention automatically:

```python
from shadowspill_client.text import build_candidates

prepare = Preparation(candidates=build_candidates(
    microbatch_tokens=[8192, 16384, 32768],
    projection_chunks=[0, 8192],
))
```

Zero leaves a computation unchunked. `moe_chunks` independently controls expert
chunks; it defaults to the full microbatch. Advanced searches can add
`retention=["cost-input", "cost-attention"]`. Retention defaults to cost-input;
chunking never changes sequence attention boundaries. See the complete
[long-context example](long-context.md).

`max_sequences` limits the global batch, not each physical microbatch. The text
preset defaults to 8× `ceil(microbatch_tokens / max_sequence_length)` sequences
per microbatch, capped by the global limit. An explicit
`max_sequences_per_microbatch=128` overrides that choice in `build_candidates`.
For generic definitions the corresponding field is `BuildCase.max_datums`.
Padding gets separate entries. The supplied text adapters use
`BatchConfig.limits["max_sequence_length"]` as the kernel length bound and keep
actual sequence lengths as input tensors. Call `validate_batch()` to check actual
packing against the prepared layouts before submission.

For measured inputs without starting a training run, use the same setup with
`service.prepare_training(**setup).result()`. Initialization can be omitted.
This runs the existing build, profiling and communication phases, and releases
their resources before the future completes:

```python
pending = service.prepare_training(**setup)
prepared = pending.result()
prepared.export().result().download("measured-inputs.zip").result()
```

The standard ZIP contains `manifest.json` plus the original phase archives for
every requested capture, including recorded rejections. The manifest identifies
each file, checksum, resolved configuration and required measured case. Programs,
profiles, target memory calibration and communication measurements retain their
existing formats for inspection and the CPU planning APIs. It contains no
initialized training state or selected Plan bundle. The ordinary client needs
only NumPy to download and inspect the manifest; native planning requires the
optional engine installation. The export also describes the operation/count
variants for CPU planning without installing the model package.

`Preparation(external_headroom_bytes=768 << 20)` reserves additional GPU memory
for each worker, using the existing profiler's headroom control. This reserve is
included in fixed device overhead and the total device budget. Omission inherits
the installed preset, then the engine default of 512 MiB; explicit zero is kept.
The resolved setup records the value and its origin.

Pass `Preparation(reuse=prepared.reuse, ...)` with the desired preparation settings
to reuse compatible measured cases in a new preparation or training run. For
example, change device/spill budgets or schedule counts and initialize training
from the original captures. Exact capture identities select reusable builds;
profiling controls and worker headroom must also match to reuse measurements. Communication and
overlap additionally require the same complete capture inventory and physical
assignment within the same service allocation. Changed cases are prepared through
the ordinary phases. The new run records its source and copied file checksums;
phase workers recheck target/toolchain compatibility before planning. Reuse requires
a completed preparation owned by the authenticated caller.

For memory and bandwidth comparisons without starting training, use
[offline sweeps](offline-sweeps.md):

```python
from shadowspill import PlanningInputs

inputs = PlanningInputs.load("measured-inputs.zip")
inputs.sweep(device_gib=[12, 16, 24]).write_report("memory.html")
```

With the optional CPU planner installed, load the export and submit selected
schedules through the ordinary artifact and creation methods:

```python
from dataclasses import replace
from shadowspill import PlanningInputs

# Keep the measured shapes/counts; choose initialization and budgets in setup.
setup["prepare"] = replace(setup["prepare"], reuse=prepared.reuse)
config = service.describe_training(**setup).result().config
inputs = PlanningInputs.load("measured-inputs.zip")
proposal = inputs.plan(config=config)
reference = proposal.write("local-plans.ssp")
uploaded = service.artifacts.upload("local-plans.ssp", reference=reference).result()
training = service.create_training_client(config=config, plans=uploaded).result()
```

`inputs.sources[key].profiles` returns an editable copy. Pass edited copies as
`profiles={key: costs}` and edited `CommunicationInfo` records as
`communication={key: costs}` to `inputs.plan`. Changed task timing requires
`metadata["duration_kind"] = "estimated"`; changed communication timing requires
`metadata["timing_kind"] = "estimated"`. Resource-phase durations must still sum
to the task duration. Original files and measured memory facts stay unchanged.
Use `candidates=[...]` to select exported alternatives explicitly; omitted
candidates cannot be recovered through hidden preparation or search.

The proposal contains numeric profiles and schedule metadata, with no executable
binaries. The server obtains code from the authenticated preparation, checks
Program identity, schedule replay, input coverage and joint memory admission,
then publishes the ordinary executable Plan bundle in `training.plans`. Timing
edits and their measured values are retained as preparation evidence. That bundle
uses the existing saved-Plan/checkpoint resume path. Local proposals currently
require the same measured capture inventory and physical assignment in the same
service allocation; new shapes/counts or target calibration need another export.

After disconnecting, reconnect to the same session and call
`service.get_preparation(pending.run_id).result()`. This only queries the original
job; it never repeats measurement. A timeout or client disconnect leaves accepted
server work running. Restarted unfinished jobs are reported as stopped.

## Runtime operations

Each call below returns a future. Setup has already selected the model, objectives,
supported operations, input limits, precision and memory budgets; the runtime loop
only supplies inputs, per-call controls and consumes results.

| Method | Effect |
| --- | --- |
| `step(global_batch, optim=None, loss_params=None, grad_scale=1, outputs=None, runtime_trace=False)` | Forward, backward and one optimizer update; requires no open accumulation. |
| `forward_backward(global_batch, loss_params=None, outputs=None, runtime_trace=False)` | Add this batch's gradients to the current accumulation, without an optimizer update. |
| `optim_step(optim=None, grad_scale=1, runtime_trace=False)` | Apply the accumulated gradients once, then clear the accumulation. |
| `forward(global_batch, outputs=None, mode="evaluation", loss_params=None, runtime_trace=False)` | Evaluate/predict without an optimizer update; evaluation mode preserves training state. |
| `discard_gradients()` | End the current accumulation without applying an update. |

`global_batch` is a sequence of `Datum` records. Per-call controls are keyword-only.
`optim` and `loss_params` override installed optimizer/objective defaults for this
call. They do not rewrite the saved configuration. An objective may expose
multiple named terms through `Objective.sum(...)`; its per-term controls retain
their names. See [definitions and objective composition](definitions.md).

Normalization is explicit. For an additive objective that sums one loss per
record, either call `step(batch, grad_scale=1 / len(batch))`, or submit all pieces
with `forward_backward(piece)` and then call
`optim_step(grad_scale=1 / len(batch))`. Apply that final scale once for the complete
update. For token-mean loss, use the total number of valid target tokens instead.
The engine does not infer either denominator. Objectives with microbatch-dependent
statistics, such as MoE balancing, can have different whole/split trajectories.
Artificial padding is excluded from the balancing objective by default, while
every scheduled padding position still counts toward the hard token cap.

Training batches with no loss participation are rejected before execution. Use
an explicitly prepared `forward()` operation for output-only work. Individual
masked records are allowed within a participating batch; zero-valued losses or
gradients from valid data still constitute an update. Finish
an accumulation with `optim_step` or `discard_gradients` before starting a complete
`step` or saving a checkpoint. Closing a run does not apply pending gradients.

Other runtime methods are `save_checkpoint`, `export_weights`, `load_checkpoint`
and ordered `close`, all returning futures. Input arrays and controls are copied
before the submission returns. Bounded staging/pending counts
apply backpressure, and snapshots that cannot fit fail promptly. Earlier pending
requests become explicit dependencies, so a rejected backward contribution cannot
be silently omitted from an already submitted optimizer update. A transport error
blocks new submissions on that handle: inspect the saved IDs and explicitly
reattach. `training.requests.status(id).result()` queries lifecycle state;
`.get(id).result()` retrieves the original outcome without resubmission.

## Results and saved metadata

`training.get_config()` returns the cached immutable configuration. Tensor results
are handles in `result.outputs`; their `.download(path).result()` is explicit.
The durable outcome remains in `result.record`. A `TrainingRequestError` preserves
the server's failure record, including known execution effects and accumulation.

`training.plans.summary().result()` reads saved metadata, including the selected
operation/batch candidates and rank Plans, chunk/policy choices, representative
inputs, coverage, memory and simulated timings. Each page contains at most 32
rows and fits the same bounded metadata format. Continue explicitly with
`.summary(page=result.next_page).result()` until `next_page` is `None`. Rows have
a `kind` and immutable `values`. These reads do not load the full bundle, run the
planner or issue a training operation; they remain available after worker close
and server restart. `.plans.download(path).result()` explicitly exports the full
bundle. Simulated durations are estimates for the recorded representative inputs,
separate from observed training latency.

`saved = training.plans.save(name="baseline").result()` gives the stored bundle
a run-local name and returns an immutable Plan handle. It copies no weights or
bundle data and does not enter the training queue. Plan names are independent
of checkpoint names. Saving the same name again is idempotent.
After reconnecting with the same `session_id`, retrieve it with
`service.get_plans(run_id, name="baseline").result()`. Both operations work after
worker close and service restart, without acquiring a writer or GPU. Pass the
handle as `plans=saved` when creating another compatible training run; supply
explicit initialization or a checkpoint as usual. Downloads remain explicit.

An immutable artifact reference can be attached to the current connection with
`service.artifact(reference)`. Its `.download(local_path).result()` streams and
verifies bytes into a new local file. Existing files are preserved; a failed
transfer removes its partial destination. Tensor data is never implicitly loaded
by a metadata result.

## Metrics and reports

`training.metrics.query(after=0, limit=100).result()` reads retained scalar outcomes
without contacting workers. `service.get_metrics(run_id, ...)` provides the same
inspection after reconnect or service restart, without loading a training client.
Each record has its request identity, history, completed update count, loss and
objective metrics. Skipped and failed requests remain visible. If required tensor
publication fails after scalar collection, the terminal failure and metric history
retain the verified loss and metrics alongside the confirmed execution effect and
counters. Unavailable scalars stay absent, including a crash before the terminal
record commits. Queries preserve the collected values and perform no averaging.

Pages contain `records`, a fixed journal boundary `through`, and an optional
`next_cursor`. Follow `next_cursor` while keeping the same `through` to finish that
snapshot. For a later live poll, start after the completed snapshot's `through`.
Cursors follow durable completion order, so an earlier submitted request that
finishes later is still returned. Restored update counters may go backward;
journal positions remain increasing and histories stay distinct. Page limits are
1–128 records and the existing metadata byte/node bounds can shorten a page.
Inputs and tensor outputs are not downloaded or logged by this query.

`training.metrics.export_jsonl("metrics.jsonl").result()` streams a fixed snapshot
to a new local file. The first line is a portable `MetricSnapshot` record naming
the run and journal range; subsequent lines are `MetricRecord` values. Use
`shadowspill_client.protocol.codec.decode` to read each line. Optional `after`,
`through` and `limit` have the same meanings as the query. Export keeps one bounded
page in memory, preserves existing files and removes its incomplete destination
on failure. Later results do not change the chosen snapshot; another explicit
export can start after its `through` boundary.

Record evaluation or reward scalars with
`training.log_metrics({"eval/reward": reward}, request_id=result.request_id).result()`.
The request must already have a completed, skipped or failed outcome. The saved
annotation keeps that request's original history and update count, even if the run
has since restored an earlier checkpoint. Queries and JSONL include these entries
with `source="caller"`; automatic measurements use `source="engine"`. Caller
values cannot replace engine values or supply objective loss.

Annotations have their own journal positions, so late metrics appear in later
polls without changing earlier snapshots. Identical values for the same request
have a stable annotation ID; explicitly repeating the call after a lost reply is
idempotent. An optional `metric_id` names an immutable annotation within that
request and rejects different values under the same ID. Each call accepts up to
128 finite numeric scalars with names of at most 256 characters. This metadata
operation works after worker close or restart, including through a read-only
training handle, and requires the authenticated run owner. It does not submit a
training operation, change counters or contact a tracker.

`summary = training.report(format="html").result()` creates a stored summary of
the retained loss, metrics, request outcomes and history/update identities.
`summary.download("training.html").result()` explicitly downloads it; `format="json"`
produces a structured version. Both include the fixed snapshot and saved Plan
reference and preserve caller annotations separately from engine measurements.
Report creation works after worker close or service restart and never executes a
model. Its `after` and `through` arguments use the metric query boundaries.

Reports default to limits of 10,000 records and 64 MiB. The byte ceiling is admitted
against available artifact storage before rendering. Exceeding either limit fails
without publishing a partial report; choose an explicit smaller range or increase
`max_records` (up to 100,000) and `max_bytes` (up to 1 GiB). Repeating identical
settings for the same fixed snapshot retrieves the same immutable artifact.

## Downloads

For large files, use `artifact.download(local_path, resume=True).result()` from
the first attempt. An interrupted transfer retains bounded progress metadata and
partial bytes in `<local_path>.shadowspill-download`. After reconnecting, attach
the same reference with `service.artifact(reference)` and explicitly repeat the
download with `resume=True`. The client verifies the retained prefix before
requesting the remaining bytes and checks the complete file checksum before
publishing the destination. No exchange retries automatically. Changed references,
corrupt prefixes and concurrent writers are rejected; existing destination files
are preserved. The client removes its partial directory after success.

## Checkpoints

A checkpoint save occupies the same ordered stream as training. Finish an open
accumulation with `optim_step` or `discard_gradients` first. Later updates wait for
the complete checkpoint to commit; a completed save does not increment the update
counter. Reusing `name` updates a run-local alias while preserving older generations.

```python
pending = training.save_checkpoint(name="step-50", metadata={"data_cursor": 50})
checkpoint = pending.result()
print(checkpoint.id)
manifest = checkpoint.manifest().result()  # Small verified metadata only.
restored = training.load_checkpoint(checkpoint.id).result()
print(restored.metadata["data_cursor"])
training.close().result()
checkpoint.download("./step-50").result()  # New local directory, after worker exit.
```

`training.close().result()` waits for ordered worker shutdown and release of the
run's owned resources, so a new run can immediately request that capacity. A
timeout stops only the caller's wait. The accepted close is neither cancelled nor
resubmitted; its durable request status can report completion while the close
future is still waiting for resource cleanup.

`weights="master"` is the default; `weights="compute"` selects compute-precision
values under the existing checkpoint reconstruction rules. Caller metadata is
frozen before submission and returned in the manifest. The save future preserves
its request identity and `.accepted()` observation. Request lookup exposes the
same handle as `training.requests.get(request_id).result().checkpoint`.

## Weight exports

`training.export_weights(name="inference").result()` returns a separate
`WeightsHandle` for inference or adapters. It uses the same ordered completed-update
boundary, authoritative `weights="master"`/`"compute"` selection, metadata ownership
and atomic rank publication. It reads only model weights and persistent buffers;
optimizer, gradient, RNG and execution-temporary payloads are excluded. Verified
unchanged frozen parameters are represented by the exact base dependency. Modified
frozen state remains in the export; without known base provenance all model weights
are retained.

The supported `format="shadowspill"` records logical tensor names, tied views and
owned shard coordinates. Export accepts this format; unsupported formats fail
before worker execution. The result has
`.manifest()` and explicit `.download(path, resume=True)` methods, and a downloaded
directory can be copied with `service.artifacts.upload(...)`. It can be inspected
after workers close. Request lookup exposes `.weights`, while `.checkpoint` is
absent. An inference export cannot be passed to `load_checkpoint` or used for
optimizer resume.

Use an owned export as `Model(..., initialize=Weights.artifact(weights.id))` to
start fresh training from its model values. The server checks its dependencies,
maps the saved tensors to the new Plans and verifies every worker's archive before
initialization. Optimizer moments, gradient buffers and controls use fresh defaults;
RNG streams use the new run's `Execution.seed`, and its update count starts at zero.
Partial exports retain their exact frozen-base dependency. Complete exports need
no historical weight source. Derived buffers, such as rotary position tables,
are recreated from the installed model definition. Custom artifact formats still
use the installed model's initializer.

For the same computation, reuse the saved Plans and change only initialization:

```python
from dataclasses import replace
from shadowspill_client import Weights

weights = training.export_weights().result()
plans = training.plans
config = training.get_config()
config = replace(config, model=replace(
    config.model, initialize=Weights.artifact(weights.id)
))
training.close().result()
fresh = service.create_training_client(config=config, plans=plans).result()
```

## Restore and state conversion

`load_checkpoint` accepts an immutable ID or run-local name. Ordered acceptance
pins the name to an immutable checkpoint, exposed by the future's
`accepted().result().checkpoint`; moving the name later cannot retarget the load.
The server checks the complete rank set and dependencies, verifies delivered files,
and validates every worker's native state layout before releasing any restore.
An open accumulation must be optimized or discarded first. A confirmed load
restores the saved update count and starts a new history while request sequences
continue increasing. `result.metadata` contains the saved caller metadata.
Validation failure preserves state; failure after worker release invalidates it.
Publication failure reports the confirmed restore count and history without replay.

Directory download verifies each immutable rank file, streams serially and writes
`manifest.json` after the rank files. It does not overwrite existing destinations.
A failed download leaves the committed server checkpoint available and does not
repeat training or saving. Plans and frozen-base dependencies are separate
artifacts; downloading the checkpoint does not implicitly copy them. Saves omit
frozen parameters only when every prepared Plan proves their storage unchanged.
The manifest's `info.frozen_base` records the exact resolved initializer, installed
model source and optional weights artifact. Trainable weights, persistent buffers
and mutated frozen storage remain in the rank files. Restore checks this identity
and required artifacts before starting work, then fills only omitted parameters
through the installed bounded loader. Later saves use the restored base identity;
loading a full checkpoint clears omission provenance. In-place restore requires compatible native
state ownership. New-run resume can convert compatible logical state into a new
Plan family before starting workers; preparation reports a `converting` stage.
The run keeps the selected source in `parent_checkpoint` and records the actual
startup artifact in `initial_checkpoint`. Converted manifests record their source
checkpoint, source Plans, logical state identity and conversion version. Conversion
preserves optimizer state and weight selection, and rejects incompatible state.
Worker checkpoints also retain Python, NumPy and Torch default generator state.
Fresh runs initialize those streams from `Execution.seed`; restores use the saved
seed and state. Evaluation calls advance separate evaluation streams, leaving
training randomness unchanged. Compatible conversion
preserves each matching logical stream. New streams use versioned seed derivation
from the saved seed, restore identity and logical stream, recorded in
`manifest.info.conversion.randomness`. Physical GPU identity does not affect seeds.
Custom model-owned generators still require explicit registered state support.

`checkpoint.download(path, resume=True)` retains an incomplete directory with its
checkpoint identity and completed/partial rank files. Repeat the call explicitly
after reconnecting. Completed ranks are verified locally and need no second
download; partial ranks use the same verified suffix transfer. `manifest.json`
appears only after every rank is complete. A directory with unrelated content or
a completed manifest is preserved and rejected as a new destination. Default
`resume=False` keeps the existing behavior of removing this attempt's incomplete
copy. Resumable downloads use local file locking to prevent concurrent writers.

## Uploads

Upload a local file or downloaded checkpoint directory with
`service.artifacts.upload(path).result()`. The returned handle refers to a saved
artifact; uploading does not start workers or restore training state. File paths
refer to the laptop. Keep the files unchanged until the future finishes.

Uploads send bounded chunks and retain verified progress on the server. After a
failed connection, call `upload` again explicitly, including from a new client.
It verifies the retained prefix and continues from the acknowledged offset;
exchanges are never retried automatically. A checkpoint directory keeps its exact
rank references and registers its manifest after every rank is committed. Completed
files are verified locally on retry and need no second upload.

```python
copied = other_service.artifacts.upload("./step-50").result()

# Copy setup dependencies explicitly, preserving their recorded identities.
for reference in (manifest.info.configuration, manifest.info.plans):
    path = f"./dependencies/{reference.id}"  # Existing local parent directory.
    service.artifact(reference).download(path).result()
    other_service.artifacts.upload(path, reference=reference).result()
```

Use `reference=...` when an existing manifest refers to that immutable artifact
ID, or to preserve the original checkpoint ID. Without it, standalone files use
a content-derived ID. Copying a checkpoint alone does not copy configuration,
Plans or a frozen base. Missing setup dependencies are reported before workers
start. Upload registration verifies file checksums and the complete rank set;
normal setup/restore still checks executable Plans and tensor compatibility.
For native resume with the copied Plans, `create_training_client(plans=plans,
checkpoint=copied)` reads defaults from the checkpoint's recorded configuration;
generated Plan summary pages do not need to be copied. Selecting different target
Plans uses their saved configuration, or the explicit `config` supplied by the
caller.

## Tracing and Nsight

Pass `runtime_trace=True` to a compute call to record that invocation. Its
`result.traces` contains downloadable ZIP handles, one per participating worker.
Each archive contains a request/Plan/rank/update association in `manifest.json`,
the existing `step.json` diagnostic view, and its `step.events.npz` event columns.
The artifacts remain downloadable after the training client closes.

Each trace also includes a standalone `report.html` comparing measured execution
with the exact saved Plan used by that invocation:

```python
result = training.step(batch, optim={"lr": 0.0003}, runtime_trace=True).result()
for worker, trace in enumerate(result.traces):
    trace.download_report(f"runtime-{worker}.html").result()
```

The HTML labels the measured and predicted clocks. Memory curves reconstruct
object lifetimes using traced timings; interpolated lease times and transfers
without timing evidence remain explicit. The device timeline excludes client
upload and report generation. `download_report` streams the verified archive
through a temporary local disk file, then copies only its bounded HTML member;
the laptop needs no engine or plotting libraries. Downloads do not overwrite
existing files. Use `trace.download("trace.zip").result()` for raw diagnostics.
If HTML rendering fails, `report_issue.json` records the reason and the raw trace
remains available; `download_report` raises that error without replaying training.

Optional trace assembly or publication failures appear in
`result.artifact_issues`; a confirmed update still succeeds and must not be
repeated. Required execution or cleanup failures retain their ordinary failure
semantics. Tracing is opt-in and its overhead belongs to that call.

Configure Nsight at creation with
`Execution(nsys={"start_step": 2, "steps": 2, "gpu_metrics": False})`.
The one-based window counts complete optimizer updates. For split training it
starts at the first contributing `forward_backward` and ends after the selected
terminal `optim_step`; evaluation between contributions stays inside the window.
Masked calls do not advance the count. Capture does not restart after a restore
or interrupted window. Device metrics default to enabled and require support and
permission on the assigned GPU; `gpu_metrics_frequency` selects their rate.
Omitting `nsys` launches ordinary workers without a collector.

`capture = training.nsys(rank=0).result()` reads bounded per-rank metadata.
`capture.record.status` is `pending`, `ready`, `failed` or `missing`; its `window`
records selected/completed updates, history and first/last request IDs. Native
reports finalize after worker exit, so a completed selected update can still have
pending report status. After `training.close().result()`, query until finalization
finishes. A ready report requires both complete worker evidence and successful
collector exit. If `capture.report` is present,
`capture.report.download("rank-0.nsys-rep").result()` explicitly downloads verified
native bytes. Failed captures may retain a partial report, identified by their
status and error. Capture publication failure does not repeat completed training.

After reconnecting with the same session, use
`service.get_nsys(run_id, rank=0).result()`. Final manifests and reports survive
worker close and service restart. A terminal run with only a pending record is
reported as missing finalization. These reads need no writer or live worker.
The deployment installs Nsight and provides collector host, disk and finalization
allowances through `ServerLimits.creation.nsys`; client settings contain no
server paths or executable names. Captured timings include profiler overhead.

The [regression application](../../../examples/training/README.md) supplies
global NumPy batches to one or more service workers, with whole or split updates
and explicit checkpoint resume. Its setup module owns the generic DP declaration;
the training loop needs only a ready client, inputs and results.
