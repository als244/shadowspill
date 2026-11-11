# GLM-5.3-Flash checkpoint LoRA

This recipe uses the same public build, measurement, planning and saved-Plan
execution APIs as other text models. Model-specific files describe checkpoint
precision, LoRA targets and the KDA input metadata. The
[shared runner](../../README.md) performs profiling, communication calibration,
candidate planning and saved-Plan training in fresh workers.

Start with the [commented Python script](example.py). Edit its paths and the
settings you want to compare. Its main configuration looks like this:

```python
from workloads.recipes import LoRA, Model, train

model = Model.from_checkpoint("/path/to/GLM-5.3-Flash", lora=LoRA())

config = {
    # Load the real checkpoint and text prepared with its tokenizer.
    "frozen_state_id": "exact-checkpoint-revision",
    "data": "/path/to/tokenized-fineweb",
    "run_dir": "runs/glm-lora",

    # One 32K sequence per GPU and update; LoRA rank/alpha default to 32.
    "ranks": 8,
    "sequence_length": 32768,
    "sequences_per_step": 1,
    "microbatch_sizes": [1],  # Exact sizes to build and compare.
    "steps": 100,
    "schedule_steps": 100,

    # Per-GPU memory and per-rank pinned-host spill, in GiB.
    "execution_gib": 48,
    "spill_gib": 96,

    # Measure unchunked, 4K and 8K candidates under each retention policy.
    "chunk_sizes": [0, 4096, 8192],
    "moe_chunk_sizes": [0],  # Inherit each projection chunk size.
    "graph_pair_policies": ["cost-input", "cost-attention", "attention-input"],
    "resolution_options": ["0", "1/4", "1/2", "3/4", "1"],
    "eval_every": 10,
    "checkpoint_every": 10,
}

# Prepare the candidates, select the fastest admitted Plan, then train.
train(model, config)
```

Run the script once, outside `torchrun`, using a fresh run directory. Importing
the example does not start training. `train` shares the CLI's defaults, settings
validation and phase workers; configuration keys use underscores and accept
`Path` objects for paths. Unknown options fail before creating a run.
Build and profiling use one
representative GPU; communication calibration and execution launch eight workers.
The [32K launcher](../glm53_flash_lora_32k_8gpu.sh) supplies the eight-GPU defaults
and a 100-update schedule. Weights, gradients, masters and optimizer data are
sharded across the DP group. The frozen base preserves checkpoint BF16/FP32
storage and FP8 data/scales. Dense GEMMs use BF16; FP8 experts use SonicMoE's FP8
LoRA path. Adapters compute in BF16 with FP32 gradients, masters and moments.
Only direct checkpoint storage is supported by the bounded loader; a requested
expert-format conversion fails before execution.

Build records provider-owned bindings for the BF16 operations in both training
and evaluation Programs. These bindings keep their measured kernel choices when
later phases use fresh compiler caches. Frozen FP8 storage is preserved; this
BF16 preparation hook does not cover the custom FP8 kernels.
The provider also records sparse forward's backend and FlashMLA LSE convention,
so execution does not repeat its first-call detection problem inside a task.
This selection binding does not package the attention binaries or its backward
compiler.

The corpus must contain `train.bin`, `val.bin` and `meta.json` from the
[text preparation tool](../data/README.md), including the base tokenizer's hash.
Each worker reads different rank-local windows. Training and validation losses
are normalized by the global token count once. `train/global_mean_loss` sums
rank contributions; `train/loss` is the worker's own contribution.

To schedule preparation and training separately, replace `train(model, config)` with
individual phase calls. Build starts a fresh directory; later calls read its
saved settings and artifacts:

```python
train(model, config, phase="build")          # One representative GPU.
saved = {"run_dir": config["run_dir"]}
train(model, saved, phase="profile")         # Measure saved task implementations.
train(model, saved, phase="communicate")     # Eight GPUs: transfers and overlap.
train(model, saved, phase="plan")            # CPU only: search and admission.
train(model, saved, phase="execute")         # Eight GPUs: load Plans and train.

# Later, reuse those measurements for another budget. No rebuild or profiling.
train(model, {**saved, "execution_gib": 40}, phase="plan")
train(model, saved, phase="execute")
```

On a scheduler, invoke only the phase for the current allocation; no GPU is
needed for `plan`. The CLI remains available through
`workloads.recipes.text.train_glm53_flash_lora` with the same option names.
Single-GPU profiling measures host transfers itself; skip `communicate` for
`ranks=1`. CPU-only `plan` reads saved artifacts and can change
`execution_gib` and `spill_gib` without importing Torch or the model.
Execution budgets include reserved external headroom; `--external-headroom-mib`
is recorded during profiling. Training and evaluation share parameters and are
jointly checked against those budgets. `--eval-every 0` omits evaluation entirely.

With several ranks, `communicate` measures host copies and collectives, then
paired task/NCCL interference for each saved training and evaluation Program.
It saves `overlap/` overlays and enriches `profiles/` and `profiled/` before
planning. The same compatible measurements can be reused across memory budgets.
New task implementations or a changed communication setup require matching
measurements. Unsupported or unstable paired samples are recorded as missing;
see [distributed calibration](../../../../docs/python/api/distributed.md#communication-calibration-and-cpu-planning).

## Microbatch sizes and orderings

`sequences_per_step` fixes the number of sequences per rank and optimizer update.
`microbatch_sizes` lists exactly how many sequences each candidate microbatch
contains; it defaults to `[1]`. Before building a fresh run, for example:

```python
config.update(
    sequences_per_step=8,       # Same total batch for every candidate.
    microbatch_sizes=[2, 4],    # Build only sizes 2 and 4.
)
```

The list must be nonempty, and every size must be a positive integer dividing
`sequences_per_step`. Repeated sizes are built once, in their first-seen order.
Invalid sizes fail before workers start. The CLI equivalent is
`--microbatch-sizes 2 4`. Changing the list requires a fresh build directory;
later phases reuse the saved list. The former maximum-size option is removed.

Each size is crossed with the chunk/retention combinations below and gets its
own training Program and task measurements. At size two, an update contains four
microbatches. CPU planning tries `1x4rp`, `2x2rp` and `4x1rp`: one pass over four,
two passes over two, or four passes over one. Within a pass, each stage processes
all its microbatches before advancing. The `r` flag reverses their backward order;
`p` pairs each terminal loss forward/backward. Both flags are enabled here.
Orderings reuse the saved Program and profiles. Search selects the lowest
predicted training time among jointly admitted training/evaluation Plans.

The total global batch is `ranks * sequences_per_step`; changing microbatch size
does not change it. The input source uses the selected geometry, while `Trainer`
executes the resulting Plan without searching or changing its input contract.
The one-sequence example and qualification below therefore have only size one
and ordering `1x1rp`.

This recipe searches all factor orderings for the selected sizes; it has no
ordering allowlist. A
[quickstart factory](../../../../benchmarking/quickstart.md#factory-inputs)
can also declare named input geometries and build controls. For an explicit
ordering list, call `plan_program(ordering=StepDataOrdering(...))` for each desired choice;
depth times breadth must equal that Program's microbatch count. See
[step ordering](../../../../docs/architecture/step.md).

## Explicit chunk and retention candidates

The default is one unchunked build with the ordinary cost/input graph pair.
The example script enables search explicitly, by setting these configuration
entries before the first `run`:

```python
config.update(
    chunk_sizes=[0, 4096, 8192],
    moe_chunk_sizes=[0],
    graph_pair_policies=["cost-input", "cost-attention", "attention-input"],
    resolution_options=["0", "1/4", "1/2", "3/4", "1"],
)
```

Each distinct chunk/retention combination gets its own build, task profiles and
overlap calibration. Planning crosses these with the configured microbatch sizes
and orderings, runs PressureFit for each, and selects the fastest admitted
training/evaluation pair. `search.json` records every result, including refusals.
Keep the list bounded: more candidates increase preparation time. `0` in
`moe_chunk_sizes` uses the projection chunk size; repeated unchunked combinations
are deduplicated. A new candidate needs a fresh build directory.
The example builds nine candidates: an unchunked control and two smaller MoE
chunk sizes under each policy. Keep the unchunked controls when comparing
throughput across memory budgets; adding chunked candidates should expand the
search choices. Add `32768` to `moe_chunk_sizes` to compare larger expert
chunks. At sequence length 32768 those experts run in one chunk, but the
composition and chunk-local retention restrictions still apply, so this is
distinct from an unchunked build. For a smaller comparison, select one
graph-pair policy while retaining both unchunked and chunked candidates.
`resolution_options` allows SAVE/RECOMPUTE mixtures within each candidate.
This recipe defaults to full recomputation; the example explicitly includes
quarter increments from no recomputation to full recomputation.

| Policy | SAVE variant | RECOMPUTE variant |
| --- | --- | --- |
| `cost-input` | Cost-based retention | Inputs and routing choices |
| `cost-attention` | Cost-based retention | Inputs, routing choices and attention's backward-required state |
| `attention-input` | Attention's backward-required state and routing choices | Inputs and routing choices |

These attention rules cover this recipe's KDA and sparse-attention overloads.
They retain the state required by backward, which can exceed the attention
output alone. They do not move checkpoint cuts or request device residency;
PressureFit still decides placement/offload. With chunking enabled, SAVE also
regenerates Sonic's sorted expert rows and routing metadata inside each chunk:
the current `ChunkLocal` contract cannot retain those coordinates across the
enclosing boundary. Supporting that would require separate saved data and
routing metadata for each chunk. The chosen route IDs remain retained. Both
variants remain available for planning.

The example policy uses the public `ChunkRegions` and `AxisChunks` interfaces
inside unchanged checkpoint regions. Sequence-wide KDA, convolution, sparse
selection and attention stay whole; shared compressed dense weights decode
outside token loops. Larger expert chunks can contain smaller projection/mHC
regions, without creating a sequence-sized intermediate between their children.
The fused mHC path requires sequence and chunk sizes divisible by32; invalid
tails fail at capture. See [the policy](policies.py) for the explicit provider
contracts and [the text helper](../../../models/shared/execution.py) for the
region selector. Advanced callers can provide their own policies to the builders.

The full-model measurements below include a combined search across unchunked and
chunked candidates. It selects 4K chunks at 32 GiB and unchunked execution at
40/48 GiB. All three selected Plans completed real training through the Python
entry point. These sampled budgets do not establish an absolute memory minimum.

Artifacts stay separate from compiler caches:

| Path under the run directory | Contents |
| --- | --- |
| `config.json`, `model.json` | Recipe settings, checkpoint/tokenizer identities and tensor inventory |
| `programs/batch-N/` or `programs/CANDIDATE/batch-N/` | Training and optional evaluation Programs |
| `profiles/`, `profiled/`, `communication/`, `overlap/` | Reusable task/transfer/paired measurements and fully profiled Programs |
| `search.json`, `selected.json` | Candidate outcomes and selected geometry/ordering |
| `plans/`, `reports/` | Saved executable Plans and simulated timelines |
| `logs/` | A log for each phase |
| `training/rank-NNNNN/` | Per-rank metrics and partial checkpoint archives |
| `training/evaluation.jsonl` | Global held-out losses written by rank zero |
| `learning.json`, `completed.json` | Optional fixed-batch check and completed progress |

For one GPU, metrics/checkpoints live directly under `training/`. Checkpoints
contain adapters, optimizer state and loop/data progress, with frozen weights
reloaded through bounded buffers from the same base checkpoint. For example:

```python
run(
    {
        "run_dir": "runs/glm-lora",
        "resume": "runs/glm-lora/training/rank-{rank:05d}/checkpoints/step_00000002.ssp",
    },
    phase="execute",
)
```

`--steps` is a total completed-update target and must fit the original
`--schedule-steps`. For a learning check, set `--repeat-training-batch` when
building and optionally require `--minimum-relative-loss-drop 0.01`. That check
requires at least four newly executed updates, including when resuming.

Profile execution after preparing the plans with the shared
[saved-Plan benchmark API](../../../../benchmarking/plan_runner/README.md#nsight-systems).
It launches one collector per global rank, masks each collector to its worker's
GPU, and saves reports under a configurable common directory. No GLM-specific
capture path is needed. Capture spans completed updates, with per-rank NVTX ranges. Evaluation and
checkpoint cadences must fall outside the interior of that capture; the CLI
rejects conflicts. Inductor/Triton caches remain in node-local temporary storage.

Small fixture tests establish pipeline behavior; they do not qualify full-size
pretrained 32K training, distributed evaluation/resume on H100s or full-model
throughput.

## Measured eight-H100 training

A full-checkpoint run exercised all 45 text layers at sequence length 32,768,
LoRA rank/alpha 32, one sequence per rank per update, and 96 GiB of pinned-host
spill per rank. Each admitted budget completed eight updates through the public
execution CLI, held-out evaluation and checkpointing on all eight ranks.
The corpus was FineWeb-Edu, tokenized with the checkpoint's tokenizer, with
5,242,522 training tokens and 1,057,080 held-out tokens. The base retains its
original FP8/BF16/FP32 storage. A preceding full-model run
also passed cold resume from update four to six, preserving the schedule, data
position and per-rank adapter/optimizer state.

| Device budget per rank | Predicted update | Measured update | Difference |
| --- | ---: | ---: | ---: |
| 48 GiB | 11.59 s | 11.71 s | +1.11% |
| 64 GiB | 11.45 s | 11.58 s | +1.15% |
| 76 GiB | 11.34 s | 11.51 s | +1.47% |

Measured times are the median of maximum-rank Trainer wall times for updates
two through six; first-update setup and final evaluation/checkpoint writes are
excluded. The `communicate` phase measured task/NCCL overlap before CPU planning.
The same budget comparison with isolated-task profiles underestimated runtime
by 12.8–13.3%. No full-training timing multiplier was fitted.

CPU planning rejected 20, 24 and 32 GiB with these task implementations and
96 GiB of spill; those budgets were not executed. This campaign used the
existing batch-one form and default graph-pair policy. It does not establish
the minimum possible budget with other chunk or retention policies.

All-rank metrics, checkpoint metadata, optimizer counters and bounded tensor
samples passed inspection. These short runs establish pipeline, capacity and
timing behavior; they do not establish convergence or full-tensor gradient
equivalence. Measurements used core `b512b262` and provider `e516a61` on one
eight-H100 node. The calibrated three-budget execution allocation lasted
22m29s, including staging the original checkpoint to local SSD.

### Chunked execution

A separate campaign used the same full model, corpus, 32K sequence length,
LoRA rank/alpha32 and eight-rank geometry with 96 GiB pinned-host spill per rank.
Its search compared six 4K/8K chunk/policy combinations and two cost-input
controls with 32K expert chunks. At both admitted budgets it selected
`attention-input`, with 8192-token projection and expert chunks. The Plans chose
RECOMPUTE for all 45 model checkpoint stages and SAVE for the required final
stage; the policy name does not mean attention state was retained in those
RECOMPUTE stages. Routing choices remain retained by the provider contract.

| Device budget per rank | Predicted update | Measured update | Difference |
| --- | ---: | ---: | ---: |
| 40 GiB | 14.49 s | 14.50 s | +0.12% |
| 48 GiB | 13.43 s | 13.92 s | +3.71% |

Each Plan completed eight real updates, held-out evaluation and checkpointing
on all eight ranks. An independent audit verified the exact Plan identities,
all-rank progress and loss aggregation, optimizer counters and finite bounded
checkpoint samples. Timing uses the same updates-two-through-six clock as the
unchunked table. These checks establish pipeline and timing behavior, not a full
numerical oracle or convergence result.

Training Plan device peaks were 38.29/42.71 GiB and spill peaks were
88.49/87.48 GiB, respectively. Timestamped NVML samples stayed below each device
budget, at 37.53/45.53 GiB per GPU; these include reserved memory and initialization
and are not exact allocation peaks. The shared 980 GiB host allocation recorded
no OOM events. Checkpoints and datasets used node-local SSD; spill used pinned
host memory.

This campaign used core `b4792dcd` and provider `403cb240`; the allocation lasted
15m43s including staging and checkpoint checks. Its winners belong to the
eight-candidate **chunked subset**. The historical unchunked table uses older
sources, so it is not a controlled measurement of chunking's throughput cost.

### Combined chunk and retention search

The expanded campaign adds fully unchunked controls under all three policies to
the same measured candidates. All 11 candidates have their own training and
evaluation Programs, task profiles and paired NCCL calibration. CPU planning
compared 55 candidate/budget cases at 24/28/32/40/48 GiB, using the five resolution
options shown in the example. Fifteen cases admitted; 40 refused. No candidate
admitted at 24 or 28 GiB with 96 GiB pinned host per rank.

| Device budget per rank | Selected execution | Predicted update | Measured update | Difference |
| --- | --- | ---: | ---: | ---: |
| 32 GiB | 4K projection / 4K MoE chunks | 18.77 s | 19.31 s | +2.88% |
| 40 GiB | Unchunked | 12.08 s | 12.46 s | +3.13% |
| 48 GiB | Unchunked | 11.25 s | 11.98 s | +6.48% |

Each selected Plan completed eight FineWeb-Edu updates, held-out evaluation and
checkpointing through `run(config, phase="execute")`, with normal fast defaults.
All use the `attention-input` candidate, but inspection of the saved Plans shows
45 RECOMPUTE model stages and the required final SAVE stage. Attention is replayed
in these winners; the candidate name describes the available SAVE policy.
The unchanged 10% simulator band passes at every budget, using the same median
maximum-rank wall time over updates two through six. At this geometry there is
one sequence per rank, so only microbatch size one and ordering `1x1rp` are legal.

An independent audit checks the minimum admitted prediction, preservation of
previous candidate choices, exact Plan/source identities, all-rank loss and
progress, checkpoint metadata, optimizer counters and finite samples from the
first 4 KiB of each tensor. These samples are not a complete numerical oracle.
Native runtime reports contained no physical-budget or external-headroom overruns. Training
Plan device peaks were 31.73/39.85/47.77 GiB and planned spill peaks were
89.49/88.49/87.48 GiB. Sampled NVML maxima were 30.04/38.03/46.03 GiB; samples are
not exact allocation peaks. Whole-node host usage reached 961.28 GiB of 980 GiB,
including eight pinned pools, process memory and checkpoint page cache, with no
OOM events or hits to the host memory limit.

The shared training/evaluation check uses one compatible measured NCCL startup
reservation when a calibration covers their complete request union. Core
`d732059a` corrects an earlier duplicate reservation; individual task profiles
and costs are unchanged. The campaign uses that runtime and public Python entry
with frozen Programs measured at `b4792dcd`, and provider `403cb240` throughout.
The eight-H100 allocation lasted 22m26s, including 3m59s staging the original
checkpoint and corpus to local SSD. Spill remained pinned host memory.
