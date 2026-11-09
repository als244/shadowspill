# GLM-5.3-Flash checkpoint LoRA

This recipe uses the same public build, measurement, planning and saved-Plan
execution APIs as other text models. Model-specific files describe checkpoint
precision, LoRA targets and the KDA input metadata. The shared
[text phases](../phases/) perform profiling, communication calibration and
candidate planning; the [recipe driver](../../phases.py) starts fresh workers.

```bash
python -m workloads.recipes.text.train_glm53_flash_lora \
  --data /path/to/tokenized-text --checkpoint /path/to/GLM-5.3-Flash \
  --frozen-state-id exact-base-revision --run-dir runs/glm-lora \
  --ranks 8 --sequence-length 32768 --sequences-per-step 1
```

Run this command once, outside `torchrun`. Build and profiling use one
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

To run phases separately, start the first command with `--phase build`, then:

```bash
python -m workloads.recipes.text.train_glm53_flash_lora --run-dir runs/glm-lora --phase profile
python -m workloads.recipes.text.train_glm53_flash_lora --run-dir runs/glm-lora --phase communicate
python -m workloads.recipes.text.train_glm53_flash_lora --run-dir runs/glm-lora --phase plan
python -m workloads.recipes.text.train_glm53_flash_lora --run-dir runs/glm-lora --phase execute
```

Single-GPU profiling measures host transfers itself; skip `communicate` for
`--ranks 1`. CPU-only `plan` reads saved artifacts and can change
`--execution-gib` and `--spill-gib` without importing Torch or the model.
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

## Explicit chunk and retention candidates

The default is one unchunked build with the ordinary cost/input graph pair.
For an explicit search, add for example:

```bash
--chunk-sizes 4096 8192 --moe-chunk-sizes 0 \
--graph-pair-policies cost-input cost-attention attention-input \
--resolution-options 0 1/2 1
```

Each distinct chunk/retention combination gets its own build, task profiles and
overlap calibration. Planning crosses these with the configured microbatch sizes
and orderings, runs PressureFit for each, and selects the fastest admitted
training/evaluation pair. `search.json` records every result, including refusals.
Keep the list bounded: more candidates increase preparation time. `0` in
`--moe-chunk-sizes` uses the projection chunk size; repeated unchunked combinations
are deduplicated. A new candidate needs a fresh build directory.
The example builds six candidates, including smaller MoE chunks. Add `32768`
to `--moe-chunk-sizes` to compare larger expert chunks, or add `0` to
`--chunk-sizes` to include unchunked controls. For a smaller comparison of chunk
sizes, select one graph-pair policy.
`--resolution-options` allows SAVE/RECOMPUTE mixtures within each candidate;
the historical default `1` permits full recomputation only.

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

The full-model measurements below are **unchunked controls**. Full FP8 chunked
training, its minimum budget and throughput are undergoing separate qualification.

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

```bash
python -m workloads.recipes.text.train_glm53_flash_lora \
  --run-dir runs/glm-lora --phase execute \
  --resume 'runs/glm-lora/training/rank-{rank:05d}/checkpoints/step_00000002.ssp'
```

`--steps` is a total completed-update target and must fit the original
`--schedule-steps`. For a learning check, set `--repeat-training-batch` when
building and optionally require `--minimum-relative-loss-drop 0.01`. That check
requires at least four newly executed updates, including when resuming.

Profile only execution after preparing the plans:

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /path/to/glm-training \
  python -m workloads.recipes.text.train_glm53_flash_lora \
  --run-dir runs/glm-lora --phase execute --nsys-start-step 3 --nsys-steps 2
```

Capture spans completed updates, with per-rank NVTX ranges. Evaluation and
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
