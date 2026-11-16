# Qwen3 MoE full training

Use the [public client application](../../../../../../../examples/training/qwen3.md)
and its [Python example](../../../../../../../examples/qwen3.py). Setup is separate
from the loop submitting `step()` or `forward_backward()` calls. The former
Qwen-specific recipe adapter and phase CLI are removed; the installed engine
still provides build/profile/plan/execute and generic parallelism interfaces.

The measurements below are preserved historical evidence from the earlier
trainer at the stated revisions. They do not qualify the new public service path.

## Eight-H100 budget sweep

The November 10–11 campaign completed seven updates at every budget with
4,096-token sequences, 64 sequences per rank and 2,097,152 global tokens per
update. Both layouts used real tokenized FineWeb-Edu data, random host
initialization, BF16 compute weights, FP32 gradients/master weights/Adam moments,
and a 96 GiB maximum pinned spill pool per rank. Evaluation and checkpoint I/O
were disabled for this throughput protocol. Measured time is the median of
the slowest rank's `Trainer.step` time over updates 2–7.

| Device budget per rank | DP8 predicted | DP8 measured | EP8 predicted | EP8 measured |
| --- | ---: | ---: | ---: | ---: |
| 20 GiB | 18.313 s | 18.712 s | 23.105 s | 25.390 s |
| 40 GiB | 17.761 s | 18.556 s | 22.964 s | 25.183 s |
| 77 GiB | 17.619 s | 18.181 s | 22.500 s | 24.540 s |

DP searched microbatch sizes `[8, 16, 32]`, projection chunks `[0, 1024]` and all
three graph-pair policies: 18 built/profiled Programs. EP searched the same
microbatch and policy lists with unchunked execution: nine Programs. Both
searched factor orderings and crossed tail counts `[0, 1, 2, 4, 8, 16]` with
resolution options `["0", "1/4", "1/2", "3/4", "1"]`. All winners selected
`cost-input` and unchunked execution. DP selected batch 16 / `1x4rp` at 20 GiB
and batch 32 / `1x2rp` at 40 and 77 GiB. EP selected batch 16 at every budget,
with `2x2rp` at 20/40 GiB and `4x1rp` at 77 GiB.

The DP record uses engine `17c48169`; EP uses `af404765`; both use MLOps
`403cb240`. These are results at the recorded revisions, not measurements of
every later change. All ranks completed their updates with finite losses,
consistent global loss reductions and admitted spill capacities. These short
throughput runs do not establish convergence or a full tensor-reference check.
EP measured about 9–10% above prediction. A subsequent seven-update replay of
the 20 GiB EP Plan measured 25.292 s versus 23.105 s predicted, with two Nsight
capture steps on every rank and a separate final runtime trace. It used engine
`cd197853` and the same provider `403cb240`.

The final runtime trace places about 2.025 s of rank 0's prediction error inside
task durations and only 0.015 s in additional inter-task idle. EP dispatch and
combine kernels are substantial parts of those tasks, but this does not identify
a particular kernel or contention mechanism as the cause of the prediction gap.
Additional EP contention calibration remains deferred. Capture start/stop and
report conversion add considerable time outside steady updates; they are
excluded from the throughput numbers above.

## Earlier 20 GiB comparison

The November 9, 2026 full-model check used FineWeb-Edu text encoded with the
official Qwen tokenizer, 4,096-token sequences, 64 sequences per rank per update,
and **20 GiB execution / 96 GiB pinned host spill per rank**. All 27 batch/order
combinations were considered; 25 were admitted. Both batch-32 choices exceeded
the required input/output capacity. The lowest predicted duration selected
batch 16 with `1x4rp` ordering.

| Same-node comparison | Predicted update | Measured update |
| --- | ---: | ---: |
| Original `2x2rp` Plan without paired rates | 18.163 s | 19.568 s |
| Recalibrated `2x2rp` Plan | 19.934 s | 19.416 s |
| Recalibrated `1x4rp` Plan | 19.344 s | 18.705 s |
| Selected `1x4rp` through the normal public CLI | 19.344 s | **18.732 s** |

Each case completed seven updates. Measured time is the median maximum-rank
Trainer wall time over updates 2–6, excluding startup, final tracing and
checkpoint I/O. The public winner reached approximately **111,956 tokens/s**;
measured time was 3.2% below prediction. Calibration used independent paired
task/collective measurements and refreshed communication measurements, with no
correction fitted to the training step. All eight public-run checkpoints passed
metadata, optimizer-counter and finite-state sample checks.

This search used the default graph-pair policy and no explicit axis-chunk policy.
It searched batch/order choices and PressureFit's placement/variant decisions.
The current shared runner also accepts independent `chunk_sizes`,
`moe_chunk_sizes` and `graph_pair_policies` lists for DP; see its
[search contract](../../configuration.md#search-and-saved-phases). The table above does
not qualify those additional candidates.

The eight host pools alone require 768 GiB. The test used a 980 GiB node host
limit to allow process memory and checkpoint page cache; observed whole-job host
peak was 943.93 GiB. The public execution phase, including initialization and
writing its full checkpoint to local SSD, took 595 s. Checkpoint throughput is
separate from steady training throughput.
