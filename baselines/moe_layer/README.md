# MoE layer benchmark

Measure the router and routed expert FFNs with forward, backward, and combined
GPU timings. The GLM-5.3-Flash geometry is **288 experts, top-8 routing, model
width 4096, expert hidden width 2048**. Qwen3-30B-A3B remains available as a
preset. No attention, normalization, loss head, optimizer, checkpoint loading,
or ShadowSpill runtime participates in these measurements.

## GLM comparison on eight H100 GPUs

Run from the repository root, inside an allocation. The driver uses the MLOps
environment; NVIDIA workers use the separate `nvidia-training` environment.

```sh
python -m baselines.moe_layer.sweep \
  --output runs/glm-moe-layer --model glm53-flash \
  --backends sonicmoe deepep hybridep ncclep \
  --distributed-tuning --tokens 2048 4096 8192 16384 32768 65536
python -m baselines.moe_layer.plot runs/glm-moe-layer
```

The default is **eight simultaneous DP replicas** and EP8. Each sample has
barriers before and after its timed region. DP gradient all-reduce is excluded;
`--dp-size 1` measures a single local replica instead. EP dispatch/combine and
their backward communication are included. Replicated-router gradient reduction
is also excluded. Rank `r` receives the same input and upstream gradient in each
backend/precision/training variant, with seeds `seed + 100 + r` and
`seed + 200 + r`. Requests save both seeds and input/gradient fingerprints.
Expert and adapter weights are initialized by global expert ID, independent of
partition, with `seed + 10000 + 16 * expert_id + projection_slot`.

The matrix has 96 cases: four implementations × BF16/FP8 × full/LoRA × six
per-rank token counts. **LoRA rank and alpha both default to 32**. Nonzero B
factors exercise the input-gradient path of a trained adapter. Real top-k then
renormalized softmax routing is used, without load forcing or token dropping.

## Distributed autotuning

Add `--distributed-tuning` to partition Quack candidates across the DP ranks.
Every rank uses rank zero's input activations and output gradient during warmup,
which produces identical routing on replicated weights. The original rank-specific
inputs are restored and fingerprint-checked before timed samples. Each rank
reports its local best; the global winner is broadcast and all candidate scores
are retained in the shared cache. NVIDIA EP workers leave this option disabled.

MLOps defaults to a GPU-time target of **50 ms warmup plus 180 ms timed per
candidate**, with three timing windows. `MLOPS_QUACK_TUNE_WARMUP_MS`,
`MLOPS_QUACK_TUNE_TIME_MS`, and `MLOPS_QUACK_TUNE_SAMPLES` override these values.
Captured graphs contain bounded numbers of calls and replay until the time target
is reached; there is no fixed 50-warmup/200-timed execution floor. A time window
can exceed its target by one replay. Cold compilation and tensor restoration add
to these targets. Existing valid cached winners are reused unless `QUACK_FORCE_CACHE_UPDATE=1`
requests fresh searches. Tuning results persist in `~/.quack/cache` (override
with `QUACK_CACHE_DIR`); keys include tensor metadata and the pruned candidate
list. The default compiled-kernel directory is `/tmp/<user>/quack_cache`;
`QUACK_CACHE_DIR` overrides both Quack caches. Ordinary GEMMs do not
copy operands to host while tuning. Read/write tensors and overlapping
input/output storage are preserved; unknown kernels use conservative backups.

Distributed tuning requires lockstep calls with matching shapes and candidates.
It is opt-in; independent stage-trace workers use the shared-cache tuner instead.

`--sonicmoe-candidate-policy expanded` broadens the SM90 search with M=64
tiles, more M=128 widths, and 1x1 clusters while retaining the original choices.
The default `baseline` already includes M=128, N=96. The experimental
`expanded-k` additionally varies K at selected anchor tiles. Quack's pruning
still removes incompatible configurations. Candidate lists are part of the
disk-cache identity, so an old narrow-pool winner cannot skip a wider search.
Each rank-zero case records candidate counts, timings and winners in
`candidate-audit.jsonl`. These policies do not change SM100 candidates.

## Results and plots

Open `index.html` in the output directory. There are **six four-panel figures**,
each exported as PNG, PDF, and SVG:

- Forward, backward, and forward + backward effective TFLOP/s line plots.
- The same three phases as runtime (ms) grouped bar plots. Each token-count
  group contains one bar per implementation.

Rows are BF16/FP8; columns are full training/LoRA. Labels are `DP (SonicMoE)`,
`EP8 (DeepEP)`, `EP8 (HybridEP)`, and `EP8 (NcclEP)`. The DP series is orange;
the EP series use distinct related blue/teal colors. Titles state the expert
count, top-k, dimensions, and LoRA rank. Missing/failed cases are never zero-valued
measurements. Optional QuackMoE results have their own label/color.

Both plot forms use **the median of the slowest rank's GPU-event interval in
each iteration**. Combined time is measured directly; adding separate forward
and backward medians can be wrong when their slowest ranks differ. Wall-clock
totals are retained in `results.csv`, `results.json`, and raw results. Events
include host launch gaps and GPU communication dependencies inside the sample.
Gradient clearing, boundary barriers, warmup, validation, and file writes are
outside timing. Memory profiling uses an additional warmed, untimed forward/backward pass.
Every rank records exact PyTorch allocated/reserved peaks separately for each
phase, before/after device-wide usage, and an NVML device-memory peak sampled
at 1 ms (a lower bound that may miss short allocations). Device-wide sampling
includes external communication buffers. Sampler errors are recorded explicitly.
No memory instrumentation or extra phase synchronization enters runtime samples.
Finite validation checks bounded views of at most 1,048,576 elements, including
noncontiguous gradients. It never materializes a full gradient-sized mask or
absolute-value temporary. Per-rank `measurements.json` is written before these
checks; failures record their phase and memory state.

By default, cases share workers, imports, CUDA contexts, and the default process
group within each environment. Immutable model weights are reused across token
counts; changing backend, precision, or training mode creates a new model. Each
shape still has its own warmup (5 iterations) and measurement (20 iterations).
`--fresh-workers` restores a separate process/group for each case. A genuine OOM
ends that worker group and restarts remaining cases in clean processes. QuackMoE
rebuilds its fixed-capacity model buffers when the token count changes.

Each case saves its standalone reproduction command, console log, per-rank
requests, every sample, finite-output/gradient checks, and memory peaks. Shared
worker commands and queues live under `workers/`. Requests record process ID,
process-group reuse and model reuse; rank results record setup, warmup,
measurement and validation wall times.
The sweep records source hashes and options in `manifest.json`. `--resume`
requires identical sources/configuration; `--retry-failed` creates a new
attempt. `--tiny` exercises small geometry and never enters performance plots.

## Implementation interface

Each implementation has its own folder under `implementations/`:

| Folder | Implementation | Environment |
|---|---|---|
| `sonicmoe/` | Replicated SonicMoE expert kernels | MLOps |
| `deepep/` | Megatron-Core MoE with DeepEP | nvidia-training |
| `hybridep/` | Megatron-Core MoE with HybridEP | nvidia-training |
| `ncclep/` | Megatron-Core MoE with NCCL EP | nvidia-training |
| `nvidia/` | Shared Megatron-Core/Transformer Engine construction | nvidia-training |
| `quackmoe/` | Optional QuackMoE with MoonEP | MLOps |

`registry.py` lazy-loads backends after device selection. A backend's `build`
returns the interface in `api.py`: `forward`, `backward`, `zero_grad`,
`named_parameters`, `named_gradients`, `close`, and descriptive `metadata`.
The runner owns inputs, synchronization, timing, validation, and output. To add
a backend, implement this contract and register its environment, layout, label,
and colors; the runner and plotter need no backend-specific branches.

NVIDIA implementations share native Megatron-Core routing, dispatch/combination,
and `TEGroupedMLP` expert computation. Their LoRA wrapper uses per-expert TE
grouped A/B linears in BF16 and native fused FP32 `main_grad` accumulation.
Adapters execute outside FP8 autocast; the base and router are frozen in LoRA.

## Precision and useful-work accounting

SonicMoE uses E4M3 current per-row scaling, with independently quantized weight
orientations. NVIDIA uses TE E4M3 per-tensor current scaling and quantized primary
expert parameters, including frozen LoRA experts. These quantization schemes
differ. Requests record the actual backend and quantized expert element count;
FP8 cases fail setup if any base expert parameter remains high precision.
Activation transport and LoRA computation stay BF16. Router weights default to
BF16; `--router-weight-dtype fp32` opts into FP32 storage for every backend.
Logits and routing scores remain FP32. Gradients accumulate in FP32. The native
Megatron router returns a gradient in the weight's storage dtype, which the
harness accumulates into an FP32 buffer; TE expert gradients write directly to
FP32 buffers. SonicMoE uses physical BF16/logical FP32 parameters to preserve
FP32 gradient outputs at the custom-operator boundary. FP8 weight initialization/quantization is outside timing; activation
quantization is timed. No optimizer/master-weight update is measured.

Useful FLOPs count FMA as two. For tokens/rank T, width D, expert hidden H,
expert count E, top-K, and LoRA rank r, let P = 3TKDH, R = TDE, and
A = TKr(2D + 3H). Full training has forward 2(P + R), backward 4(P + R).
LoRA has forward 2(P + R + A), backward 2(P + R) + 4A. Thus:

- Full combined: `18TKDH + 6TDE`.
- LoRA combined: `12TKDH + 4TDE + 6TKr(2D + 3H)`.

Padding, redundant internal work, elementwise functions, routing, and
communication are excluded from the FLOP numerator but remain in elapsed time.
Divide per-rank FLOPs by seconds and 1e12; do not multiply by EP group size again.

Useful options: `--model`, shape overrides `--width/--hidden/--experts/--top-k`,
`--dp-size` (8), `--ep-size` (8), `--lora-rank` (32), `--lora-alpha` (32),
`--nvidia-python`, `--warmup` (5), `--iterations` (20), `--timeout` (1800),
`--fresh-workers`, `--dry-run`, `--tiny`, `--resume`, and `--retry-failed`. The legacy default model
is Qwen and the default backend subset remains SonicMoE/QuackMoE; select the GLM
preset and four requested backends explicitly as above.
