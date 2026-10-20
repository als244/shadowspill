# Standalone training baselines

This directory contains baselines that reuse workload definitions where useful
and do not import or run ShadowSpill's planner, trainer, simulator, or allocator.
Results, traces, checkouts, environments, and compiler caches belong under
`runs/`, outside the tracked source files.

The current target is Qwen3-30B-A3B full training on eight H100s: 4,096-token
sequences and 2,097,152 global tokens per optimizer update. Every rank sees
different synthetic inputs. No checkpoint download is needed for the FSDP2
throughput baseline. Full-model multi-GPU qualification and performance measurements are still pending.

## Environment

`baselines/scripts/setup.sh` creates the `baselines` Conda environment every
harness here runs in: ShadowSpill built from this checkout with mlops and its
providers (through `scripts/setup.sh`), then DeepSpeed, torchtitan and torchao
at the versions verified with PyTorch 2.13. It checks that every harness entry
point imports and records the installed packages in
`baselines/scripts/environment.lock.json`. The Megatron-Core/Bridge and NeMo
AutoModel recipe harnesses have their own pinned environments.

## Shared layer

`baselines/common/` is what every harness shares and no training system owns:
`geometry` (one optimizer update's shape and the tokens per update of each
hardware tier), `precision` (the dtype contract mirroring the quickstart's
flags, and the refusal of a combination a harness cannot honor), `models` (the
catalog model a harness builds, full or LoRA, with the shared objective and the
useful work a step performs), `results` (the result row schema and the
measurement contract), `provenance` (sources, packages, hardware and the
kernels that ran) and `data` (synthetic tokens per rank, or windows of a
pre-tokenized corpus).

## Sweeps

`python -m baselines.sweep --harness <name> --config <matrix.json> --output <dir>
--gpus <n>` runs a matrix of one harness's cases as independent, resumable
processes on one node. `cases/<case>/attempt-NNN/` holds each attempt's
command, console log and `row.json`; `manifest.json` records the harness, the
sources, the environment and the cases, and a resume must match it;
`README.md` indexes the latest outcome of every case. A case that runs out of
device memory is a row on the frontier's boundary and the sweep continues;
any other failure stops it for diagnosis unless `--continue-on-error`. A
harness joins by exposing `Settings`, `command` and `SOURCE_PATHS` in
`baselines/<name>/settings.py`.

## Reports

`python -m baselines.report <rows root> [<rows root> ...] --output <dir>` reads
every `row.json` under the roots and draws, per model, training mode, device
and world size, global tokens per second against peak device memory and
against peak host memory: one point per completed throughput row, colored by
harness, with each harness's Pareto frontier as a step whose points are
labeled by the settings that vary among that harness's rows; out-of-memory
rows are listed as the boundary. Figures are PNG, SVG and PDF with a table
per figure, `rows.csv` of every point and `index.html` linking them.

## Implementations

| Directory | Implementation | Status |
|---|---|---|
| [`moe_layer/`](moe_layer/README.md) | One Qwen MoE router/expert layer: SonicMoE DP1 versus QuackMoE EP8; BF16/FP8 and full/LoRA | 48 H100 forward/backward cases validated; four-plot report and raw-sample audit complete |
| [`fsdp2/`](fsdp2/README.md) | Native PyTorch FSDP2 over the existing MLOps Qwen model; local experts, no EP | Configurable sweep harness; six single-GPU numerical cases passed, multi-GPU qualification pending |
| [`automodel/`](automodel/README.md) | Pinned NVIDIA NeMo AutoModel FSDP2 + EP8, TE/DeepEP | Derived-case generator ready; environment/GPU qualification pending |
| [`megatron_core/`](megatron_core/README.md) | Official Qwen model through NVIDIA Megatron Bridge / Megatron-Core | Pinned model/recipe and topology-aware command generator; environment/GPU qualification pending |
| [`references/`](references/automodel.json) | Unmodified NVIDIA AutoModel YAML, source commit and digest | Preserved for upstream reproduction |

The supplied [NVIDIA YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/b203f3f9f9f3140c660942a088ff36bbc3a5e856/examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml)
uses AutoModel with **FSDP2 + EP8**, Transformer Engine, and DeepEP. It is not
a Megatron-Core recipe. It also enables `fake_balanced_gate`; keep that
distinction explicit in any performance comparison. The original YAML, license
notice, commit, and SHA256 are preserved in `references/`.

## Measurement contract

- Normalize the local summed objective by tokens per rank per update. FSDP2's
  gradient averaging then produces the global mean exactly once.
- Exclude warmup and profiled steps from throughput results. Keep every rank's
  cycle times and report the median of the slowest rank in each measured step.
- Record actual microbatch size, checkpointed layers, compilation, precision,
  optimizer settings, memory peaks, and failures. Do not silently lower a batch
  size or change a precision when a configuration fails.
- FSDP2 uses FP32 master parameter shards, FP32 reduced/accumulated gradient
  shards, and FP32 AdamW moments, with BF16 compute weights. Native autograd may
  round individual weight-gradient outputs to BF16 before FSDP converts them to
  FP32. This differs from a captured backward that retains FP32 outputs; the
  runner records this distinction rather than claiming identical rounding.
- A PyTorch allocator cap does not include NCCL or other external allocations.
  Label it as an allocator cap, rather than a certified total GPU-memory limit.

Nsight collection covers the 20 and 77 GiB ShadowSpill winners. The single-layer
MoE microbenchmark precedes full-model baseline GPU work.
The previously prepared ShadowSpill EP campaign has been superseded and
its automatic launch has been disabled.
