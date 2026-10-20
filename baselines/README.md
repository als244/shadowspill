# Standalone training baselines

This directory holds the training systems ShadowSpill is measured against,
over the same model definitions (`workloads/models`), the same objective,
geometry and precision contract, and one result row schema, so a report can
draw every system's throughput against its memory on one figure. None of
them imports or runs ShadowSpill's planner, trainer, simulator or allocator.
Rows, traces and sweep outputs go under the output directory a sweep or a
harness is given, outside the tracked source files.

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

## Corpora

`python -m baselines.scripts.prepare_corpus --tokenizer <path or id> --shards
'<glob of parquet files>' --output <dir> [--tokens N]` tokenizes a local text
sample once per tokenizer family into `train.bin`, `val.bin` and `meta.json`,
the files `workloads.recipes.text.prepare_data` writes, stopping at the last
whole document within `--tokens`. A harness reads the result with `--data
<dir>`; every model family that shares the tokenizer reads the same corpus,
and a corpus whose vocabulary exceeds the model's is refused.

## Implementations

| Directory | Implementation | Status |
|---|---|---|
| [`moe_layer/`](moe_layer/README.md) | One Qwen MoE router/expert layer: SonicMoE DP1 versus QuackMoE EP8; BF16/FP8 and full/LoRA | 48 H100 forward/backward cases validated; four-plot report and raw-sample audit complete |
| [`fsdp2/`](fsdp2/README.md) | Native PyTorch FSDP2 over any catalog model, full or LoRA, with recompute, both activation offloaders, CPU offload, the torchao scenarios, a corpus and a checkpoint; local experts, no EP | The tiny qualification matrix passes across the five families against the eager reference; multi-rank bring-up pending |
| [`deepspeed/`](deepspeed/README.md) | DeepSpeed ZeRO stages 1 to 3 over any catalog model, full or LoRA, with optimizer and parameter offload, activation checkpointing and CPU checkpointing, a corpus and a checkpoint | Harness and tiny qualification matrix; bring-up pending |
| [`automodel/`](automodel/README.md) | Pinned NVIDIA NeMo AutoModel FSDP2 + EP8, TE/DeepEP | Derived-case generator ready; environment/GPU qualification pending |
| [`megatron_core/`](megatron_core/README.md) | Official Qwen model through NVIDIA Megatron Bridge / Megatron-Core | Pinned model/recipe and topology-aware command generator; environment/GPU qualification pending |
| [`references/`](references/automodel.json) | Unmodified NVIDIA AutoModel YAML, source commit and digest | Preserved for upstream reproduction |

The supplied [NVIDIA YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/b203f3f9f9f3140c660942a088ff36bbc3a5e856/examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml)
uses AutoModel with **FSDP2 + EP8**, Transformer Engine, and DeepEP. It is not
a Megatron-Core recipe. It also enables `fake_balanced_gate`; keep that
distinction explicit in any performance comparison. The original YAML, license
notice, commit, and SHA256 are preserved in `references/`.

## Offload options by harness

Every harness exposes its framework's own offloading, by the framework's
names, as a sweep dimension; nothing is offloaded that the framework does
not offload itself.

| Harness | Parameters, gradients, optimizer state | Optimizer state alone | Activations |
|---|---|---|---|
| `fsdp2/` | `--offload cpu`: FSDP2's `CPUOffloadPolicy` (shards and state on the host, the step on CPU) | `--optimizer cpu-offload`: torchao's `CPUOffloadOptimizer` with the shards on the device | `--activation-offload cpu` (PyTorch's `offload_wrapper`) or `overlapped` (torchtune's stream-overlapped offloader, vendored); both beside `--recompute` |
| `deepspeed/` | `--offload-param cpu\|nvme` (ZeRO-3 parameters between uses) with `--offload-optimizer` | `--offload-optimizer cpu\|nvme` (ZeRO-2 and 3, DeepSpeed's CPU AdamW) | `--activation-offload cpu`: DeepSpeed's CPU checkpointing of recomputed blocks' inputs |
| `megatron_core/` | `--cpu-offload-layers N` with weights (Megatron-Core's layer-level `cpu_offloading`; not beside recomputation or pipeline parallelism) | `--optimizer-cpu-offload` (Megatron-Core's `optimizer_cpu_offload`, a fraction of the state) | `--cpu-offload-layers N` with `--no-cpu-offload-weights` |
| `automodel/` | `--cpu-offload`: FSDP2's `CPUOffloadPolicy` through the recipe's `distributed.offload_policy` | none in AutoModel | none in AutoModel (activation checkpointing only) |
| torchtitan (planned) | `training.enable_cpu_offload`: FSDP2's `CPUOffloadPolicy` | none in torchtitan | none in torchtitan outside its experiments (activation checkpointing only) |
| ShadowSpill (planned) | the execution and spill budgets and the SSD spill of its planner stand in for all three columns | | |

## Measurement contract

- Normalize the local summed objective by tokens per rank per update. FSDP2's
  gradient averaging then produces the global mean exactly once.
- Exclude warmup and profiled steps from throughput results. Keep every rank's
  cycle times and report the median of the slowest rank in each measured step.
- Record actual microbatch size, checkpointed layers, compilation, precision,
  optimizer settings, memory peaks, and failures. Do not silently lower a batch
  size or change a precision when a configuration fails.
- Every harness exposes the precision contract of `baselines/common/precision.py`
  and states what it honors; by default compute is bfloat16 with float32
  masters, gradients and moments, and a request a harness cannot honor is
  refused, never substituted.
- Useful work is counted from the model as built by the convention in
  `baselines/common/models.py`; recomputation, the optimizer and communication
  are not credited.
- A PyTorch allocator cap does not include NCCL or other external allocations.
  Label it as an allocator cap, rather than a certified total GPU-memory limit.
