# Standalone training baselines

This directory holds the training systems ShadowSpill is measured against,
and ShadowSpill itself through its public training API, under one geometry
and precision contract and one result row schema, so a report can draw
every system's throughput against its memory on one figure. ShadowSpill,
FSDP2 and DeepSpeed train the catalog's model definitions
(`workloads/models`) with the shared objective; Megatron-Core and AutoModel
train NVIDIA's model of the same architecture, built from its Hugging Face
configuration, with their frameworks' own losses. Every engine's useful
work is counted from the catalog model. The MoE layer benchmark
(`moe_layer/`) stands apart: one layer's router and experts, with its own
sweep and results and no result row. No other engine imports or runs
ShadowSpill's planner, trainer, simulator or allocator. Rows, traces and
sweep outputs go under the output directory a sweep or an engine is given,
outside the tracked source files.

## Environment

Three environments hold the trainers, and each engine the launcher starts
names its own (`ENVIRONMENT` in its settings); the MoE layer benchmark
names none and runs in any environment that holds its kernels
(`moe_layer/README.md`).

`baselines/scripts/setup.sh` creates the `baselines` Conda environment:
ShadowSpill built from this checkout with mlops and its providers (through
`scripts/setup.sh`), then DeepSpeed, torchtitan and torchao at the versions
verified with PyTorch 2.13. The FSDP2 and DeepSpeed trainers run there, and
so do the launcher, the sweeps and the experiments with their reports, which
import the catalog with its implementations, readers and LoRA. The script
checks that ShadowSpill, mlops, DeepSpeed, torchtitan and torchao import
beside PyTorch 2.13, as do the FSDP2, torchao and DeepSpeed pieces the
engines call, and gives the environment its own toolkit home: a tree of
links to the machine's compiler toolkit, with the NCCL headers PyTorch links
in place of the toolkit's, for the DeepSpeed extensions that include them.
ShadowSpill's trainer runs in the `shadowspill` Conda environment, which
`scripts/setup.sh --python <its interpreter>` installs into; without
`--python` that script creates `.venv`.

Megatron-Core's and AutoModel's trainers run in the `nvidia-training`
environment, which `bash baselines/scripts/setup_nvidia_training.sh` builds with
no options on any machine. It is apart from `baselines`, so the qualified
engines' environment does not change: NVIDIA's revisions pin transformers,
cuDNN Frontend and quack at versions mlops's stack does not take, and mlops is
not installed there. It holds Megatron Bridge with its Megatron-Core and NeMo
AutoModel at their pinned revisions, installed editable from checkouts inside
the environment (`<prefix>/src`). The resolution prefers the versions NVIDIA's
own lock files pin at those revisions, Bridge's `uv.lock` and AutoModel's for
what Bridge's lacks or AutoModel's requirements exclude, and departs from one
only where a requirement forces it, except PyTorch 2.13's own pins (triton and
the NVIDIA libraries its wheel names). NCCL is PyTorch's own 2.29.7, as in the
`baselines` and `shadowspill` environments, except where 9.0 or 10.0 is a
target: there the setup also builds Transformer Engine's NCCL expert-parallel
library (Megatron's `ncclep` dispatcher), with NCCL 2.30.7, the release it is
validated against, and DeepEP for Megatron's and AutoModel's `deepep` and
`hybridep` dispatchers, as Bridge's CI image builds it. The compiler toolkit is
the environment's own, NVIDIA's toolkit wheels of the release PyTorch is built
with, laid out inside the environment, so no build takes a toolkit the machine
has; Transformer Engine and APEX are compiled for the architectures of the GPUs
the machine that builds shows, or for the list `--cuda-archs` names (a login
node building for its compute nodes).

Each script records the environment it built inside that environment, in
`<prefix>/environment.lock.json`: the installed packages, the platform and
the GPU of the machine that built it, and for `setup_nvidia_training.sh`
also the toolkit, the architectures, the pinned revisions and the versions
that depart from NVIDIA's locks. The repository keeps no record of a machine's
environment: every `row.json` records its own, its trainer's Python
version and installed packages, NCCL among them, under `provenance`, and
the host and devices under `hardware`.

## Shared layer

`baselines/common/` is what every engine shares and no training system owns:
`settings` (the `Settings` every engine subclasses: the fields every case
takes, their validation, and a case as a mapping, a command line and a
row's description), `geometry` (one optimizer update's shape), `precision`
(the dtype contract mirroring the quickstart's flags, and the refusal of a
combination an engine cannot honor), `models` (the catalog model an engine
builds, full or LoRA, with the shared objective and the useful work a step
performs), `training` (the catalog model with the shared objective as its
forward, which ShadowSpill, FSDP2 and DeepSpeed train, and the helpers
around its construction), `reference` (the tiny-model check of an engine's
updates against a copy compiled whole,
[`PRECISION.md`](PRECISION.md#what-the-tiny-qualification-compares)),
`results` (the result row schema and the measurement contract),
`provenance` (the sources, packages and hardware a row records) and `data`
(synthetic tokens per rank, or windows of a pre-tokenized corpus).

`launch` is the one entry point of every engine but the MoE layer
benchmark, which runs its own sweep: `python -m baselines.common.launch
--engine <name> --gpus <n> --output <dir> <the engine's flags>` validates
and resolves a case in its own interpreter (which imports the catalog),
writes `case.json`, sets the environment variables the case's settings
decide, and starts torchrun with the engine's trainer, `python -m
baselines.<name>.train`, in the engine's environment (`--python` names
another). `runner` is what every trainer shares: it opens the case, serves
the tokens, times each update with one clock, and writes each rank's result
and the row. An engine's `train.py` provides `train(case, clock)` and a
`main()` that hands it to `runner.main`. ShadowSpill's `main` also passes
`standalone=False`, since its trainer imports ShadowSpill, which the runner
refuses in every other engine; `device_first=False`, since its backend
installs its own allocator in PyTorch's place and the trainer probes the
device under it; and its own `failure`, which records a budget the planner
cannot fit as the frontier's boundary.

## Precision

[`PRECISION.md`](PRECISION.md) is the precision contract across the
engines: what the model, master, gradient and optimizer state dtypes and
the roundings mean, where each engine creates, accumulates over
microbatches and reduces across ranks its gradients and at which dtype, and
which combinations each one honors or refuses.

## LoRA

A LoRA case trains adapters on the shared definition's targets
(`workloads/lora`: every linear layer but the head, and the routed experts
of each mixture block) in its framework's own default layout: the
catalog's in ShadowSpill, FSDP2 and DeepSpeed; Megatron Bridge's in
Megatron-Core, one adapter on each fused `linear_qkv` and `linear_fc1`; and
AutoModel's, one on each Hugging Face projection. The layouts hold
different numbers of trainable parameters at the same adapter rank, so
engines are compared by their total trainable parameters, the elements
of every trainable tensor, which each row records from the model its
framework built.

## Sweeps

`python -m baselines.sweep --engine <name> --config <matrix.json> --output <dir>
--gpus <n>` runs a matrix of one engine's cases as independent, resumable
processes on one node. `cases/<case>/attempt-NNN/` holds each attempt's
command, console log and `row.json`; `manifest.json` records the engine, the
sources, the environment and the cases the matrix names, and a resume must
find the same engine, world size, interpreter and environment, though the
matrix and the code may change: a case keeps its name and results by
its settings' digest, and only the cases a matrix adds are attempted;
`README.md` indexes the latest outcome of every case the matrix names. A case that runs out of
device memory is a row on the frontier's boundary and the sweep continues;
any other failure stops it for diagnosis unless `--continue-on-error`. A
resume (`--resume`) attempts every case with no outcome, never attempted or
interrupted, keeps the completed ones, and attempts the failed,
out-of-memory and timed-out ones again only with `--retry-failed`. The
sweep's console names its cases, the settings its matrix varies and the
outcomes so far, then gives a line as each case starts, with those settings,
and a line as it ends, with its outcome and, when it completed, its tokens
per second, update time, GPU peak held and in use, and host peak; a failed
or timed-out case's line names its log. A trainer's own output never
reaches it: each attempt's stdout and stderr, in order, are its
`console.log`. Every attempt starts through the launcher, with `--python`
(by default the engine's environment) as the trainer's interpreter. An engine joins by
exposing `Settings`, `SOURCE_PATHS`, `ENVIRONMENT`, `add_arguments` and
`from_namespace` in `baselines/<name>/settings.py`, and `train(case, clock)`
with the `main()` that hands it to `runner.main` in
`baselines/<name>/train.py`.

## Experiments

An experiment compares the engines on one model, geometry and precision
contract: its folder under `experiments/` holds its spec, `experiment.json`,
with the settings every case shares, one matrix per engine with that
engine's own options, a README stating the scenario, and, once published,
its results. `python -m baselines.experiment --spec <experiment.json>
--output <dir>` runs its engines' sweeps in order, one engine at a time or
all, resuming where it stopped and attempting only the cases a grown matrix
adds, or from scratch with `--overwrite`, and ends with the report and a summary over everything the experiment
holds. [`experiments/README.md`](experiments/README.md) is the guide to
writing, running, summarizing, publishing and tracing an experiment;
[`experiments/GENERATING.md`](experiments/GENERATING.md) describes
`baselines.generate_experiment`, which writes a throughput and memory
experiment's folder from a template.

## Reports

`python -m baselines.report <rows root> [<rows root> ...] --output <dir>` reads
every `row.json` under the roots and draws, per model, training mode,
sequence length, sequences per step, device and world size, which title
them, two figures: global tokens per second against peak device
memory, and against peak host memory. Each completed throughput row is a
point, colored and marked by engine; each engine's Pareto frontier against
device memory is solid and outlined on both figures, and the rest faint. A
table per group lists its points with the settings that vary among that
engine's rows and marks the frontier; out-of-memory rows are listed as the
boundary. Figures are PNG, SVG and PDF,
with `rows.csv` of every point and `index.html` linking them.

## Corpora

`python -m baselines.scripts.prepare_corpus --tokenizer <path or id> --shards
'<glob of parquet files>' --output <dir> [--tokens N]` tokenizes a local text
sample once per tokenizer family into `train.bin`, `val.bin` and `meta.json`,
the files `workloads.recipes.text.prepare_data` writes, stopping at the last
whole document within `--tokens`. An engine reads the result with `--data
<dir>`; every model family that shares the tokenizer reads the same corpus,
and a corpus whose vocabulary exceeds the model's is refused.

## Implementations

| Directory | Implementation | Qualification |
|---|---|---|
| [`moe_layer/`](moe_layer/README.md) | One Qwen MoE router/expert layer: SonicMoE DP1 versus QuackMoE EP8; BF16/FP8 and full/LoRA | Its own sweep, with `--tiny` as the smoke-test geometry: every case checks that its output and gradients are present and finite, the trained gradients float32, and under LoRA that no frozen weight has one |
| [`engines/shadowspill/`](engines/shadowspill/README.md) | ShadowSpill through `Trainer` over the `ShadowSpill` backend, full or LoRA, with the device and spill budgets (pinned host memory) as the configuration, a corpus and a checkpoint | Tiny cases whose every update is compared with the reference copy compiled whole (`PRECISION.md`) |
| [`engines/fsdp2/`](engines/fsdp2/README.md) | Native PyTorch FSDP2, full or LoRA, with recompute, both activation offloaders, CPU offload, the torchao scenarios, a corpus and a checkpoint; local experts, no EP | As ShadowSpill's |
| [`engines/deepspeed/`](engines/deepspeed/README.md) | DeepSpeed ZeRO stages 1 to 3, full or LoRA, with optimizer and parameter offload, activation checkpointing and CPU checkpointing, DeepCompile with its offloads, a corpus and a checkpoint | As ShadowSpill's, with and without DeepCompile |
| [`engines/automodel/`](engines/automodel/README.md) | NVIDIA's model of a catalog architecture through NeMo AutoModel's fine-tuning recipe and its own loop, full or LoRA, from a Hugging Face configuration or a checkpoint, with AutoModel's options under their own names (FSDP2, Megatron-FSDP and DDP, tensor, pipeline, context and expert parallelism, every backend field, its four losses, FP8, compilation) | Tiny cases whose first update is checked against the dtypes the contract's mechanism holds, with no reference copy, since the model is NVIDIA's (`PRECISION.md`); a LoRA case that adapts pretrained weights adapts the tiny checkpoint `python -m baselines.scripts.write_checkpoint` writes |
| [`engines/megatron_core/`](engines/megatron_core/README.md) | NVIDIA's model of a catalog architecture through Megatron Bridge's training loop over Megatron-Core, full or LoRA, from a Hugging Face configuration or a Bridge recipe, with every Megatron option under its own name (parallelism, sharding, recomputation, activation and optimizer offloading, FP8 and FP4, graph capture) | As AutoModel's |
| [`references/`](references/automodel.json) | NVIDIA's AutoModel benchmark YAML, unmodified, with its source commit and digest | Not an engine (below) |

ShadowSpill, FSDP2 and DeepSpeed train any catalog model but those a
quickstart factory builds (`glm53`, `glm53-tiny`, `glm53-flash` and
`glm53-flash-tiny`), which `ModelRequest` refuses, as it does for the
NVIDIA engines. Every engine but the MoE layer benchmark is qualified
by its matrix through the sweep, `python -m baselines.sweep --engine
<name> --config baselines/engines/<name>/configs/qualification.json --output
<dir> --gpus 1`; the table says what its cases check.

The supplied [NVIDIA YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/b203f3f9f9f3140c660942a088ff36bbc3a5e856/examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml)
uses AutoModel with **FSDP2 + EP8**, Transformer Engine, and DeepEP, not
Megatron-Core;
[`engines/automodel/README.md`](engines/automodel/README.md#automodels-benchmark-recipes)
gives it as engine options. It also enables `fake_balanced_gate`, which
spreads the tokens uniformly over the experts in place of the router's
choice, so its throughput is that of perfectly balanced routing. The
original YAML, license notice, commit, and SHA256 are preserved in
`references/`.

## Offload options by engine

Every engine exposes the offloading of the systems it runs as a sweep
dimension, and nothing is offloaded by code of the engine's own: FSDP2's
offloads are PyTorch's and torchao's, and its `overlapped` is torchtune's
offloader, vendored with its license. Each option carries its framework's
name where the framework names it. The table says what each moves; by
default every engine keeps everything on the device, and an empty cell is
that default.

| Engine | Option | Weights the compute uses | Weights the optimizer steps | Gradients | Optimizer state and step | Activations |
|---|---|---|---|---|---|---|
| `shadowspill/` | `--execution-gib G`, `--spill-gib H` | the planner's: each object is kept in the execution pool, or spilled to the pinned spill pool and fetched back before a task needs it, as its plan for the two budgets says | the planner's | the planner's | the planner's; the step on the device | the planner's, or recomputed |
| `fsdp2/` | `--offload-policy cpu` (FSDP2's `CPUOffloadPolicy`) | device: each block's gathered from the host before it runs, freed as resharding says | host: the shards, the float32 masters when the contract keeps them and the weights themselves otherwise | host: reduced into the shards | host; the step on the CPU | device |
| `fsdp2/` | `--optimizer cpu-offload` (torchao's `CPUOffloadOptimizer`) | device | device, and a pinned host copy the optimizer steps and copies back; weights under 4096 elements stay on the device with a device optimizer | device, copied to the host after each backward | host; the step on the CPU | device |
| `fsdp2/` | and `--offload-gradients` (torchao's `offload_gradients`, one microbatch per update) | device | as above | host: each device gradient freed once it is copied | host; the step on the CPU | device |
| `fsdp2/` | `--activation-offload cpu` (PyTorch's `offload_wrapper`) or `overlapped` (torchtune's `OffloadActivations`, vendored) | | | | | host: each block's saved values, copied synchronously or on side streams; beside `--recompute`, a recomputed block's input |
| `deepspeed/` | `--offload-param cpu\|nvme` (ZeRO-3) | host or NVMe between uses, fetched to the device as each module runs | as `--offload-optimizer` says | as `--offload-optimizer` says | as `--offload-optimizer` says | device |
| `deepspeed/` | `--offload-optimizer cpu\|nvme` (ZeRO-2 and 3) | device, unless the parameters are offloaded | host or NVMe: DeepSpeed's master copy | host: the reduced partitions | host or NVMe; the step on the CPU, DeepSpeed's CPU AdamW | device |
| `deepspeed/` | `--activation-offload cpu` (DeepSpeed's CPU checkpointing) | | | | | host: the inputs of recomputed blocks |
| `deepspeed/` | `--deepcompile-offload parameters\|optimizer-states\|activations` (DeepCompile, ZeRO-3) | as DeepCompile's passes place them, within its memory budget (`deepspeed/README.md`) | | | | |
| `megatron_core/` | `--optimizer-cpu-offload --optimizer-offload-fraction F` (Megatron-Core's hybrid device optimizer) | device | the share F of the float32 masters on the host, the rest on the device | device, copied to the host for the host's share | the share F on the host, stepped there with PyTorch's AdamW; the rest on the device | device |
| `megatron_core/` | `--cpu-offloading` (Transformer Engine's) or `--fine-grained-activation-offloading --offload-modules ...` (Megatron-Core's) | | | | | host: whole layers' saved values, or those of the named modules |
| `automodel/` | `--cpu-offload` (FSDP2's `CPUOffloadPolicy`, two or more ranks) | device: each block's gathered from the host before it runs | host: the shards | host | host; the step on the CPU | device |

## Measurement contract

- The shared objective normalizes the local summed loss by tokens per rank
  per update, and FSDP2's gradient averaging then produces the global mean
  exactly once; Megatron-Core and AutoModel train with their frameworks'
  own losses.
- Exclude warmup and profiled steps from throughput results. Keep every rank's
  cycle times and report the median of the slowest rank in each measured step.
- Record actual microbatch size, checkpointed layers, compilation, precision,
  optimizer settings, memory peaks, and failures. Do not silently lower a batch
  size or change a precision when a configuration fails.
- Every engine exposes the precision contract of
  `baselines/common/precision.py` and states what it honors; by default
  compute and gradients are bfloat16 with float32 masters and moments, a
  weight-gradient multiply adding into a bfloat16 running gradient itself
  (fused narrow-gradient accumulation; FSDP2, DeepSpeed and AutoModel, which
  cannot, default to adding after, and FSDP2, which sums gradients into its
  float32 masters, to float32 gradients), and a request an engine cannot
  honor is refused, never substituted.
- Useful work is counted, for every engine, from the catalog model as the
  launcher builds it on `meta`, by the convention in
  `baselines/common/models.py`, so Megatron-Core's and AutoModel's is that
  of the catalog's model of their architecture; recomputation, the
  optimizer and communication are not credited.
- A PyTorch allocator cap does not include NCCL or other external allocations.
  Label it as an allocator cap, rather than a certified total GPU-memory limit.
