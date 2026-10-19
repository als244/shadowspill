# Qwen MoE layer microbenchmark

Benchmark the **router and expert FFNs** from one Qwen3-30B-A3B layer:
hidden width 2048, expert width 768, 128 experts, top-8 routing. No attention,
normalization, loss head, optimizer, checkpoint loading, or ShadowSpill runtime.

The default sweep runs all SonicMoE cases in one process on a single device
first, then QuackMoE with eight-way expert parallelism: 4K, 8K, 16K, 32K, 64K,
and 128K tokens **per rank**; BF16 and FP8; full training and rank-16 LoRA. This
produces 48 measurements and four plots, with one DP and one EP curve per plot.
DP here means local compute on one replicated model; its inter-rank gradient
synchronization is excluded.

## Run and reproduce

Use an environment containing the MLOps SonicMoE, Quack and MoonEP dependencies.
Launch from the repository root inside an allocated node with eight H100 GPUs.
The launcher selects each worker's device before importing the kernel packages.

```sh
python -m baselines.moe_layer.sweep --output runs/moe-layer
python -m baselines.moe_layer.plot runs/moe-layer
```

Open `runs/moe-layer/index.html`. The four figures have PNG, PDF and SVG forms;
`figures/moe-layer-four-panel.{png,pdf,svg}` combines them into a 2×2 figure
with shared axes and legend. BF16/FP8 are rows; full training/LoRA are columns.
`results.csv` and `results.json` retain their values. The README contains a
linked table of all cases, including failures. Replotting needs only Python,
Matplotlib and the result files, and can run on a different machine.
Plot headings identify the layer's expert count, top-k, model dimension, and
expert FFN dimension from the saved requests. Precision and training mode are
shown separately; model names are not used as plot titles.

Each case has a fresh process (or torchrun process group), `command.json`,
console log, per-rank requests, every timing sample, validation and memory peaks.
`manifest.json` records source hashes and all sweep options. Use `--resume` only
with identical source/configuration; failed attempts require `--retry-failed`.
Retries create new attempt directories. Different settings require a new root.
All result directories should remain below ignored `runs/`.

| Argument | Default | Purpose |
|---|---|---|
| `--tokens` | `4096 8192 16384 32768 65536 131072` | Token counts per rank |
| `--ep-size` | `8` | QuackMoE group size; must divide 128 |
| `--backends` | `sonicmoe quackmoe` | Execution order or backend subset |
| `--precisions` | `bf16 fp8_current` | Expert compute precision |
| `--training` | `full lora` | Train expert/router weights, or only adapters |
| `--warmup`, `--iterations` | `5`, `20` | Untimed warmup count and measured samples |
| `--lora-rank` | `16` | Adapter rank; alpha equals rank |
| `--cpu-threads` | `4` | CPU threads per process |
| `--timeout` | `1800` | Maximum seconds per case; kills the whole process group |
| `--seed` | `7301` | Reproducible initialization and rank-distinct inputs |
| `--tiny` | off | Small smoke-test geometry; never reported as performance |
| `--dry-run` | off | Record expanded commands without launching GPUs |
| `--resume`, `--retry-failed` | off | Reuse validated cases / retry failed cases |

The worker and plotter expose `--help` too. `plot --dpi` controls raster export
resolution (default 240); PDF and SVG remain vector graphics.

## Precision and timing

FP8 means E4M3 current per-row scaling, with independently quantized weight
orientations. Weight quantization occurs at setup, as for published compute
weights; activation quantization is timed. EP activation transport remains
BF16 in both precision modes. LoRA factors use BF16 compute, alpha/rank = 1,
FP32 accumulated gradients, and nonzero B factors to exercise the trained path.
Base weights and router are frozen under LoRA. Full mode computes their
gradients. SonicMoE's native autograd BF16 casts can round intermediate weight
gradients before FP32 accumulation; this is recorded alongside the results.

Real top-k routing is used, without artificial load balancing or token dropping.
Both implementations use the same router initialization and input distribution.
The residual supplied to the SonicMoE facade is zero; its auxiliary loss is
unused. QuackMoE's default 32 communication SMs and overlap are retained.

Each sample includes forward and backward, routing, allocation/launch overhead,
and EP dispatch/combine. Gradient clearing, synchronization before the sample,
initialization, JIT warmup, validation and file writes are outside it. No
optimizer or per-step replicated-router gradient reduction is timed. EP's
expert gradients already represent all ranks' dispatched tokens.

The plotted denominator is the **median of the slowest rank's wall time in each
iteration**. GPU-event forward, backward and total durations are saved separately.
Memory peaks are PyTorch allocator measurements; MoonEP and other external
allocations may not appear in them. Missing/nonfinite gradients fail the case.

Useful FLOPs count an FMA as two. With tokens per rank $T$, width $D$, expert
width $H$, experts $E$, top-$K$ routing and LoRA rank $r$:

- Full forward + backward: $18TKDH + 6TDE$.
- LoRA forward + backward: $12TKDH + 4TDE + 6TKr(2D + 3H)$.

The numerator excludes padding, internal recomputation, elementwise routing,
activation math, and communication. Their cost remains in the denominator.
Divide these per-rank FLOPs by seconds and $10^{12}$ for TFLOP/s per GPU;
do not multiply the EP numerator by the group size again.
