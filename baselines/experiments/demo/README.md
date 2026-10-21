# demo

| Model | Training | Sequence length | Sequences per step | World size | Hardware |
|---|---|---|---|---|---|
| `llama3-1b` | full training | 1,024 tokens | 8 (8,192 tokens) | 1 | NVIDIA GeForce RTX 5090, 32 GiB |

These define the experiment: every case of every engine shares them, and
they title its figures.

A check that the whole experiment path works, small enough to run in
minutes: Llama 3 1B trained in full by ShadowSpill, FSDP2 and DeepSpeed, two
cases per engine. Llama 3 1B is the smallest catalog size whose rows count
as throughput results, so the demo exercises the summary and the report as
a round's experiment does. Run it after a change to the runner, the
summary, the report or an engine, before a long experiment.

## Scenario

| | |
|---|---|
| Model | `llama3-1b`, its `mlops` implementation, full training, initialized weights and synthetic tokens |
| Geometry | 1,024-token sequences, 8,192 tokens per update in two 4,096-token microbatches, 1 GPU |
| Precision | bfloat16 weights stepped in place, bfloat16 gradients and AdamW moments; each engine's rounding and narrow-gradient fusion |
| Run | 1 warmup and 3 measured updates per case; at most 1800 s a case |

## Engines

| Engine | Cases |
|---|---|
| `shadowspill` | execution budgets of 4 and 8 GiB, a 16 GiB spill pool |
| `fsdp2` | without offload, and with FSDP2's CPU offload policy |
| `deepspeed` | ZeRO-3 with the optimizer offloaded, the parameters on the device or offloaded |

## Running it

From the repository's root:

```bash
python -u -m baselines.experiment --spec baselines/experiments/demo/experiment.json --dry-run
python -u -m baselines.experiment --spec baselines/experiments/demo/experiment.json
```

Everything a run writes goes into `output/` in this folder, which git
ignores; `--overwrite` runs it again from scratch.

What to look at:

* the console: every engine under its heading, a start and a result line
  per case, then the report and the summary;
* `output/summary.md`: each engine's outcomes, the fastest case within 4,
  8, 16 and 32 GiB, each engine's frontier and the report's figures;
* a case's own output in
  `output/sweeps/<engine>/cases/<case>/attempt-001/console.log`;
* the same command again, which attempts nothing, and `--engine fsdp2`,
  which runs one engine;
* `python -m baselines.summary baselines/experiments/demo/output --publish <dir>`, which
  writes the summary and its figures with no path of the machine;
* `python -m baselines.trace baselines/experiments/demo/output fsdp2 <case> --steps 1`, which
  traces one case under Nsight Systems.
