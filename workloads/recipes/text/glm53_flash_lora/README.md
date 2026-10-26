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

Artifacts stay separate from compiler caches:

| Path under the run directory | Contents |
| --- | --- |
| `config.json`, `model.json` | Recipe settings, checkpoint/tokenizer identities and tensor inventory |
| `programs/batch-N/` | Training and optional evaluation Programs |
| `profiles/`, `profiled/`, `communication/` | Reusable measurements and fully profiled Programs |
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
  --run-dir runs/glm-lora --phase execute --nsys-start-step 2 --nsys-steps 3
```

Capture spans completed updates, with per-rank NVTX ranges. Evaluation and
checkpoint cadences must fall outside the interior of that capture; the CLI
rejects conflicts. Inductor/Triton caches remain in node-local temporary storage.

Full-size pretrained 32K training and distributed evaluation/resume still need
qualification on the target H100 allocation. Small fixture tests establish
pipeline behavior; they are not full-model throughput results.
