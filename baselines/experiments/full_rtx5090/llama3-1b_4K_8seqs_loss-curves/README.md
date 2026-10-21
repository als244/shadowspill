# llama3-1b_4K_8seqs_loss-curves

Loss curves from a real checkpoint: Llama-3.2-1B fine-tuned for 50 updates on
its corpus by ShadowSpill, FSDP2 and DeepSpeed under one contract, float32
masters, gradients and AdamW moments, so that the three engines' losses can be
compared update by update. It is not a throughput experiment: each engine runs
one case, and the summary's budgets are not set. It is written by hand, since
`baselines/generate_experiment.py` makes throughput and memory experiments; run
it with the common runner as [the guide](../../README.md) describes.

## Scenario

| | |
|---|---|
| Model | `llama3-1b`, its `mlops` implementation, full training |
| Weights and data | `{models}/Llama-3.2-1B`, read as a Hugging Face checkpoint; the corpus `{corpora}/llama3`; seed 0 |
| Geometry | 4,096-token sequences, 32,768 tokens per update in 4,096-token microbatches, 1 GPU |
| Precision | weights bfloat16 with float32 masters, float32 gradients and AdamW moments; rounding to nearest; no narrow-gradient fusion; learning rate 1e-5, weight decay 0.01 |
| Run | 1 warmup and 50 measured updates per case; 8 CPU threads; at most 3600 s a case |
| Variables | `{models}`: the directory holding the Hugging Face checkpoints, one directory per checkpoint; `{corpora}`: the directory holding the tokenized corpora prepare_corpus writes, one directory per tokenizer family |

## Engines

| Engine | Case |
|---|---|
| `shadowspill` | `execution_gib` 16, `spill_gib` 32 |
| `fsdp2` | `recompute` 1 |
| `deepspeed` | `zero_stage` 1, `recompute` 1 |

## Running it

From the root of a clone of the committed tree, the dry run first:

```bash
python -u -m baselines.experiment \
  --spec baselines/experiments/full_rtx5090/llama3-1b_4K_8seqs_loss-curves/experiment.json \
  --output <results>/llama3-1b_4K_8seqs_loss-curves \
  --variable models=<dir> --variable corpora=<dir> --dry-run
```

then the same command without `--dry-run`. Every update's loss is in each
case's `rank-NNNNN/result.json` (`losses`), and the row's measurement holds
every rank's.
