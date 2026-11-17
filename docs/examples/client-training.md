# Prepare data for the training examples

The [homepage examples](../../README.md) use real text corpora, the model's own
tokenizer, and the public client/server API. Prepare data once, then keep the
resulting token files on the client. Training itself needs only NumPy and
`shadowspill-client` on that machine.

## Prepare UltraChat

Both fine-tuning examples use the same UltraChat conversations and assistant-token
masks. Qwen3.5-9B and Qwen3.5-35B-A3B use the same tokenizer. Run this from the
repository root in the server's training environment, which includes tokenizer
and parquet dependencies:

```bash
python -m tools.datasets.prepare_chat \
    --tokenizer Qwen/Qwen3.5-9B \
    --dataset HuggingFaceH4/ultrachat_200k \
    --files data/train_sft-00000-of-00003-a3ecf92756993583.parquet \
    --max-train-tokens 100000000 \
    --out data/ultrachat-qwen35
```

Use an existing local model directory in `--tokenizer` if you already have the
checkpoint. The source is the first shard of UltraChat's
[`train_sft` split](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/main/data).
The tool renders complete conversations with the checkpoint's chat template and
writes token files, assistant masks and metadata. Its first million tokens are
held out for validation. Increase `--shards` and `--max-train-tokens` for more data.

On a separate client, copy the directory to its checkout, for example:

```bash
mkdir -p data
rsync -a training-host:/path/to/shadowspill/data/ultrachat-qwen35 data/
```

The homepage's `ConversationBatches` reads complete, variable-length conversations
through memory maps. It groups them into batches of at most 65,536 model positions
and 1,024 conversations. The 8,192 length setting is an upper bound per conversation,
not a padded or fixed length. Conversations above it are skipped, as are those
without supervised targets; the reader never cuts a conversation or joins two.
Each sequence has one extra original token for next-token shifting.

Each conversation attends independently. The service handles physical packing,
padding and placement; the client checks that a prepared layout can fit each
actual batch before submitting it. Loss is averaged over all assistant targets in
the global batch, irrespective of conversation length. Prompts and padding do not
enter that denominator. The final partial batch is included.

The server loads the selected checkpoint and trains its text decoder or adapters.
The client owns the dataset and never loads model weights. There is no server
dataset directory to configure.

## Eight-GPU data-parallel pretraining

Prepare a separate corpus with the **Qwen3** tokenizer; its token IDs are not
interchangeable with Qwen3.5's:

```bash
python -m tools.datasets.prepare_text \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --dataset HuggingFaceFW/fineweb-edu \
    --files 'sample/10BT/{:03d}_00000.parquet' \
    --max-train-tokens 100000000 \
    --out data/fineweb-qwen3
```

Start `serve.py` once on a host allocated eight GPUs. The service discovers the
assigned devices from its environment and launches workers. For a scheduler, use
its allocation mechanism to obtain that host, then run the service inside the
allocation. Run the client once on your laptop or login host; it does not need
`torchrun` or one process per GPU.

The homepage passes this configuration to `create_training_client`:

| Setting | Value |
| --- | --- |
| Architecture | Qwen3-30B-A3B, initialized from scratch |
| Parallelism | Eight-way data parallelism with sharded parameter storage |
| Sequence length | 4,096 positions |
| Global batch | 512 sequences, or 2,097,152 scheduled tokens |
| Microbatch | 64K tokens per rank; four microbatches per full update |
| Packing allowance | 128 sequences per microbatch by default; 4K maximum length |
| Computation chunks | Unchunked projections and MoE |
| Training precision | BF16 parameters; FP32 gradients, masters and Adam state |
| Execution budget | 20 GiB per worker |
| Spill budget | 96 GiB per worker, 768 GiB in total |

The allocation needs room beyond the execution and spill pools for staging,
workers and profiling. These are requested budgets; inspect `plan.html` for actual
planned occupancy and estimated time on your hardware.

`data_parallel(...)` returns the existing portable parallelism configuration.
Setup infers input fields from the model and objective schemas, including optional
defaulted masks and weights. The setting reaches
preparation before graph capture, profiling or planning. Builds use the declared
parameter ownership and input placement, and each resulting Plan is tied to that
configuration. Changing parallelism requires preparing compatible Plans.

A client call submits the global batch. The service packs it over eight input
partitions and accumulates the selected microbatches before one optimizer update.
The token objective uses a global token mean; the secondary `moe_balance` objective
excludes padding and retains its per-microbatch statistics. See
[objective settings](../python/catalog.md#objectives) for that separate term.

## Use your own token sequences

The same normalization is available without the corpus reader:

```python
from shadowspill_client.text import text_batch

batch = text_batch(token_sequences)
result = training.step(batch, optim={"lr": 0.0003}).result()
```

For supervised fine-tuning, supply `loss_masks=assistant_masks`, with one boolean
mask aligned to each original token sequence. Only unmasked shifted targets enter
the denominator. Padding added later by the service never enters it. The helper
uses one denominator across the whole list, including sequences of different
lengths. Construct the list once before slicing it for several `forward_backward`
calls that belong to one update.

For explicit weighting, use `text_datum(tokens, weights=...)` or construct `Datum`
records yourself. The underlying objective sums weighted token losses; it does
not introduce a second normalization.

Next, see [checkpoints and resume](../python/client.md#save-and-resume), the
[client guide](../python/client.md) or [Advanced usage](../python/advanced/README.md).
