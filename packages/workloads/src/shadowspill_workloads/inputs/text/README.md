# Text data

These helpers prepare data and rank-local inputs for phase benchmarks and
independent references. They do not build
models, choose execution plans, or manage ShadowSpill pools.

| Module | Responsibility |
| --- | --- |
| `corpus.py` | Tokenize parquet text and write corpus metadata |
| `chat.py` | Render and tokenize conversations with assistant-token masks |
| `tokenizer.py` | Identify local tokenizer files |
| `fixed.py` | Resumable fixed-length `TokenUpdates` with one batch per example |
| `calls.py` | `Inputs`: fixed windows as ordinary per-example calls, with the resume position |
| `documents.py` | Read tokenized documents and handle oversized documents |
| `packing.py` | Pack whole documents and construct length tensors |
| `packed.py` | `PackedTokens` corpus configuration |
| `source.py` | Packed update iteration and candidate geometry inputs |
| `units.py` | Units of `max_seq_len` tokens divided among ranks: the held-out calls each rank evaluates, padded to its microbatches |

```bash
python -m tools.datasets.prepare_text --help
```

The command-line programs live in `tools/datasets/prepare_text.py` and
`tools/datasets/prepare_chat.py`; they call these shared helpers.

Use `from shadowspill_workloads.inputs.text.fixed import TokenUpdates` for fixed
sequences or `from shadowspill_workloads.inputs.text.packed import PackedTokens` for
whole-document packing. A directory `prepare_chat.py` wrote carries
`train.mask.bin` and `val.mask.bin`, one byte per token; `PackedTokens` reads
them and only the marked tokens, the assistant's, are targets of the loss, so
instruction tuning never trains on the prompt.

Public training clients submit global `Datum` batches through the shared
[client API](../../../../../../docs/python/client.md). The
[demo collection](../../../../../../examples/demos/README.md) shows client-owned
data iteration and batch submission.
