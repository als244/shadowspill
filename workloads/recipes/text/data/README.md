# Text data

These helpers prepare rank-local inputs for text recipes. They do not build
models, choose execution plans, or manage ShadowSpill pools.

| Module | Responsibility |
| --- | --- |
| `prepare.py` | Download/tokenize parquet text and write corpus metadata; also used by baseline preparation |
| `prepare_chat.py` | Render conversations through the chat template, tokenize them, and write the mask marking the assistant's tokens beside the token files |
| `tokenizer.py` | Identify local tokenizer files |
| `fixed.py` | Resumable fixed-length `TokenUpdates` with one batch per example |
| `documents.py` | Read tokenized documents and handle oversized documents |
| `packing.py` | Pack whole documents and construct length tensors |
| `packed.py` | `PackedTokens` corpus configuration |
| `source.py` | Packed update iteration and candidate geometry inputs |
| `units.py` | Units of `max_seq_len` tokens divided among ranks: the held-out calls each rank evaluates, padded to its microbatches |

```bash
python -m workloads.recipes.text.data.prepare --help
```

Use `from workloads.recipes.text.data.fixed import TokenUpdates` for fixed
sequences or `from workloads.recipes.text.data.packed import PackedTokens` for
whole-document packing. A directory `prepare_chat.py` wrote carries
`train.mask.bin` and `val.mask.bin`, one byte per token; `PackedTokens` reads
them and only the marked tokens, the assistant's, are targets of the loss, so
instruction tuning never trains on the prompt. Recipe entry points remain
directly under `text/`; the phase-based Llama recipe keeps
build/profile/plan/execute under `text/llama3/`.
