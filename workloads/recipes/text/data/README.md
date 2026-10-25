# Text data

These helpers prepare rank-local inputs for text recipes. They do not build
models, choose execution plans, or manage ShadowSpill pools.

| Module | Responsibility |
| --- | --- |
| `prepare.py` | Download/tokenize parquet text and write corpus metadata; also used by baseline preparation |
| `tokenizer.py` | Identify local tokenizer files |
| `fixed.py` | Resumable fixed-length `TokenUpdates` with one batch per example |
| `documents.py` | Read tokenized documents and handle oversized documents |
| `packing.py` | Pack whole documents and construct length tensors |
| `packed.py` | `PackedTokens` corpus configuration |
| `source.py` | Packed update iteration and candidate geometry inputs |

```bash
python -m workloads.recipes.text.data.prepare --help
```

Use `from workloads.recipes.text.data.fixed import TokenUpdates` for fixed
sequences or `from workloads.recipes.text.data.packed import PackedTokens` for
whole-document packing. Recipe entry points remain directly under `text/`;
the phase-based Llama recipe keeps build/profile/plan/execute under `text/llama3/`.
