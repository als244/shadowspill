# Generating an experiment

`python -m baselines.generate_experiment` writes a throughput and memory
experiment's folder from a template: its spec, one matrix per engine, a
README and a record of how it was made, every value written out. The folder
it writes is an experiment like any other (`README.md`), edited by hand
afterwards as it needs; an experiment can as well be written without it.

## A command

```bash
python -m baselines.generate_experiment lora_rtx5090/qwen35-27b_4K_8seqs \
  --template lora --model qwen35-27b --checkpoint Qwen3.6-27B --corpus qwen35 \
  --sequence-length 4096 --tokens-per-step 32768 \
  --device "NVIDIA GeForce RTX 5090" --gpu-memory-gib 32 --host-memory-gib 188 \
  --description "LoRA fine-tuning of Qwen3.6 27B from its checkpoint on one 32 GiB GPU" \
  [--variable models=<dir> --variable corpora=<dir>]
```

| Input | What it decides |
|---|---|
| `PATH` | The folder, any path under `baselines/experiments`, by the convention `<kind>_<hardware>/<model>_<sequence length>K_<sequences per update>seqs`; a folder that exists is never written over |
| `--template NAME` | A folder of `templates/`: `full` (full training) or `lora` (LoRA from a checkpoint) |
| `--model`, `--implementation` | The catalog model and its implementation (`mlops`) |
| `--lora JSON` | `configure_lora`'s options, for the `lora` template (rank 16, alpha 16) |
| `--checkpoint NAME`, `--corpus NAME` | The checkpoint's directory under the `{models}` variable and the corpus's under `{corpora}`; left out, the experiment trains initialized weights on synthetic tokens |
| `--sequence-length`, `--tokens-per-step`, `--gpus` | The geometry, and the world size every case runs at (1) |
| `--device` | The GPU's name as its driver reports it, `NVIDIA GeForce RTX 5090`, for the README |
| `--gpu-memory-gib`, `--host-memory-gib` | The machine the experiment is sized for |
| `--setting NAME=JSON` | Replaces a shared setting after the template is filled; repeatable |
| `--description TEXT` | What the experiment is for, at the top of its README |
| `--variable NAME=VALUE` | The variables' values for the check below, so that the checkpoint and corpus are read |

## Templates

A template is a folder of `templates/` holding a spec and each engine's
matrix in which a string `"@name"` stands for a value the inputs decide:

| Placeholder | Value |
|---|---|
| `@model`, `@implementation`, `@lora`, `@gpus`, `@sequence_length`, `@tokens_per_step` | As given |
| `@checkpoint`, `@corpus` | `{models}/<checkpoint>` and `{corpora}/<corpus>`, left out when not given |
| `@variables` | The variables those need, with what each value is, left out when none |
| `@microbatches` | Tokens per microbatch from one sequence up to one rank's whole update, in powers of two |
| `@accumulated_microbatches`, `@single_microbatch` | Those with more than one microbatch per update, and the one with a single microbatch |
| `@recompute`, `@recomputed` | 0 to 1 in quarters, and the same without 0 |
| `@execution_budgets` | A quarter, three eighths, a half, three quarters and seven eighths of the GPU's memory, in whole GiB |
| `@summary_budgets` | Those and the whole GPU |
| `@spill_gib` | The largest multiple of 16 GiB within four fifths of the host's memory |

A key whose value is left out is removed, and a grid an empty value leaves
without a case is dropped: with one microbatch per update, the grids of
`@accumulated_microbatches` go.

`full` and `lora` share their ShadowSpill and FSDP2 matrices:

* ShadowSpill: every microbatch size at every execution budget, the spill
  pool at `@spill_gib`;
* FSDP2, with FSDP2's CPU offload policy in every case: every microbatch
  size, recompute fraction and activation offload (PyTorch's and torchtune's);
  and, at full recompute without activation offload, prefetch depth 0
  (FSDP2's own), 1, 2 and 4 in both directions crossed with
  reshard-after-forward, reshard-after-backward and gradient sync, the last
  two left out at a single microbatch, where they have no effect.

Their DeepSpeed matrices differ. `full` runs ZeRO-3 with the optimizer
offloaded and the parameters on the device or offloaded; `lora` crosses
the parameter offload with the optimizer offload. Each runs recompute in
quarters, with and without CPU checkpointing.

A new template is a new folder under `templates/` with an `experiment.json`
and the matrices its `sweeps` name.

## What it writes

Before anything is written, the experiment is checked as its dry run would
check it: every engine's matrix is expanded and every case validated, and a
case an engine refuses stops the generator with the reason. The checkpoint
and corpus are read only when `--variable` gives the variables' values;
otherwise the dry run reads them. Then the folder holds:

* `experiment.json` and one `<engine>.json` per engine;
* `README.md`: first the table of what defines the experiment (model,
  training mode, sequence length, sequences per step, world size,
  hardware), then the
  description, the scenario (model, data, geometry,
  contract, run, the machine it is sized for, the variables), each engine's
  matrix grid by grid with its case count, how to run it with the common
  runner, and where its results go;
* `generation.json`: the template, every input, the derived values, each
  engine's case count, when it was generated and the commit the generator
  ran from. The machine's paths given with `--variable` are not recorded.

The folder is the experiment from then on: the generator never writes over
it, and a template's later change reaches only the experiments generated
after it.
