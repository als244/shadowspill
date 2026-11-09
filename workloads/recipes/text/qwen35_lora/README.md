# Qwen3.5 MoE checkpoint LoRA

`workloads.recipes.text.train_qwen35_lora` runs the saved
build/profile/plan/execute phases on a local base checkpoint and tokenized corpus.
The recipe defaults to LoRA rank/alpha 32, 32K tokens, eight sequences per update,
28 GiB execution memory and 144 GiB pinned host spill.

`--microbatch-sizes 1 2 4 8` specifies the default candidate list explicitly.
Choose a smaller list to bound preparation work: with
`--sequences-per-step 8 --microbatch-sizes 2 4`, only sizes two and four get
training and evaluation Programs and profiles. Each size must be a positive
integer dividing the sequences per update. Repeated sizes are built once.
Changing the step size requires a matching valid list; invalid geometry fails
before workers start.

CPU planning tries all factor orderings for each built size and selects the
lowest predicted training time among jointly admitted training/evaluation Plans.
Microbatch size does not change the total update batch. The selected Plan and
input geometry are passed to `Trainer` for execution.

Start a fresh run with `--data`, `--checkpoint` and `--frozen-state-id`. Later
phases need only `--run-dir` and `--phase`; they reuse `config.json`, including
its exact microbatch list. Change memory budgets during `--phase plan` to reuse
the saved measurements. Changing the microbatch list requires a fresh build.
