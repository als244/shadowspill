"""Train a catalog language model on tokenized text; edit paths and budgets."""

from shadowspill_workloads.recipes import Model, train

# Full training from random weights, using the real Llama decoder and token loss.
# Other examples: "olmoe-7b-a1b", "qwen3-1.7b", "qwen35-2b".
model = Model.from_config("llama3-1b")
config = {
    "run_dir": "runs/llama-pretrain",
    "data": "/data/fineweb-llama",  # Tokenized train.bin / val.bin plus meta.json.
    "sequence_length": 2048,
    "sequences_per_step": 4,  # Four sequences per rank and optimizer update.
    "microbatch_sizes": [1, 2, 4],  # Build/profile these alternatives, then select.
    "execution_gib": 24,  # Total allowance per GPU, including overhead.
    "spill_gib": 64,  # Pinned host capacity per rank.
    "steps": 100,  # Schedule horizon defaults to this total.
    "eval_every": 20,
    "checkpoint_every": 20,
}


if __name__ == "__main__":
    # Launches fresh workers for build/profile/search and the selected training Plan.
    # The adapter supplies shifted targets and globally normalized next-token loss.
    train(model, config)
