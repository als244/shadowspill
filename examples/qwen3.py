"""Edit paths and budgets, then run once outside torchrun."""

from shadowspill_workloads.recipes import Model, train

model = Model.from_config("qwen3moe-30b-a3b")
config = {
    "run_dir": "runs/qwen3-pretrain",
    "data": "/data/fineweb-qwen3",
    "ranks": 8,
    "parallelism": "dp",  # Use "ep" for experts distributed over all eight GPUs.
    "sequence_length": 4096,
    "sequences_per_step": 64,  # Per rank: 2,097,152 global tokens per update.
    "microbatch_sizes": [4, 16],
    # Cross saved tails with evenly distributed SAVE/RECOMPUTE choices on CPU.
    "tail_save_counts": [0, 1, 2, 4, 8, 16],
    "execution_gib": 20,
    "spill_gib": 96,  # Per rank: allow additional host RAM beyond the eight pools.
    "steps": 100,
    "schedule_steps": 100,
    "learning_rate": 3e-4,
    "eval_every": 0,  # Enable with a validation corpus when wanted.
    "checkpoint_every": 10,
}


if __name__ == "__main__":
    train(model, config)
