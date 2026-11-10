"""Edit the paths below, then run this file from an installed repository checkout."""

from pathlib import Path

from workloads.recipes import LoRA, Model, train

model_path = Path("/path/to/GLM-5.3-Flash")

config = {
    # Real checkpoint and corpus prepared with its tokenizer.
    "frozen_state_id": "exact-checkpoint-revision",
    "data": Path("/path/to/tokenized-fineweb"),
    "run_dir": Path("runs/glm-lora"),
    # One 32K sequence per rank and update; LoRA rank/alpha default to 32.
    "ranks": 8,
    "sequence_length": 32768,
    "sequences_per_step": 1,
    "microbatch_sizes": [1],  # Exact candidate sizes; each divides sequences_per_step.
    "steps": 100,
    "schedule_steps": 100,
    # Per-GPU allowance and per-rank pinned-host spill pool, in GiB.
    "execution_gib": 48,
    "spill_gib": 96,
    # Compare unchunked, 4K and 8K execution under each retention policy.
    # MoE inherits each chunk size; add 32768 to compare larger expert chunks.
    "chunk_sizes": [0, 4096, 8192],
    "moe_chunk_sizes": [0],
    "graph_pair_policies": ["cost-input", "cost-attention", "attention-input"],
    "resolution_options": ["0", "1/4", "1/2", "3/4", "1"],
    # Each tail count crosses every resolution; zero keeps pure even mixtures.
    "tail_save_counts": [0, 1, 2, 4, 8, 16],
    # Evaluate and save adapters, optimizer state and data progress periodically.
    "eval_every": 10,
    "checkpoint_every": 10,
}


if __name__ == "__main__":
    # Build and profile every candidate; measure communication overlap;
    # select the fastest Plan that fits; train through the ordinary Trainer.
    model = Model.from_checkpoint(model_path, lora=LoRA(rank=32, alpha=32))
    train(model, config)

    # To prepare now and train later, use phase="build", "profile",
    # "communicate", "plan" and "execute" separately with the same run_dir.
    # Replan a new budget using saved measurements, without rebuilding:
    # train(model, {"run_dir": config["run_dir"], "execution_gib": 40}, phase="plan")
