"""Edit the paths and resource limits, then run this Python file."""

from pathlib import Path

from workloads.recipes import LoRA, Model, train

model_path = Path("/checkpoints/Qwen3.5-35B-A3B")
config = {
    "run_dir": Path("runs/qwen35-lora"),
    "data": Path("/data/fineweb-qwen35"),
    "frozen_state_id": "exact-base-checkpoint-revision",
    # Four independent 64K contexts per update, regardless of microbatch choice.
    "sequence_length": 65536,
    "sequences_per_step": 4,
    "microbatch_sizes": [1, 2, 4],
    # Build/profile each combination; select the fastest admitted Plan.
    "chunk_sizes": [8192],
    "moe_chunk_sizes": [8192, 32768],
    "graph_pair_policies": ["cost-input", "cost-attention", "attention-input"],
    "resolution_options": ["0", "1/4", "1/2", "3/4", "1"],
    "execution_gib": 77,
    "spill_gib": 200,  # Pinned host memory; reserve extra RAM for the process.
    "steps": 100,
    "schedule_steps": 100,
    "eval_every": 10,
    "checkpoint_every": 10,
}


if __name__ == "__main__":
    model = Model.from_checkpoint(model_path, lora=LoRA(rank=32, alpha=32))
    train(model, config)
    # Each saved phase can also run separately:
    # train(model, {"run_dir": config["run_dir"]}, phase="execute")
    # Change a budget with phase="plan", then execute the new selected Plan.
