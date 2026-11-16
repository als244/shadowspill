"""Edit client/server paths and resource limits, then run this Python file."""

from pathlib import Path

from examples.training.qwen35_lora.loop import train
from examples.training.qwen35_lora.settings import Settings
from examples.training.qwen35_lora.setup import start
from examples.training.text_data import TokenBatches
from shadowspill_client import SSH, ServiceClient

settings = Settings(
    data="/data/fineweb-qwen35",  # Token corpus on the client.
    checkpoint="/checkpoints/Qwen3.5-35B-A3B",  # Base weights on the server.
    sequence_length=65536,
    sequences_per_step=4,  # Global update: 256K tokens, including scheduled padding.
    microbatch_sizes=(1, 2, 4),
    chunk_sizes=(8192,),
    moe_chunk_sizes=(8192, 32768),
    graph_pair_policies=("cost-input", "cost-attention", "attention-input"),
    resolution_options=("0", "1/4", "1/2", "3/4", "1"),
    tail_save_counts=(0, 1, 2, 4, 8, 16),
    execution_gib=77,
    spill_gib=200,
    steps=100,
    schedule_steps=100,
    eval_every=10,
    checkpoint_every=10,
)


if __name__ == "__main__":
    directory = Path("runs/qwen35-lora")
    directory.mkdir(parents=True)
    source = TokenBatches(
        settings.data,
        settings.sequence_length,
        settings.sequences_per_step,
        normalize=False,
    )
    validation = TokenBatches(
        settings.data,
        settings.sequence_length,
        settings.sequences_per_step,
        split="val",
        normalize=False,
    )
    with ServiceClient(ssh=SSH("chicago", 8000)) as service:
        training, cursor = start(service, settings, source, validation, directory)
        try:
            # This loop accepts an already initialized client; it owns only data,
            # schedules, step/backward calls, evaluation and checkpoint cadence.
            train(training, settings, source, validation, directory, cursor=cursor)
        finally:
            training.close().result()
