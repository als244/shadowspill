"""Edit client/server paths and budgets, then run once outside torchrun."""

from pathlib import Path

from examples.training.qwen.loop import train
from examples.training.qwen.settings import qwen3_settings
from examples.training.qwen.setup import start
from examples.training.text_data import TokenBatches
from shadowspill_client import SSH, ServiceClient

settings = qwen3_settings(
    data="/data/fineweb-qwen3",  # Token corpus on the client.
    preset="30b-a3b",  # Random weights initialized on the server.
    ranks=8,
    parallelism="dp",  # "ep" distributes expert weights over the eight workers.
    sequence_length=4096,
    sequences_per_step=512,  # Global: 2,097,152 tokens per update.
    microbatch_sizes=(4, 16),  # Sequences per worker microbatch.
    tail_save_counts=(0, 1, 2, 4, 8, 16),
    execution_gib=20,
    spill_gib=96,  # Per worker; allow additional host RAM outside the pools.
    steps=100,
    schedule_steps=100,
    learning_rate=3e-4,
    eval_every=0,
    checkpoint_every=10,
)


if __name__ == "__main__":
    directory = Path("runs/qwen3-pretrain")
    directory.mkdir(parents=True)
    source = TokenBatches(
        settings.data,
        settings.sequence_length,
        settings.sequences_per_step,
        normalize=False,
    )
    with ServiceClient(ssh=SSH("chicago", 8000)) as service:
        training, cursor = start(service, settings, source, None, directory)
        try:
            train(training, settings, source, None, directory, cursor=cursor)
        finally:
            training.close().result()
