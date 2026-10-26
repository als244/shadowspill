"""Rank-local GLM calls, with KDA metadata and one global token normalizer."""

from workloads.recipes.text.data.calls import Inputs as TextInputs
from workloads.recipes.text.glm import GLMTokenUpdates, next_token_prediction


class Inputs(TextInputs):
    data_type = GLMTokenUpdates

    def __init__(self, options, *, split="train", **kwargs):
        super().__init__(
            options,
            split=split,
            repeat_first_update=split == "train" and options["repeat_training_batch"],
            **kwargs,
        )

    def arguments(self, update):
        calls = tuple(
            self.data.microbatches(update, sequences_per_microbatch=self.size)
        )
        return {
            "args": [batch for batch, _ in calls],
            "kwargs": [{"normalizer": scale} for _, scale in calls],
        }


def loss(model, *batch, normalizer, head_chunk_size):
    return (
        next_token_prediction(model, batch, head_chunk_size=head_chunk_size)
        * normalizer
    )
