"""Real next-token data and packed-sequence inputs for GLM text models."""

import math
from itertools import accumulate
from statistics import fmean

import torch

from .data.fixed import TokenUpdates


class GLMTokenUpdates(TokenUpdates):
    """Use the text corpus/resume contract with GLM's explicit KDA metadata."""

    def __init__(self, *args, repeat_first_update=False, **kwargs):
        self.repeat_first_update = repeat_first_update
        super().__init__(*args, **kwargs)

    def _update(self, index):
        return super()._update(0 if self.repeat_first_update else index)

    def state_dict(self):
        return {
            **super().state_dict(),
            "repeat_first_update": self.repeat_first_update,
        }

    def load_state_dict(self, state):
        if state.get("repeat_first_update", False) != self.repeat_first_update:
            raise ValueError("checkpoint fixed-batch policy differs")
        super().load_state_dict(state)

    def microbatches(self, update, *, sequences_per_microbatch):
        for batch, scale in super().microbatches(
            update, sequences_per_microbatch=sequences_per_microbatch
        ):
            tokens, targets, lengths = batch
            boundaries = (0, *accumulate(lengths))
            cumulative = torch.tensor(boundaries, dtype=torch.int64)
            chunks = torch.tensor(
                [
                    (sequence, chunk)
                    for sequence, length in enumerate(lengths)
                    for chunk in range((length + 63) // 64)
                ],
                dtype=torch.int64,
            )
            yield (
                (tokens.flatten(), targets.flatten(), boundaries, cumulative, chunks),
                scale,
            )


def next_token_prediction(model, batch, *, head_chunk_size=1024):
    """Sum shifted-target loss; the splitter supplies global token normalization."""
    tokens, targets, boundaries, cumulative, chunks = batch
    return model.loss(
        tokens,
        targets,
        boundaries,
        cumulative,
        chunks,
        chunk_size=head_chunk_size,
        reduction="sum",
    )


def fixed_batch_learning_report(losses, *, minimum_relative_drop):
    """Compare the first token-mean loss with the last three on identical data."""
    if not 0 <= minimum_relative_drop < 1:
        raise ValueError("minimum_relative_drop must lie in [0, 1)")
    if len(losses) < 4:
        raise ValueError("fixed-batch learning check requires at least four updates")
    if not all(math.isfinite(loss) and loss >= 0 for loss in losses) or losses[0] <= 0:
        raise ValueError("fixed-batch losses must be finite and initially positive")
    final_mean = fmean(losses[-3:])
    relative_drop = (losses[0] - final_mean) / losses[0]
    return {
        "global_mean_losses": list(losses),
        "initial_loss": losses[0],
        "final_three_mean_loss": final_mean,
        "relative_drop": relative_drop,
        "minimum_relative_drop": minimum_relative_drop,
        "passed": relative_drop >= minimum_relative_drop,
    }
