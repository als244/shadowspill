"""The case's tokens as an AutoModel dataset.

Every engine trains on the tokens ``baselines.common.runner`` serves: one
synthetic draw per data-parallel replica, or the replica's corpus window of
each update. ``CaseTokens`` serves them as AutoModel's language-model
datasets serve batches, each ``sequences per microbatch x sequence length``
input tokens with their next tokens as labels and their positions, and
AutoModel's own dataloader routes each rank to its replica through the
dataset's ``shard``.
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset

#: The case the trainer runs, and its vocabulary, set before AutoModel's
#: recipe builds its dataloader.
_serving: dict = {}


def serve(case, vocabulary) -> None:
    """Make ``case``'s tokens what ``CaseTokens`` serves."""
    _serving.update(case=case, vocabulary=vocabulary)


class CaseTokens(IterableDataset):
    """``steps`` updates of one data-parallel replica's microbatches."""

    def __init__(self, steps: int, replica: int = 0, replicas: int = 1) -> None:
        super().__init__()
        self.steps, self.replica, self.replicas = steps, replica, replicas

    def shard(self, num_shards: int, index: int) -> CaseTokens:
        """The replica AutoModel's dataloader routes rank ``index`` of
        ``num_shards`` data-parallel ranks to."""
        return CaseTokens(self.steps, replica=index, replicas=num_shards)

    def __iter__(self):
        case = _serving["case"]
        geometry = case.geometry
        tokens = case.data(
            _serving["vocabulary"], replica=self.replica, replicas=self.replicas
        )
        case.record["automodel_replica"] = [self.replica, self.replicas]
        batch, length = geometry.tokens_per_microbatch, geometry.sequence_length
        rows = geometry.sequences_per_microbatch
        positions = torch.arange(length).expand(rows, -1)
        for step in range(self.steps):
            inputs, targets = tokens(step)
            for index in range(geometry.accumulation):
                start = index * batch
                yield {
                    "input_ids": inputs[0, start : start + batch].view(rows, length),
                    "labels": targets[0, start : start + batch].view(rows, length),
                    "position_ids": positions,
                }
