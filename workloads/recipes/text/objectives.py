"""Text objectives supplied to the generic runner."""

from __future__ import annotations


def next_token_prediction(model, batch, **options):
    """Sum next-token loss; the batch already supplies shifted target tokens."""
    tokens, targets, lengths = batch
    return model.loss(tokens, targets, seq_lens=lengths, reduction="sum", **options)


def next_token_prediction_with_metrics(model, batch, **options):
    """Next-token loss plus the model's metrics, using already-shifted targets."""
    tokens, targets, lengths = batch
    return model.loss(
        tokens,
        targets,
        seq_lens=lengths,
        reduction="sum",
        return_metrics=True,
        **options,
    )
