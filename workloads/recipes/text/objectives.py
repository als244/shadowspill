"""Text objectives supplied to the generic runner."""

from __future__ import annotations


def model_loss(model, batch, **options):
    tokens, targets, lengths = batch
    return model.loss(tokens, targets, seq_lens=lengths, reduction="sum", **options)


def model_loss_with_metrics(model, batch, **options):
    tokens, targets, lengths = batch
    return model.loss(
        tokens,
        targets,
        seq_lens=lengths,
        reduction="sum",
        return_metrics=True,
        **options,
    )
