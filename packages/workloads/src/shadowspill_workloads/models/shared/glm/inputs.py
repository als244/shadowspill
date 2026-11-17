"""Packed text inputs for backbones consuming explicit sequence/chunk tensors."""

import mlops
import torch


def packed_inputs(tokens, sequence_lengths=None):
    """Prepare one device-owned boundary/chunk description for every layer.

    Public adapters supply zero-padded INT32 sequence lengths beside [1,T]
    tokens. The metadata helper keeps their values as runtime data, including
    empty reserved slots and the final sentinel for unused chunk rows.
    """
    if tokens.ndim != 2:
        raise ValueError("token inputs must have shape [batch, sequence]")
    if sequence_lengths is None:
        sequence_lengths = torch.full(
            (tokens.shape[0],), tokens.shape[1], dtype=torch.int32, device=tokens.device
        )
    elif not isinstance(sequence_lengths, torch.Tensor):
        if (
            tokens.shape[0] != 1
            or not sequence_lengths
            or any(type(n) is not int or n <= 0 for n in sequence_lengths)
            or sum(sequence_lengths) != tokens.numel()
        ):
            raise ValueError("sequence lengths must cover packed [1,T] tokens")
        sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=tokens.device
        )
    elif tokens.shape[0] != 1:
        raise ValueError("runtime sequence lengths require packed [1,T] tokens")
    if sequence_lengths.numel() == 0:
        raise ValueError("sequence lengths require at least one reserved slot")
    cumulative, chunks = mlops.prepare_packed_sequence_metadata(
        sequence_lengths, tokens
    )
    return tokens.reshape(-1), cumulative, chunks


def token_features(self, tokens, sequence_lengths=None):
    flat, cumulative, chunks = packed_inputs(tokens, sequence_lengths)
    return {
        "head_hidden": self.hidden(flat, cumulative, cumulative, chunks).reshape(
            *tokens.shape, -1
        ),
        "head": self.lm_head,
    }
