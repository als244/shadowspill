"""Explicit forward and reverse recurrences used as the numerical reference."""

from __future__ import annotations

import torch


def _delta_rule_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    batch, sequence, key_heads, key_width = query.shape
    value_heads, value_width = value.shape[2:]
    repeat = value_heads // key_heads
    query = query.float().repeat_interleave(repeat, dim=2)
    key = key.float().repeat_interleave(repeat, dim=2)
    state = torch.zeros(
        batch,
        value_heads,
        key_width,
        value_width,
        dtype=torch.float32,
        device=query.device,
    )
    outputs: list[torch.Tensor] = []
    for index in range(sequence):
        state = state * decay[:, index].float().exp()[:, :, None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", key[:, index], state)
        update = beta[:, index].float()[:, :, None] * (
            value[:, index].float() - prediction
        )
        state = state + key[:, index, :, :, None] * update[:, :, None, :]
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", query[:, index] * key_width**-0.5, state)
        )
    return torch.stack(outputs, dim=1).to(value.dtype)


def _delta_rule_backward_reference(
    gradient: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit reverse recurrence for the pure-PyTorch delta-rule reference."""

    batch, sequence, key_heads, key_width = query.shape
    value_heads, value_width = value.shape[2:]
    repeat = value_heads // key_heads
    expanded_query = query.float().repeat_interleave(repeat, dim=2)
    expanded_key = key.float().repeat_interleave(repeat, dim=2)
    float_value = value.float()
    float_beta = beta.float()
    multiplier = decay.float().exp()
    state = torch.zeros(
        batch,
        value_heads,
        key_width,
        value_width,
        dtype=torch.float32,
        device=query.device,
    )
    states: list[torch.Tensor] = []
    decayed_states: list[torch.Tensor] = []
    updates: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for index in range(sequence):
        decayed = state * multiplier[:, index, :, None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", expanded_key[:, index], decayed)
        update = float_beta[:, index, :, None] * (float_value[:, index] - prediction)
        state = decayed + expanded_key[:, index, :, :, None] * update[:, :, None, :]
        decayed_states.append(decayed)
        predictions.append(prediction)
        updates.append(update)
        states.append(state)

    query_gradient = torch.zeros_like(expanded_query)
    key_gradient = torch.zeros_like(expanded_key)
    value_gradient = torch.zeros_like(float_value)
    beta_gradient = torch.zeros_like(float_beta)
    decay_gradient = torch.zeros_like(decay, dtype=torch.float32)
    state_gradient = torch.zeros_like(state)
    scale = key_width**-0.5
    float_gradient = gradient.float()
    for index in reversed(range(sequence)):
        output_gradient = float_gradient[:, index]
        scaled_query = expanded_query[:, index] * scale
        query_gradient[:, index] = (
            torch.einsum("bhv,bhkv->bhk", output_gradient, states[index]) * scale
        )
        current_state_gradient = state_gradient + torch.einsum(
            "bhk,bhv->bhkv", scaled_query, output_gradient
        )
        key_gradient[:, index] += torch.einsum(
            "bhkv,bhv->bhk", current_state_gradient, updates[index]
        )
        update_gradient = torch.einsum(
            "bhk,bhkv->bhv", expanded_key[:, index], current_state_gradient
        )
        difference = float_value[:, index] - predictions[index]
        beta_gradient[:, index] = (update_gradient * difference).sum(dim=-1)
        value_gradient[:, index] = update_gradient * float_beta[:, index, :, None]
        prediction_gradient = -update_gradient * float_beta[:, index, :, None]
        key_gradient[:, index] += torch.einsum(
            "bhv,bhkv->bhk", prediction_gradient, decayed_states[index]
        )
        decayed_gradient = current_state_gradient + torch.einsum(
            "bhk,bhv->bhkv", expanded_key[:, index], prediction_gradient
        )
        decay_gradient[:, index] = (decayed_gradient * decayed_states[index]).sum(
            dim=(-1, -2)
        )
        state_gradient = decayed_gradient * multiplier[:, index, :, None, None]

    query_gradient = query_gradient.view(
        batch, sequence, key_heads, repeat, key_width
    ).sum(dim=3)
    key_gradient = key_gradient.view(batch, sequence, key_heads, repeat, key_width).sum(
        dim=3
    )
    return (
        query_gradient.to(query.dtype).contiguous(),
        key_gradient.to(key.dtype).contiguous(),
        value_gradient.to(value.dtype).contiguous(),
        beta_gradient.to(beta.dtype).contiguous(),
        decay_gradient.to(decay.dtype).contiguous(),
    )
