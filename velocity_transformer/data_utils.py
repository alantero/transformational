from __future__ import annotations

from typing import Iterable

import torch

from .vocab import (
    end_token,
    is_note_on,
    is_velocity,
    pad_token,
    velocity_bin_from_token,
    velocity_token_from_bin,
)

IGNORE_INDEX = -100


def strip_padding(sequence: torch.Tensor | Iterable[int]) -> list[int]:
    if isinstance(sequence, torch.Tensor):
        values = sequence.tolist()
    else:
        values = list(sequence)
    return [token for token in values if token != pad_token]


def compact_sequence_for_velocity_prediction(
    sequence: torch.Tensor | Iterable[int],
    default_velocity_bin: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Remove explicit velocity tokens and align their labels with the following note_on.

    Returns
    -------
    compact_tokens
        Sequence without any set_velocity_* tokens.
    labels
        Velocity-bin labels aligned to positions in compact_tokens.
        Non-note positions are filled with IGNORE_INDEX.
    note_on_positions
        Positions in compact_tokens that correspond to note_on tokens.
    """

    compact_tokens: list[int] = []
    labels: list[int] = []
    note_on_positions: list[int] = []

    pending_velocity: int | None = None
    last_velocity: int | None = default_velocity_bin

    for token in strip_padding(sequence):
        if is_velocity(token):
            pending_velocity = velocity_bin_from_token(token)
            last_velocity = pending_velocity
            continue

        compact_tokens.append(token)

        if is_note_on(token):
            note_on_positions.append(len(compact_tokens) - 1)
            velocity_bin = pending_velocity if pending_velocity is not None else last_velocity
            if velocity_bin is None:
                labels.append(IGNORE_INDEX)
            else:
                labels.append(velocity_bin)
            pending_velocity = None
        else:
            labels.append(IGNORE_INDEX)
            if token == end_token:
                pending_velocity = None

    return compact_tokens, labels, note_on_positions


def reconstruct_sequence_with_predicted_velocities(
    compact_tokens: torch.Tensor | Iterable[int],
    predicted_bins: torch.Tensor | Iterable[int],
) -> list[int]:
    compact_list = compact_tokens.tolist() if isinstance(compact_tokens, torch.Tensor) else list(compact_tokens)
    predicted_list = predicted_bins.tolist() if isinstance(predicted_bins, torch.Tensor) else list(predicted_bins)

    if len(compact_list) != len(predicted_list):
        raise ValueError(
            "compact_tokens and predicted_bins must have the same length "
            f"(got {len(compact_list)} and {len(predicted_list)})"
        )

    reconstructed: list[int] = []
    for token, velocity_bin in zip(compact_list, predicted_list):
        if is_note_on(token):
            if velocity_bin < 0:
                raise ValueError("Predicted bins for note_on positions must be non-negative")
            reconstructed.append(velocity_token_from_bin(int(velocity_bin)))
        reconstructed.append(token)
    return reconstructed
