"""
Vocabulary and helpers compatible with ../t5-midi.

The ids match the exact event layout used by t5-midi so that existing sharded
datasets can be consumed without regeneration.
"""

from __future__ import annotations

note_on_events = 128
note_off_events = 128
time_shift_events = 125
velocity_events = 32

LTH = 1000
DIV = LTH // time_shift_events
BIN_STEP = 128 // velocity_events

note_on_vocab = [f"note_on_{i}" for i in range(note_on_events)]
note_off_vocab = [f"note_off_{i}" for i in range(note_off_events)]
time_shift_vocab = [f"time_shift_{i}" for i in range(time_shift_events)]
velocity_vocab = [f"set_velocity_{i}" for i in range(velocity_events)]

num_sentinels = 100
sentinel_tokens = [f"<sentinel_{i}>" for i in range(num_sentinels)]

vocab = (
    ["<pad>"]
    + note_on_vocab
    + note_off_vocab
    + time_shift_vocab
    + velocity_vocab
    + ["<start>", "<end>"]
    + sentinel_tokens
)
vocab_size = len(vocab)

pad_token = vocab.index("<pad>")
start_token = vocab.index("<start>")
end_token = vocab.index("<end>")
sentinel_ids = [vocab.index(token) for token in sentinel_tokens]

note_on_start = 1
note_on_end = note_on_start + note_on_events
note_off_start = note_on_end
note_off_end = note_off_start + note_off_events
time_shift_start = note_off_end
time_shift_end = time_shift_start + time_shift_events
velocity_start = time_shift_end
velocity_end = velocity_start + velocity_events


def is_note_on(token_id: int) -> bool:
    return note_on_start <= token_id < note_on_end


def is_note_off(token_id: int) -> bool:
    return note_off_start <= token_id < note_off_end


def is_time_shift(token_id: int) -> bool:
    return time_shift_start <= token_id < time_shift_end


def is_velocity(token_id: int) -> bool:
    return velocity_start <= token_id < velocity_end


def velocity_bin_from_token(token_id: int) -> int:
    if not is_velocity(token_id):
        raise ValueError(f"Token id {token_id} is not a velocity token")
    return token_id - velocity_start


def velocity_token_from_bin(velocity_bin: int) -> int:
    if not 0 <= velocity_bin < velocity_events:
        raise ValueError(f"velocity_bin must be in [0, {velocity_events}), got {velocity_bin}")
    return velocity_start + velocity_bin


def token_name(token_id: int) -> str:
    return vocab[token_id]
