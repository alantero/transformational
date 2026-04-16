"""MIDI ↔ token-id bridge.

Tokenisation is implemented directly here (based on qsyn_mt / Oore et al. 2018)
rather than delegating to t5-midi's midi_parser.  The t5-midi file-based tokeniser
does a write-then-read roundtrip through a temp file when merging tracks, which
introduces timing drift (seconds→ticks→seconds quantisation error).

By reading the MIDI once with pretty_midi and building the token sequence entirely
in memory we avoid that drift completely.  The token IDs are identical to those
produced by t5-midi (the vocabularies match), so the model consumes the sequences
without any difference.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import sys
from pathlib import Path

import pretty_midi

from .vocab import (
    BIN_STEP,
    DIV,
    end_token,
    start_token,
    LTH,
    note_off_events,
    note_off_start,
    note_on_events,
    note_on_start,
    time_shift_events,
    time_shift_start,
    velocity_events,
    velocity_start,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MIDINoteReference:
    instrument_index: int
    note_index: int


@dataclass
class MIDITokenizationResult:
    token_ids: list[int]
    tempo_bpm: int
    source_pretty_midi: pretty_midi.PrettyMIDI
    note_on_refs: list[MIDINoteReference]
    selected_instrument_indices: list[int]
    merged_tracks: bool

def _round(a: float) -> int:
    """Round-half-up (matches qsyn_mt's round_())."""
    b = int(a // 1)
    return b + (1 if (a % 1) >= 0.5 else 0)


def _time_to_tokens(delta_ms: int) -> list[int]:
    """Convert a millisecond delta into a list of time_shift token IDs.

    Matches qsyn_mt's time_cutter + time_to_events logic exactly:
      - Emit ⌊delta_ms / LTH⌋ maximum-length time_shift tokens (125 steps = 1000 ms each).
      - Emit one final token for the leftover (if any).
    """
    tokens: list[int] = []
    for _ in range(delta_ms // LTH):
        # maximum time_shift: 125 steps → token at time_shift_start + 124
        tokens.append(time_shift_start + time_shift_events - 1)
    leftover_steps = _round((delta_ms % LTH) / DIV)
    if leftover_steps > 0:
        tokens.append(time_shift_start + leftover_steps - 1)
    return tokens


def _velocity_to_bin(velocity: int) -> int:
    return min(velocity // BIN_STEP, velocity_events - 1)


def _bin_to_velocity(bin_idx: int) -> int:
    return int(bin_idx * BIN_STEP)


def _get_first_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    _, tempos = pm.get_tempo_changes()
    return float(tempos[0]) if tempos.size > 0 else 120.0


def _select_instruments(
    pm: pretty_midi.PrettyMIDI,
    *,
    merge_tracks: bool,
    skip_drums: bool,
) -> list[tuple[int, pretty_midi.Instrument]]:
    selected: list[tuple[int, pretty_midi.Instrument]] = []
    for instrument_index, instrument in enumerate(pm.instruments):
        if skip_drums and instrument.is_drum:
            continue
        selected.append((instrument_index, instrument))
        if not merge_tracks:
            break
    return selected


# ---------------------------------------------------------------------------
# Tokenisation  (MIDI → token ids)
# ---------------------------------------------------------------------------

def _tokenize_pretty_midi(
    instruments: list[tuple[int, pretty_midi.Instrument]],
) -> tuple[list[int], list[MIDINoteReference]]:
    """Tokenise a PrettyMIDI object into a flat list of token IDs.

    The selected instruments are merged and sorted chronologically (qsyn_mt style).
    No file I/O is performed — the conversion is entirely in memory.

    Token order per note event:  set_velocity | note_on | ... | note_off
    (velocity token is emitted immediately before each note_on, matching t5-midi.)
    """
    # Keep the stable time-only sort used by t5-midi.
    events: list[tuple[float, str, int, int, MIDINoteReference | None]] = []
    for instrument_index, instrument in instruments:
        for note_index, note in enumerate(instrument.notes):
            note_ref = MIDINoteReference(instrument_index=instrument_index, note_index=note_index)
            events.append((note.start, "note_on", note.pitch, note.velocity, note_ref))
            events.append((note.end, "note_off", note.pitch, 0, None))

    events.sort(key=lambda event: event[0])

    # Wrap with <start> / <end> exactly as preprocessing2.py does.
    # compact_sequence_for_velocity_prediction keeps these tokens in the compact
    # sequence (they are not velocity tokens), so the model sees <start> as its
    # first token during both training and inference.
    tokens: list[int] = [start_token]
    note_on_refs: list[MIDINoteReference] = []
    current_time = 0.0

    for time, event_type, pitch, velocity, note_ref in events:
        delta_ms = max(0, int(round((time - current_time) * 1000)))
        current_time = time

        if delta_ms > 0:
            tokens.extend(_time_to_tokens(delta_ms))

        if event_type == "note_on":
            vel_bin = _velocity_to_bin(velocity)
            tokens.append(velocity_start + vel_bin)   # set_velocity_*
            tokens.append(note_on_start  + pitch)     # note_on_*
            if note_ref is not None:
                note_on_refs.append(note_ref)
        else:  # note_off
            tokens.append(note_off_start + pitch)     # note_off_*

    tokens.append(end_token)
    return tokens, note_on_refs


def tokenize_midi_file_for_velocity(
    midi_path: str,
    repo_path: str | None = None,   # kept for API compatibility, unused
    *,
    merge_tracks: bool = True,
    skip_drums: bool = True,
) -> MIDITokenizationResult:
    """Tokenise a MIDI file and keep the note mapping for lossless velocity write-back."""
    del repo_path  # intentionally unused, kept for CLI/API compatibility

    pm = pretty_midi.PrettyMIDI(midi_path)
    tempo = int(_get_first_tempo(pm))
    selected_instruments = _select_instruments(pm, merge_tracks=merge_tracks, skip_drums=skip_drums)
    if not selected_instruments:
        raise ValueError("The MIDI file does not contain any selected non-drum instruments")

    token_ids, note_on_refs = _tokenize_pretty_midi(selected_instruments)
    return MIDITokenizationResult(
        token_ids=token_ids,
        tempo_bpm=tempo,
        source_pretty_midi=pm,
        note_on_refs=note_on_refs,
        selected_instrument_indices=[instrument_index for instrument_index, _ in selected_instruments],
        merged_tracks=merge_tracks,
    )


def midi_file_to_token_ids(
    midi_path: str,
    repo_path: str | None = None,   # kept for API compatibility, unused
    *,
    merge_tracks: bool = True,
    skip_drums: bool = True,
) -> tuple[list[int], int]:
    """Tokenise a MIDI file and return (token_ids, tempo_bpm).

    Parameters
    ----------
    midi_path : str
        Path to the input MIDI file.
    repo_path : str | None
        Ignored (kept for backwards compatibility with the old bridge API).
    merge_tracks : bool
        If True (default), all non-drum instruments are merged before tokenisation.
        If False, only the first non-drum instrument is used.
    skip_drums : bool
        Exclude drum tracks from tokenisation.
    """
    tokenization = tokenize_midi_file_for_velocity(
        midi_path,
        repo_path=repo_path,
        merge_tracks=merge_tracks,
        skip_drums=skip_drums,
    )
    return tokenization.token_ids, tokenization.tempo_bpm


def apply_velocity_bins_to_midi(
    tokenization: MIDITokenizationResult,
    predicted_bins: list[int],
    *,
    merge_selected_tracks: bool | None = None,
) -> pretty_midi.PrettyMIDI:
    """Apply predicted velocity bins back onto the original MIDI to preserve exact timing."""
    if len(predicted_bins) != len(tokenization.note_on_refs):
        raise ValueError(
            "predicted_bins must match the number of note_on events "
            f"(got {len(predicted_bins)} and {len(tokenization.note_on_refs)})"
        )

    midi_copy = deepcopy(tokenization.source_pretty_midi)
    for note_ref, velocity_bin in zip(tokenization.note_on_refs, predicted_bins):
        note = midi_copy.instruments[note_ref.instrument_index].notes[note_ref.note_index]
        note.velocity = _bin_to_velocity(int(velocity_bin))

    if merge_selected_tracks is None:
        merge_selected_tracks = tokenization.merged_tracks
    if merge_selected_tracks:
        if not tokenization.selected_instrument_indices:
            raise ValueError("No selected instruments are available to merge")
        first_instrument = midi_copy.instruments[tokenization.selected_instrument_indices[0]]
        merged_instrument = pretty_midi.Instrument(
            program=first_instrument.program,
            is_drum=first_instrument.is_drum,
            name=first_instrument.name or "merged_velocity",
        )
        for instrument_index in tokenization.selected_instrument_indices:
            instrument = midi_copy.instruments[instrument_index]
            merged_instrument.notes.extend(deepcopy(instrument.notes))
            merged_instrument.control_changes.extend(deepcopy(instrument.control_changes))
            merged_instrument.pitch_bends.extend(deepcopy(instrument.pitch_bends))

        merged_instrument.notes.sort(key=lambda note: (note.start, note.end, note.pitch, note.velocity))
        merged_instrument.control_changes.sort(key=lambda cc: (cc.time, cc.number, cc.value))
        merged_instrument.pitch_bends.sort(key=lambda bend: (bend.time, bend.pitch))
        midi_copy.instruments = [merged_instrument]
    return midi_copy


# ---------------------------------------------------------------------------
# Reconstruction  (token ids → MIDI)
# ---------------------------------------------------------------------------

def token_ids_to_pretty_midi(
    token_ids: list[int],
    repo_path: str | None = None,   # kept for API compatibility
    *,
    name: str = "velocity",
    tempo: int = 120,
) -> pretty_midi.PrettyMIDI:
    """Reconstruct a PrettyMIDI object from a token-id sequence.

    Parameters
    ----------
    tempo : int
        BPM of the original MIDI.  MUST be passed explicitly — tempo is not
        encoded in the tokens (timing is in absolute millisecond deltas).
        Using the wrong tempo makes the output play at the wrong speed.
    """
    mid = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    track = pretty_midi.Instrument(program=0, name=name)

    current_time = 0.0   # seconds
    note_starts: dict[int, deque[tuple[float, int]]] = defaultdict(deque)
    current_vel = 64

    for token_id in token_ids:
        if not isinstance(token_id, int):
            token_id = int(token_id)

        if token_id <= 0:  # <pad>
            continue

        # ---- note_on -------------------------------------------------------
        if note_on_start <= token_id < note_on_start + note_on_events:
            pitch = token_id - note_on_start
            note_starts[pitch].append((current_time, current_vel))

        # ---- note_off ------------------------------------------------------
        elif note_off_start <= token_id < note_off_start + note_off_events:
            pitch = token_id - note_off_start
            active_notes = note_starts.get(pitch)
            if active_notes:
                start, note_velocity = active_notes.popleft()
                end = max(current_time, start + 0.001)  # guarantee positive duration
                track.notes.append(pretty_midi.Note(
                    velocity=note_velocity, pitch=pitch, start=start, end=end
                ))
                if not active_notes:
                    note_starts.pop(pitch, None)

        # ---- time_shift ----------------------------------------------------
        elif time_shift_start <= token_id < time_shift_start + time_shift_events:
            steps = token_id - time_shift_start + 1   # steps ∈ [1, 125]
            current_time += (steps * DIV) / 1000.0

        # ---- set_velocity --------------------------------------------------
        elif velocity_start <= token_id < velocity_start + velocity_events:
            vel_bin = token_id - velocity_start
            current_vel = _bin_to_velocity(vel_bin)

    # close any notes that never received a note_off
    for pitch, active_notes in note_starts.items():
        while active_notes:
            start, note_velocity = active_notes.popleft()
            track.notes.append(pretty_midi.Note(
                velocity=note_velocity, pitch=pitch,
                start=start, end=max(current_time, start + 0.001)
            ))

    mid.instruments.append(track)
    return mid


# ---------------------------------------------------------------------------
# Legacy t5-midi module loader (kept for other potential uses)
# ---------------------------------------------------------------------------

def _resolve_repo_path(repo_path: str | None) -> Path:
    if repo_path is not None:
        path = Path(repo_path).expanduser().resolve()
    else:
        path = (Path(__file__).resolve().parents[2] / "t5-midi").resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find t5-midi repository at {path}. "
            "Pass --t5_midi_repo explicitly."
        )
    return path


def load_t5_midi_modules(repo_path: str | None = None):
    path = _resolve_repo_path(repo_path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import tokenizer    # type: ignore
    import vocabulary   # type: ignore
    return tokenizer, vocabulary
