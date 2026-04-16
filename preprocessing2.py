#!/usr/bin/env python3
"""
preprocessing2.py – Velocity-aware MIDI preprocessing
------------------------------------------------------
One sequence per song: picks the most expressive non-drum track (highest
velocity diversity), tokenises it, and extracts a single window of
max_tokens.  With ~200k songs this gives ~200k training sequences with
maximum variety across songs instead of many windows from few songs.

Restart-safe: maintains a checkpoint log so interrupted runs resume
where they left off.  Shard numbering picks up after the highest
existing shard file.

Future passes: run with --pass N to extract different random windows
from the same files (different seed per pass).  The log is per-pass
so passes are independent.

Usage
-----
    python preprocessing2.py songs.csv /output/dir 1024
    python preprocessing2.py songs.csv /output/dir 1024 --augment
    python preprocessing2.py songs.csv /output/dir 1024 --pass 1
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import math
import os
import random
import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from random import seed as random_seed
from random import shuffle

import numpy as np
import pretty_midi
import torch
from tqdm import tqdm

from velocity_transformer.vocab import (
    BIN_STEP,
    DIV,
    LTH,
    end_token,
    note_off_start,
    note_on_events,
    note_on_start,
    pad_token,
    start_token,
    time_shift_events,
    time_shift_start,
    velocity_events,
    velocity_start,
)

# --------------------------------------------------------------------------- #
# checkpoint log helpers (unchanged)
# --------------------------------------------------------------------------- #

def load_log(path: str) -> dict[str, str]:
    """Return {'/path/midi.mid': 'OK'|'FAIL'}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return dict(line.rstrip().split(",", 1) for line in f if "," in line)


def append_log(path: str, midi_path: str, flag: str):
    with open(path, "a") as f:
        f.write(f"{midi_path},{flag}\n")


def next_shard_index(out_dir: str, prefix: str) -> int:
    pat = re.compile(rf"{prefix}_shard_(\d+)\.pt$")
    idxs = [
        int(m.group(1))
        for p in glob.glob(os.path.join(out_dir, f"{prefix}_shard_*.pt"))
        if (m := pat.search(os.path.basename(p)))
    ]
    return max(idxs) + 1 if idxs else 0


# --------------------------------------------------------------------------- #
# track scoring – pick the most expressive non-drum track
# --------------------------------------------------------------------------- #

def _velocity_bins(notes):
    """Return the velocity bin for each note (0..31)."""
    return [min(n.velocity // BIN_STEP, velocity_events - 1) for n in notes]


def score_track(instrument: pretty_midi.Instrument) -> tuple[int, float, int]:
    """Score a track by velocity expressiveness.

    Returns (unique_bins, entropy_bits, num_notes) — higher is better.
    Drums and empty tracks get (-1, 0, 0).
    """
    if instrument.is_drum or not instrument.notes:
        return (-1, 0.0, 0)
    bins = _velocity_bins(instrument.notes)
    counts = Counter(bins)
    unique = len(counts)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return (unique, entropy, total)


def select_best_track(
    pm: pretty_midi.PrettyMIDI,
) -> tuple[pretty_midi.Instrument | None, int]:
    """Return (best_instrument, instrument_index) or (None, -1)."""
    best: pretty_midi.Instrument | None = None
    best_idx = -1
    best_score = (-1, 0.0, 0)
    for idx, inst in enumerate(pm.instruments):
        sc = score_track(inst)
        if sc > best_score:
            best_score = sc
            best = inst
            best_idx = idx
    return best, best_idx


# --------------------------------------------------------------------------- #
# tokenisation (Oore et al. 2018, identical to t5-midi)
# --------------------------------------------------------------------------- #

def _round_half_up(a: float) -> int:
    b = int(a // 1)
    return b + (1 if (a % 1) >= 0.5 else 0)


def _time_to_tokens(delta_ms: int) -> list[int]:
    tokens: list[int] = []
    for _ in range(delta_ms // LTH):
        tokens.append(time_shift_start + time_shift_events - 1)
    leftover = _round_half_up((delta_ms % LTH) / DIV)
    if leftover > 0:
        tokens.append(time_shift_start + leftover - 1)
    return tokens


def tokenize_instrument(instrument: pretty_midi.Instrument) -> list[int]:
    """Tokenise one instrument's notes into a flat token-id list.

    Output order per note:  set_velocity  note_on  ...  note_off
    (velocity immediately before its note_on, matching t5-midi.)
    Does NOT include <start>/<end> wrappers — caller adds those.
    """
    events: list[tuple[float, str, int, int]] = []
    for note in instrument.notes:
        events.append((note.start, "on", note.pitch, note.velocity))
        events.append((note.end, "off", note.pitch, 0))
    events.sort(key=lambda e: e[0])

    tokens: list[int] = []
    current_time = 0.0
    for time, etype, pitch, velocity in events:
        delta_ms = max(0, int(round((time - current_time) * 1000)))
        current_time = time
        if delta_ms > 0:
            tokens.extend(_time_to_tokens(delta_ms))
        if etype == "on":
            vel_bin = min(velocity // BIN_STEP, velocity_events - 1)
            tokens.append(velocity_start + vel_bin)
            tokens.append(note_on_start + pitch)
        else:
            tokens.append(note_off_start + pitch)
    return tokens


# --------------------------------------------------------------------------- #
# optional data augmentation (pitch shift + time stretch)
# --------------------------------------------------------------------------- #

def augment_track(
    instrument: pretty_midi.Instrument,
    pitch_min: int = 36,
    pitch_max: int = 96,
    stretch_min: float = 0.9,
    stretch_max: float = 1.1,
) -> pretty_midi.Instrument:
    """Return a copy of *instrument* with random pitch shift and time stretch.

    Pitch is shifted so all notes fall within [pitch_min, pitch_max].
    Timing is uniformly stretched by a random factor in [stretch_min, stretch_max].
    Velocity is preserved.
    """
    notes = instrument.notes
    if not notes:
        return instrument

    pitches = np.array([n.pitch for n in notes])
    shift_lo = pitch_min - int(pitches.min())
    shift_hi = pitch_max - int(pitches.max())
    if shift_hi >= shift_lo:
        shift = int(np.random.choice(np.arange(shift_lo, shift_hi + 1)))
    else:
        shift = 0
    stretch = float(np.random.uniform(stretch_min, stretch_max))

    out = pretty_midi.Instrument(
        program=instrument.program,
        is_drum=instrument.is_drum,
        name=instrument.name,
    )
    for note in notes:
        out.notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=int(note.pitch + shift),
                start=note.start * stretch,
                end=note.end * stretch,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# MIDI → one sequence
# --------------------------------------------------------------------------- #

def process_midi_file(
    path: str,
    *,
    max_tokens: int = 1024,
    min_unique_bins: int = 3,
    min_notes: int = 8,
    augment: bool = False,
    pitch_min: int = 36,
    pitch_max: int = 96,
    stretch_min: float = 0.9,
    stretch_max: float = 1.1,
) -> list[list[int]]:
    """Process one MIDI file → at most one token sequence.

    1. Select the non-drum track with the highest velocity diversity.
    2. Optionally augment (pitch shift + time stretch).
    3. Tokenise into the Oore et al. vocabulary.
    4. Take a single random window of *max_tokens* (or the full sequence if shorter).
    5. Wrap with <start> / <end>.

    Returns a list with zero or one sequence (list keeps the executor interface).
    """
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        raise

    best_track, _track_idx = select_best_track(pm)
    if best_track is None:
        return []

    # Check minimum velocity diversity
    unique_bins = score_track(best_track)[0]
    if unique_bins < min_unique_bins:
        return []

    # Optional augmentation
    if augment:
        best_track = augment_track(
            best_track,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            stretch_min=stretch_min,
            stretch_max=stretch_max,
        )

    # Tokenise the full track (no start/end yet)
    raw_tokens = tokenize_instrument(best_track)

    # Space available for content between <start> and <end>
    content_max = max_tokens - 2

    if len(raw_tokens) <= content_max:
        seq = [start_token] + raw_tokens + [end_token]
    else:
        offset = random.randint(0, len(raw_tokens) - content_max)
        seq = [start_token] + raw_tokens[offset : offset + content_max] + [end_token]

    # Check minimum note_on count in the final window
    n_notes = sum(
        1 for t in seq if note_on_start <= t < note_on_start + note_on_events
    )
    if n_notes < min_notes:
        return []

    return [seq]


# wrapper for ProcessPoolExecutor (must be top-level for pickling)
def _process_file(path: str, **kw) -> list[list[int]]:
    return process_midi_file(path, **kw)


# --------------------------------------------------------------------------- #
# shard writer – marks MIDI as OK only after shard flush
# --------------------------------------------------------------------------- #

def stream_save_sequences(
    iterable,
    out_dir: str,
    prefix: str,
    seq_len: int,
    log_path: str,
    start_idx: int = 0,
    shard_size: int = 100_000,
):
    os.makedirs(out_dir, exist_ok=True)
    buf: list[list[int]] = []
    ok_buffer: set[str] = set()
    idx = start_idx

    def flush():
        nonlocal idx
        if not buf:
            return
        random.shuffle(buf)
        pad = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in buf],
            batch_first=True,
            padding_value=pad_token,
        )
        if pad.size(1) < seq_len:
            pad = torch.cat(
                [
                    pad,
                    torch.full(
                        (pad.size(0), seq_len - pad.size(1)),
                        pad_token,
                        dtype=pad.dtype,
                    ),
                ],
                dim=1,
            )
        tmp = os.path.join(out_dir, f".tmp_{prefix}_{idx:03d}.pt")
        final = os.path.join(out_dir, f"{prefix}_shard_{idx:03d}.pt")
        torch.save(pad, tmp)
        os.replace(tmp, final)

        for pth in ok_buffer:
            append_log(log_path, pth, "OK")
        ok_buffer.clear()
        buf.clear()
        idx += 1

    for midi_path, seq in iterable:
        buf.append(seq)
        ok_buffer.add(midi_path)
        if len(buf) >= shard_size:
            flush()
    flush()


# --------------------------------------------------------------------------- #
# parallel sequence generator with bounded futures
# --------------------------------------------------------------------------- #

def sequence_generator(
    files: list[str],
    log_path: str,
    workers: int,
    desc: str,
    **pf_kw,
):
    done = load_log(log_path)
    todo = [p for p in files if p not in done]
    skipped = len(files) - len(todo)
    if skipped:
        print(f"  [{desc}] skipping {skipped} already-processed files")

    with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(
        total=len(todo), desc=desc
    ) as bar:
        it = iter(todo)
        pending: dict = {}

        for _ in range(min(workers * 2, len(todo))):
            try:
                p = next(it)
            except StopIteration:
                break
            fut = ex.submit(_process_file, p, **pf_kw)
            pending[fut] = p

        while pending:
            done_set, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done_set:
                path = pending.pop(fut)
                try:
                    seqs = fut.result()
                    flag = "OK-buffer"
                except Exception:
                    seqs, flag = [], "FAIL"
                    append_log(log_path, path, flag)

                if flag == "OK-buffer":
                    for s in seqs:
                        yield path, s

                bar.update(1)
                gc.collect()

                try:
                    nxt = next(it)
                    nf = ex.submit(_process_file, nxt, **pf_kw)
                    pending[nf] = nxt
                except StopIteration:
                    pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    pa = argparse.ArgumentParser(
        description="Preprocess MIDI files into velocity-prediction training shards. "
        "One sequence per song from the most expressive track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pa.add_argument("source", help="CSV with a 'filepath' column pointing to MIDI files")
    pa.add_argument("dest", help="Output base directory (train/ and val/ created inside)")
    pa.add_argument("length", type=int, help="Max sequence length in tokens (e.g. 1024)")

    pa.add_argument("--min-unique-bins", type=int, default=3,
                    help="Skip tracks with fewer unique velocity bins")
    pa.add_argument("--min-notes", type=int, default=8,
                    help="Skip sequences with fewer note_on tokens")
    pa.add_argument("--train-fraction", type=float, default=0.8,
                    help="Fraction of files for training (rest goes to val)")
    pa.add_argument("--shard-size", type=int, default=100_000,
                    help="Max sequences per shard file")
    pa.add_argument("--workers", type=int, default=8,
                    help="Parallel MIDI processing workers")
    pa.add_argument("--pass", type=int, default=0, dest="pass_num",
                    help="Pass number: each pass uses a different random seed "
                         "to extract different windows. Log files are per-pass.")

    aug = pa.add_argument_group("data augmentation (disabled by default)")
    aug.add_argument("--augment", action="store_true",
                     help="Enable random pitch shift + time stretch")
    aug.add_argument("--pitch-range", nargs=2, type=int, default=[36, 96],
                     metavar=("MIN", "MAX"))
    aug.add_argument("--stretch-range", nargs=2, type=float, default=[0.9, 1.1],
                     metavar=("MIN", "MAX"))

    args = pa.parse_args()

    seed = 42 + args.pass_num
    random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Read MIDI paths from CSV
    with open(args.source, newline="", encoding="utf-8") as f:
        midi_paths = [r["filepath"] for r in csv.DictReader(f)]

    shuffle(midi_paths)
    split = int(len(midi_paths) * args.train_fraction)
    train_files, val_files = midi_paths[:split], midi_paths[split:]
    print(f"Source CSV: {len(midi_paths)} files → {len(train_files)} train, {len(val_files)} val")
    print(f"Pass {args.pass_num} (seed={seed}), augment={'ON' if args.augment else 'OFF'}")

    pf_kw = dict(
        max_tokens=args.length,
        min_unique_bins=args.min_unique_bins,
        min_notes=args.min_notes,
        augment=args.augment,
        pitch_min=args.pitch_range[0],
        pitch_max=args.pitch_range[1],
        stretch_min=args.stretch_range[0],
        stretch_max=args.stretch_range[1],
    )

    base = args.dest.rstrip("/")
    train_dir = os.path.join(base, "train")
    val_dir = os.path.join(base, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    pass_suffix = f"_pass{args.pass_num}" if args.pass_num > 0 else ""

    # --- TRAIN ---
    train_log = os.path.join(base, f"train_done{pass_suffix}.txt")
    train_iter = sequence_generator(
        train_files, train_log, workers=args.workers, desc="TRAIN", **pf_kw
    )
    stream_save_sequences(
        train_iter,
        train_dir,
        "train",
        seq_len=args.length,
        log_path=train_log,
        start_idx=next_shard_index(train_dir, "train"),
        shard_size=args.shard_size,
    )

    # --- VAL ---
    val_log = os.path.join(base, f"val_done{pass_suffix}.txt")
    val_iter = sequence_generator(
        val_files, val_log, workers=args.workers, desc="VAL", **pf_kw
    )
    stream_save_sequences(
        val_iter,
        val_dir,
        "val",
        seq_len=args.length,
        log_path=val_log,
        start_idx=next_shard_index(val_dir, "val"),
        shard_size=args.shard_size,
    )

    print("Done.")


if __name__ == "__main__":
    main()
