#!/usr/bin/env python3
"""Velocity inference for the VelocityTransformer encoder.

Inference design rationale
--------------------------
This is an encoder-only (bidirectional) model doing per-token ordinal classification.
The following techniques from the autoregressive literature do NOT apply here:
  - Greedy / top-k / nucleus sampling  (autoregressive generation only)
  - KV-cache                           (no sequential token generation)
  - Speculative decoding               (no sequential bottleneck)
  - MC Dropout                         (empirically hurts encoder calibration in >50% of cases)

What we use instead, in order of impact:

1. Center masking  (best known strategy for sliding-window boundary artifacts)
   Tokens at the edges of each window have truncated bidirectional context and
   produce worse predictions. We only trust predictions from the center fraction
   of each window, discarding the edges entirely. Every token must appear as the
   "center" of at least one window, which drives the minimum required overlap.

2. Checkpoint ensemble  (strongest single improvement for encoder models, 2025)
   Average the logits of multiple saved checkpoints before decoding.  Even 3
   checkpoints from different training steps improve calibration meaningfully.

3. Expected-value decoding  (correct for ordinal labels)
   Velocity bins 0-31 are ordered. argmax ignores that ordering.
   E[bin] = sum(i * softmax(logit_i)) respects it and reduces MAE.

4. Temperature scaling  (post-hoc calibration, tune on validation set)
   Rescale logits by 1/T before softmax. T < 1 sharpens, T > 1 softens.

5. Multi-offset TTA  (cheap quality boost, purely additive)
   Re-run the windowed inference with different starting offsets so every token
   appears in windows centered at different points. Average the accumulated logits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from velocity_transformer.data_utils import (
    IGNORE_INDEX,
    compact_sequence_for_velocity_prediction,
    reconstruct_sequence_with_predicted_velocities,
)
from velocity_transformer.midi_bridge import midi_file_to_token_ids, token_ids_to_pretty_midi
from velocity_transformer.model import VelocityTransformer
from velocity_transformer.training_utils import autocast_context, detect_device, resolve_autocast_dtype


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict MIDI velocity tokens using an encoder-only VelocityTransformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- model ---
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Primary model directory (config.json + model.pt).",
    )
    parser.add_argument(
        "--ensemble_checkpoints", type=str, nargs="*", default=[],
        help="Additional checkpoint directories to ensemble with the primary model. "
             "Logits are averaged before decoding. "
             "Example: --ensemble_checkpoints ckpt-5000 ckpt-10000",
    )

    # --- input (mutually exclusive) ---
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input_shard",  type=str, help=".pt shard file (2-D padded tensor).")
    inp.add_argument("--input_tensor", type=str, help=".pt file with one sequence tensor.")
    inp.add_argument("--input_json",   type=str, help="JSON file with a list of token ids.")
    inp.add_argument("--input_text",   type=str, help="Text file with whitespace-separated token ids.")
    inp.add_argument("--input_ids",    type=str, help="Comma-separated token ids.")
    inp.add_argument("--midi_path",    type=str, help="Input MIDI file (requires t5-midi).")
    parser.add_argument(
        "--sample_index", type=int, default=0,
        help="Row index when reading from a shard or 2-D tensor.",
    )

    # --- sliding window ---
    parser.add_argument(
        "--window_length", type=int, default=0,
        help="Inference window length in tokens. 0 = use config.max_sequence_length.",
    )
    parser.add_argument(
        "--center_fraction", type=float, default=0.34,
        help="Fraction of each window whose predictions are trusted. "
             "Only the center (center_fraction * window_length) tokens contribute. "
             "Edges are discarded because their bidirectional context is truncated. "
             "Lower = better quality, higher compute. Default 0.34 ≈ center third.",
    )
    parser.add_argument(
        "--max_window_batch", type=int, default=64,
        help="Max windows per forward pass. Reduce if you run out of GPU memory.",
    )

    # --- decoding ---
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Softmax temperature. Tune on your validation set: "
             "<1 sharpens predictions, >1 softens. Optimal is usually 0.5-0.9.",
    )
    parser.add_argument(
        "--decoding", choices=["expected_value", "argmax"], default="expected_value",
        help="expected_value: E[bin]=sum(i*p_i), correct for ordinal labels. "
             "argmax: picks the single most likely bin, ignores ordinal structure.",
    )
    parser.add_argument(
        "--tta_passes", type=int, default=1,
        help="Test-time augmentation passes. Each pass uses a different window offset "
             "so every token appears in a different window context. "
             "1 = disabled. 3 is a good balance of quality vs compute.",
    )

    # --- system ---
    parser.add_argument("--device",    type=str, default="auto")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "none"], default="auto")
    parser.add_argument("--compile",   action="store_true", help="torch.compile the model.")
    parser.add_argument(
        "--t5_midi_repo", type=str, default=None,
        help="Path to the t5-midi repo root. Required for --midi_path and --output_midi_path.",
    )

    # --- output ---
    parser.add_argument("--output_tokens_path", type=str, default=None, help="Save predicted token ids as .pt.")
    parser.add_argument("--output_json_path",   type=str, default=None, help="Save predicted token ids as JSON.")
    parser.add_argument("--output_midi_path",   type=str, default=None, help="Write reconstructed MIDI file.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_input_sequence(args: argparse.Namespace) -> list[int]:
    if args.input_shard:
        t = torch.load(args.input_shard, map_location="cpu")
        if t.dim() != 2:
            raise ValueError("--input_shard must point to a 2-D tensor shard")
        return t[args.sample_index].tolist()
    if args.input_tensor:
        t = torch.load(args.input_tensor, map_location="cpu")
        if t.dim() == 2:
            t = t[args.sample_index]
        if t.dim() != 1:
            raise ValueError("--input_tensor must be 1-D or 2-D with --sample_index")
        return t.tolist()
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as fh:
            return list(json.load(fh))
    if args.input_text:
        raw = Path(args.input_text).read_text(encoding="utf-8").strip()
        return [int(x) for x in raw.split()] if raw else []
    if args.input_ids:
        return [int(x.strip()) for x in args.input_ids.split(",") if x.strip()]
    if args.midi_path:
        return midi_file_to_token_ids(args.midi_path, repo_path=args.t5_midi_repo)
    raise RuntimeError("No input source provided")


# ---------------------------------------------------------------------------
# Sliding-window inference with center masking
# ---------------------------------------------------------------------------

def _window_starts(seq_len: int, window_length: int, stride: int) -> list[int]:
    """Return the start positions of all windows, ensuring full sequence coverage."""
    starts: list[int] = []
    s = 0
    while s < seq_len:
        starts.append(s)
        if s + window_length >= seq_len:
            break
        s += stride
    # guarantee the tail is always covered
    last = max(0, seq_len - window_length)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _run_single_model_windows(
    model: torch.nn.Module,
    input_ids: torch.Tensor,           # (seq_len,)
    *,
    window_length: int,
    stride: int,
    center_fraction: float,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    max_window_batch: int,
    num_velocity_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate Hann-within-center-masked logits over all windows.

    For each window we:
      1. Compute center mask: only the middle `center_fraction` of each window is trusted.
      2. Within the trusted center region, apply a Hann taper so the very center
         contributes slightly more than the inner edges.
      3. Discard (weight=0) all tokens outside the trusted region.

    This eliminates boundary artifacts caused by truncated bidirectional context while
    still smoothly blending adjacent windows within the trusted zone.

    Returns
    -------
    logits_sum : (seq_len, num_velocity_bins)
    weight_sum : (seq_len,)
    """
    seq_len = input_ids.size(0)
    logits_sum = torch.zeros((seq_len, num_velocity_bins), dtype=torch.float32)
    weight_sum = torch.zeros(seq_len, dtype=torch.float32)

    starts = _window_starts(seq_len, window_length, stride)

    # precompute per-window center masks + Hann weights (shape: window_length)
    # We do this once for the full window length; the last window may be shorter.
    half_margin = int((1.0 - center_fraction) / 2.0 * window_length)

    for batch_start in range(0, len(starts), max_window_batch):
        batch_s = starts[batch_start: batch_start + max_window_batch]
        chunks, masks = [], []
        for s in batch_s:
            e = min(s + window_length, seq_len)
            chunk = input_ids[s:e]
            pad_len = window_length - chunk.size(0)
            if pad_len > 0:
                chunk = torch.cat([chunk, torch.zeros(pad_len, dtype=torch.long)])
            attn = torch.ones(window_length, dtype=torch.long)
            if pad_len > 0:
                attn[-pad_len:] = 0
            chunks.append(chunk)
            masks.append(attn)

        batched_ids  = torch.stack(chunks).to(device)   # (B, W)
        batched_mask = torch.stack(masks).to(device)    # (B, W)

        with torch.inference_mode():
            with autocast_context(device, autocast_dtype):
                all_logits = model(
                    input_ids=batched_ids, attention_mask=batched_mask
                )["logits"]                              # (B, W, V)
            all_logits = all_logits.float().cpu()

        for i, s in enumerate(batch_s):
            e = min(s + window_length, seq_len)
            actual_len = e - s

            # --- center mask ---
            # margin shrinks for short windows (sequence tail)
            margin = min(half_margin, actual_len // 3)
            center_start = margin
            center_end   = actual_len - margin

            if center_end <= center_start:
                # window too short to have a margin: trust everything
                center_start, center_end = 0, actual_len

            # Hann taper within the trusted center region
            center_len = center_end - center_start
            positions  = torch.arange(center_len, dtype=torch.float32)
            hann = (
                0.5 * (1.0 - torch.cos(2.0 * math.pi * positions / max(center_len - 1, 1)))
            ).clamp_min(1e-6)                           # (center_len,)

            abs_center_start = s + center_start
            abs_center_end   = s + center_end

            logits_sum[abs_center_start:abs_center_end] += (
                all_logits[i, center_start:center_end] * hann.unsqueeze(-1)
            )
            weight_sum[abs_center_start:abs_center_end] += hann

    return logits_sum, weight_sum


def _accumulate_tta(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    window_length: int,
    stride: int,
    center_fraction: float,
    tta_passes: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    max_window_batch: int,
    num_velocity_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run multiple TTA passes with different window offsets and accumulate."""
    seq_len = input_ids.size(0)
    total_logits = torch.zeros((seq_len, num_velocity_bins), dtype=torch.float32)
    total_weights = torch.zeros(seq_len, dtype=torch.float32)

    tta_step = max(1, stride // max(tta_passes, 1))
    for pass_idx in range(tta_passes):
        offset = pass_idx * tta_step
        if offset >= seq_len:
            break
        ids_pass = input_ids[offset:]
        l, w = _run_single_model_windows(
            model, ids_pass,
            window_length=window_length,
            stride=stride,
            center_fraction=center_fraction,
            device=device,
            autocast_dtype=autocast_dtype,
            max_window_batch=max_window_batch,
            num_velocity_bins=num_velocity_bins,
        )
        end = offset + ids_pass.size(0)
        total_logits[offset:end] += l
        total_weights[offset:end] += w

    return total_logits, total_weights


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

def predict_velocity_bins(
    models: list[torch.nn.Module],
    input_ids: torch.Tensor,
    *,
    window_length: int,
    center_fraction: float,
    tta_passes: int,
    max_window_batch: int,
    temperature: float,
    decoding: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    num_velocity_bins: int,
) -> torch.Tensor:
    """Run the full pipeline and return predicted bin indices (int64, shape [seq_len]).

    Pipeline
    --------
    1. For each model in the ensemble, run TTA sliding-window inference with
       center masking + Hann taper within the trusted zone.
    2. Average the accumulated logits across all ensemble members.
    3. Apply temperature scaling.
    4. Decode with expected value (ordinal-correct) or argmax.
    """
    seq_len = input_ids.size(0)

    # stride derived from center_fraction: every token must be covered by at least
    # one window's trusted center zone
    center_tokens = max(1, int(center_fraction * window_length))
    stride = max(1, center_tokens)

    ensemble_logits = torch.zeros((seq_len, num_velocity_bins), dtype=torch.float32)
    ensemble_weights = torch.zeros(seq_len, dtype=torch.float32)

    for model in models:
        l, w = _accumulate_tta(
            model, input_ids,
            window_length=window_length,
            stride=stride,
            center_fraction=center_fraction,
            tta_passes=tta_passes,
            device=device,
            autocast_dtype=autocast_dtype,
            max_window_batch=max_window_batch,
            num_velocity_bins=num_velocity_bins,
        )
        ensemble_logits  += l
        ensemble_weights += w

    averaged_logits = ensemble_logits / ensemble_weights.clamp_min(1e-6).unsqueeze(-1)

    if temperature != 1.0:
        averaged_logits = averaged_logits / temperature

    if decoding == "expected_value":
        # Velocity bins are ordinal. E[bin] = sum(i * p_i) respects the ordering
        # and reduces MAE vs argmax, especially when the model is uncertain between
        # two adjacent bins.
        probs       = F.softmax(averaged_logits, dim=-1)                      # (T, V)
        bin_indices = torch.arange(num_velocity_bins, dtype=torch.float32)    # (V,)
        expected    = (probs * bin_indices).sum(dim=-1)                       # (T,)
        return expected.round().long().clamp(0, num_velocity_bins - 1)
    else:
        return averaged_logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = detect_device(args.device)
    autocast_dtype = resolve_autocast_dtype(device, args.precision)

    # --- load all models (primary + ensemble) ---
    checkpoint_paths = [args.checkpoint_path] + list(args.ensemble_checkpoints)
    models: list[torch.nn.Module] = []
    for ckpt in checkpoint_paths:
        m = VelocityTransformer.from_pretrained(ckpt, map_location="cpu")
        m.to(device)
        m.eval()
        if args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available in this PyTorch version")
            m = torch.compile(m)
        models.append(m)

    model_config = VelocityTransformer.from_pretrained(
        args.checkpoint_path, map_location="cpu"
    ).config

    print(f"Device        : {device}")
    print(f"Models        : {len(models)} ({'ensemble' if len(models) > 1 else 'single'})")
    print(f"Decoding      : {args.decoding}  temperature={args.temperature}")
    print(f"Center fraction: {args.center_fraction}  TTA passes: {args.tta_passes}")

    # --- tokenise input ---
    original_sequence = load_input_sequence(args)
    compact_tokens, labels, note_on_positions = compact_sequence_for_velocity_prediction(
        original_sequence
    )
    if not compact_tokens:
        raise ValueError("The input sequence is empty after removing padding and velocity tokens")

    window_length = args.window_length or model_config.max_sequence_length
    if window_length <= 0:
        raise ValueError("window_length must be > 0")

    # --- run inference ---
    input_ids = torch.tensor(compact_tokens, dtype=torch.long)
    predicted_bins = predict_velocity_bins(
        models,
        input_ids,
        window_length=window_length,
        center_fraction=args.center_fraction,
        tta_passes=args.tta_passes,
        max_window_batch=args.max_window_batch,
        temperature=args.temperature,
        decoding=args.decoding,
        device=device,
        autocast_dtype=autocast_dtype,
        num_velocity_bins=model_config.num_velocity_bins,
    )

    predicted_sequence = reconstruct_sequence_with_predicted_velocities(
        compact_tokens, predicted_bins
    )

    # --- save outputs ---
    if args.output_tokens_path:
        torch.save(
            torch.tensor(predicted_sequence, dtype=torch.long),
            args.output_tokens_path,
        )
    if args.output_json_path:
        with open(args.output_json_path, "w", encoding="utf-8") as fh:
            json.dump(predicted_sequence, fh)
    if args.output_midi_path:
        midi = token_ids_to_pretty_midi(predicted_sequence, repo_path=args.t5_midi_repo)
        midi.write(args.output_midi_path)

    # --- metrics (only when the input already has velocity info) ---
    valid_labels = torch.tensor(labels).ne(IGNORE_INDEX)
    if valid_labels.any():
        target_bins = torch.tensor(labels)[valid_labels]
        pred_bins   = predicted_bins[valid_labels]
        abs_error   = (pred_bins - target_bins).abs()
        print(
            f"\nVelocity accuracy on {int(valid_labels.sum())} note_on positions:\n"
            f"  acc        = {pred_bins.eq(target_bins).float().mean():.4f}\n"
            f"  mae (bins) = {abs_error.float().mean():.4f}\n"
            f"  within ±1  = {abs_error.le(1).float().mean():.4f}"
        )

    print(
        f"\nInference complete: original_tokens={len(original_sequence)} "
        f"compact_tokens={len(compact_tokens)} note_ons={len(note_on_positions)}"
    )


if __name__ == "__main__":
    main()
