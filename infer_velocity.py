#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from velocity_transformer.data_utils import (
    IGNORE_INDEX,
    compact_sequence_for_velocity_prediction,
    reconstruct_sequence_with_predicted_velocities,
)
from velocity_transformer.midi_bridge import midi_file_to_token_ids, token_ids_to_pretty_midi
from velocity_transformer.model import VelocityTransformer
from velocity_transformer.training_utils import autocast_context, detect_device, resolve_autocast_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MIDI velocity tokens from tokenized input sequences.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model directory with config.json and model.pt")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_shard", type=str, help="Path to a .pt shard containing 2-D padded sequences")
    input_group.add_argument("--input_tensor", type=str, help="Path to a .pt file containing one sequence tensor")
    input_group.add_argument("--input_json", type=str, help="Path to a JSON file containing a list of token ids")
    input_group.add_argument("--input_text", type=str, help="Path to a text file with whitespace-separated token ids")
    input_group.add_argument("--input_ids", type=str, help="Comma-separated token ids")
    input_group.add_argument("--midi_path", type=str, help="Input MIDI file; requires t5-midi for tokenization")

    parser.add_argument("--sample_index", type=int, default=0, help="Row index when reading from a shard or 2-D tensor")
    parser.add_argument("--window_length", type=int, default=0, help="Inference window length; 0 uses the value stored in config")
    parser.add_argument("--window_overlap", type=int, default=0, help="Overlap between windows; 0 defaults to window_length // 4")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "none"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--t5_midi_repo", type=str, default=None, help="Path to ../t5-midi when using MIDI IO")
    parser.add_argument("--output_tokens_path", type=str, default=None, help="Where to save predicted token ids as .pt")
    parser.add_argument("--output_json_path", type=str, default=None, help="Where to save predicted token ids as JSON")
    parser.add_argument("--output_midi_path", type=str, default=None, help="Where to write the reconstructed MIDI")
    return parser.parse_args()


def load_input_sequence(args: argparse.Namespace) -> list[int]:
    if args.input_shard:
        tensor = torch.load(args.input_shard, map_location="cpu")
        if tensor.dim() != 2:
            raise ValueError("--input_shard must point to a 2-D tensor shard")
        return tensor[args.sample_index].tolist()
    if args.input_tensor:
        tensor = torch.load(args.input_tensor, map_location="cpu")
        if tensor.dim() == 2:
            tensor = tensor[args.sample_index]
        if tensor.dim() != 1:
            raise ValueError("--input_tensor must be a 1-D tensor or a 2-D tensor with --sample_index")
        return tensor.tolist()
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as handle:
            return list(json.load(handle))
    if args.input_text:
        content = Path(args.input_text).read_text(encoding="utf-8").strip()
        return [int(token) for token in content.split()] if content else []
    if args.input_ids:
        return [int(token.strip()) for token in args.input_ids.split(",") if token.strip()]
    if args.midi_path:
        return midi_file_to_token_ids(args.midi_path, repo_path=args.t5_midi_repo)
    raise RuntimeError("No input source provided")


def main() -> None:
    args = parse_args()

    device = detect_device(args.device)
    autocast_dtype = resolve_autocast_dtype(device, args.precision)
    print(f"Using device: {device}")

    model = VelocityTransformer.from_pretrained(args.checkpoint_path, map_location="cpu")
    model_config = model.config
    model.to(device)
    model.eval()

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this environment")
        model = torch.compile(model)

    original_sequence = load_input_sequence(args)
    compact_tokens, labels, note_on_positions = compact_sequence_for_velocity_prediction(original_sequence)
    if not compact_tokens:
        raise ValueError("The input sequence is empty after removing padding and velocity tokens")

    window_length = args.window_length or model_config.max_sequence_length
    if window_length <= 0:
        raise ValueError("window_length must be > 0")
    window_overlap = args.window_overlap or max(1, window_length // 4)
    stride = max(1, window_length - window_overlap)

    input_ids = torch.tensor(compact_tokens, dtype=torch.long)
    logits_sum = torch.zeros((input_ids.size(0), model_config.num_velocity_bins), dtype=torch.float32)
    logits_count = torch.zeros((input_ids.size(0), 1), dtype=torch.float32)

    with torch.inference_mode():
        start = 0
        while start < input_ids.size(0):
            end = min(start + window_length, input_ids.size(0))
            chunk = input_ids[start:end].unsqueeze(0).to(device)
            attention_mask = torch.ones_like(chunk)
            with autocast_context(device, autocast_dtype):
                chunk_logits = model(input_ids=chunk, attention_mask=attention_mask)["logits"][0]
            logits_sum[start:end] += chunk_logits.float().cpu()
            logits_count[start:end] += 1.0
            if end == input_ids.size(0):
                break
            start += stride

    averaged_logits = logits_sum / logits_count.clamp_min(1.0)
    predicted_bins = averaged_logits.argmax(dim=-1)
    predicted_sequence = reconstruct_sequence_with_predicted_velocities(compact_tokens, predicted_bins)

    if args.output_tokens_path:
        torch.save(torch.tensor(predicted_sequence, dtype=torch.long), args.output_tokens_path)
    if args.output_json_path:
        with open(args.output_json_path, "w", encoding="utf-8") as handle:
            json.dump(predicted_sequence, handle)
    if args.output_midi_path:
        midi = token_ids_to_pretty_midi(predicted_sequence, repo_path=args.t5_midi_repo)
        midi.write(args.output_midi_path)

    valid_labels = torch.tensor(labels).ne(IGNORE_INDEX)
    if valid_labels.any():
        target_bins = torch.tensor(labels)[valid_labels]
        pred_bins = predicted_bins[valid_labels]
        abs_error = (pred_bins - target_bins).abs()
        accuracy = pred_bins.eq(target_bins).float().mean().item()
        mae_bins = abs_error.float().mean().item()
        within_1 = abs_error.le(1).float().mean().item()
        print(
            f"Original velocity comparison on {int(valid_labels.sum().item())} note_on positions: "
            f"acc={accuracy:.4f}, mae_bins={mae_bins:.4f}, within_1={within_1:.4f}"
        )

    print(
        "Inference complete:",
        f"original_tokens={len(original_sequence)}",
        f"compact_tokens={len(compact_tokens)}",
        f"note_ons={len(note_on_positions)}",
    )


if __name__ == "__main__":
    main()
