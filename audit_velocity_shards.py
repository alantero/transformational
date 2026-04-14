#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from velocity_transformer.data_utils import IGNORE_INDEX, compact_sequence_for_velocity_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit velocity diversity in already-processed t5-midi shards. "
            "The analysis works at sequence and shard level; original MIDI-file "
            "boundaries cannot be recovered unless that metadata was stored elsewhere."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Dataset root with train/ and val/, a split directory, or a single .pt shard",
    )
    parser.add_argument(
        "--split",
        choices=["auto", "train", "val", "both"],
        default="auto",
        help="Which split(s) to inspect when dataset_path is a dataset root",
    )
    parser.add_argument("--max_shards", type=int, default=0, help="Optional cap on the number of shards to inspect per split")
    parser.add_argument("--max_sequences", type=int, default=0, help="Optional cap on the number of sequences to inspect per split")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top flat/expressive shards and sequences to report")
    parser.add_argument(
        "--dominant_threshold",
        type=float,
        default=0.9,
        help="Threshold above which a single velocity bin is considered dominant in a sequence",
    )
    parser.add_argument(
        "--low_std_threshold",
        type=float,
        default=1.0,
        help="Threshold below which sequence velocity std is considered low",
    )
    parser.add_argument("--json_output", type=str, default=None, help="Optional JSON file to save the full report")
    return parser.parse_args()


def resolve_split_paths(dataset_path: str, split: str) -> list[tuple[str, Path]]:
    path = Path(dataset_path).expanduser().resolve()
    if path.is_file():
        return [(path.stem, path)]

    train_dir = path / "train"
    val_dir = path / "val"
    has_named_splits = train_dir.is_dir() or val_dir.is_dir()

    if not has_named_splits:
        return [(path.name, path)]

    if split == "auto":
        split = "both"

    splits: list[tuple[str, Path]] = []
    if split in ("train", "both") and train_dir.is_dir():
        splits.append(("train", train_dir))
    if split in ("val", "both") and val_dir.is_dir():
        splits.append(("val", val_dir))
    return splits


def list_shards(path: Path, max_shards: int) -> list[Path]:
    if path.is_file():
        shards = [path]
    else:
        shards = sorted(
            child for child in path.iterdir() if child.is_file() and child.suffix == ".pt" and not child.name.startswith(".tmp_")
        )
    if max_shards > 0:
        shards = shards[:max_shards]
    return shards


def quantile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def velocity_stats_from_labels(labels: list[int], *, dominant_threshold: float, low_std_threshold: float) -> dict[str, float | int | bool | list[int]]:
    velocity_bins = [label for label in labels if label != IGNORE_INDEX]
    if not velocity_bins:
        return {
            "num_notes": 0,
            "num_unique_velocity_bins": 0,
            "dominant_velocity_bin": -1,
            "dominant_fraction": 0.0,
            "velocity_mean": 0.0,
            "velocity_std": 0.0,
            "velocity_range": 0,
            "velocity_entropy_bits": 0.0,
            "velocity_change_ratio": 0.0,
            "is_flat": False,
            "is_two_bins_or_less": False,
            "is_low_std": False,
            "is_dominant": False,
            "histogram": [0] * 32,
        }

    array = np.asarray(velocity_bins, dtype=np.int64)
    counts = np.bincount(array, minlength=32)
    probs = counts[counts > 0] / counts.sum()
    dominant_bin = int(np.argmax(counts))
    dominant_fraction = float(counts[dominant_bin] / counts.sum())
    unique_bins = int(np.count_nonzero(counts))
    transitions = float(np.mean(array[1:] != array[:-1])) if array.size > 1 else 0.0
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
    std = float(np.std(array))

    return {
        "num_notes": int(array.size),
        "num_unique_velocity_bins": unique_bins,
        "dominant_velocity_bin": dominant_bin,
        "dominant_fraction": dominant_fraction,
        "velocity_mean": float(np.mean(array)),
        "velocity_std": std,
        "velocity_range": int(np.max(array) - np.min(array)),
        "velocity_entropy_bits": entropy,
        "velocity_change_ratio": transitions,
        "is_flat": unique_bins == 1,
        "is_two_bins_or_less": unique_bins <= 2,
        "is_low_std": std <= low_std_threshold,
        "is_dominant": dominant_fraction >= dominant_threshold,
        "histogram": counts.astype(int).tolist(),
    }


def analyze_split(
    split_name: str,
    split_path: Path,
    *,
    max_shards: int,
    max_sequences: int,
    top_k: int,
    dominant_threshold: float,
    low_std_threshold: float,
) -> dict:
    shards = list_shards(split_path, max_shards=max_shards)
    if not shards:
        raise ValueError(f"No .pt shards found under {split_path}")

    total_sequences = 0
    total_sequences_with_notes = 0
    total_notes = 0
    total_compact_tokens = 0
    flat_sequences = 0
    two_bins_or_less_sequences = 0
    low_std_sequences = 0
    dominant_sequences = 0
    empty_sequences = 0
    overall_histogram = np.zeros(32, dtype=np.int64)

    unique_bins_values: list[float] = []
    std_values: list[float] = []
    entropy_values: list[float] = []
    dominant_fraction_values: list[float] = []
    change_ratio_values: list[float] = []
    notes_per_sequence_values: list[float] = []

    flattest_sequences: list[dict] = []
    most_expressive_sequences: list[dict] = []
    shard_reports: list[dict] = []

    for shard_index, shard_path in enumerate(shards):
        if max_sequences and total_sequences >= max_sequences:
            break

        shard_tensor = torch.load(shard_path, map_location="cpu")
        if shard_tensor.dim() != 2:
            raise ValueError(f"Shard {shard_path} must contain a 2-D tensor")

        shard_sequence_count = 0
        shard_sequences_with_notes = 0
        shard_notes = 0
        shard_flat = 0
        shard_two_bins_or_less = 0
        shard_low_std = 0
        shard_dominant = 0
        shard_histogram = np.zeros(32, dtype=np.int64)
        shard_unique_bins: list[float] = []
        shard_std: list[float] = []
        shard_entropy: list[float] = []

        for row_index in range(shard_tensor.size(0)):
            if max_sequences and total_sequences >= max_sequences:
                break

            compact_tokens, labels, _ = compact_sequence_for_velocity_prediction(shard_tensor[row_index])
            velocity_stats = velocity_stats_from_labels(
                labels,
                dominant_threshold=dominant_threshold,
                low_std_threshold=low_std_threshold,
            )

            total_sequences += 1
            shard_sequence_count += 1
            total_compact_tokens += len(compact_tokens)

            num_notes = int(velocity_stats["num_notes"])
            if num_notes == 0:
                empty_sequences += 1
                continue

            total_sequences_with_notes += 1
            shard_sequences_with_notes += 1
            total_notes += num_notes
            shard_notes += num_notes

            overall_histogram += np.asarray(velocity_stats["histogram"], dtype=np.int64)
            shard_histogram += np.asarray(velocity_stats["histogram"], dtype=np.int64)

            unique_bins = float(velocity_stats["num_unique_velocity_bins"])
            std = float(velocity_stats["velocity_std"])
            entropy = float(velocity_stats["velocity_entropy_bits"])
            dominant_fraction = float(velocity_stats["dominant_fraction"])
            change_ratio = float(velocity_stats["velocity_change_ratio"])

            unique_bins_values.append(unique_bins)
            std_values.append(std)
            entropy_values.append(entropy)
            dominant_fraction_values.append(dominant_fraction)
            change_ratio_values.append(change_ratio)
            notes_per_sequence_values.append(float(num_notes))

            shard_unique_bins.append(unique_bins)
            shard_std.append(std)
            shard_entropy.append(entropy)

            if bool(velocity_stats["is_flat"]):
                flat_sequences += 1
                shard_flat += 1
            if bool(velocity_stats["is_two_bins_or_less"]):
                two_bins_or_less_sequences += 1
                shard_two_bins_or_less += 1
            if bool(velocity_stats["is_low_std"]):
                low_std_sequences += 1
                shard_low_std += 1
            if bool(velocity_stats["is_dominant"]):
                dominant_sequences += 1
                shard_dominant += 1

            sequence_report = {
                "shard": str(shard_path),
                "shard_index": shard_index,
                "row_index": row_index,
                "num_notes": num_notes,
                "num_unique_velocity_bins": int(unique_bins),
                "dominant_velocity_bin": int(velocity_stats["dominant_velocity_bin"]),
                "dominant_fraction": dominant_fraction,
                "velocity_std": std,
                "velocity_entropy_bits": entropy,
                "velocity_range": int(velocity_stats["velocity_range"]),
                "velocity_change_ratio": change_ratio,
            }
            flattest_sequences.append(sequence_report)
            most_expressive_sequences.append(sequence_report)

        if shard_sequence_count == 0:
            continue

        shard_reports.append(
            {
                "shard": str(shard_path),
                "num_sequences": shard_sequence_count,
                "num_sequences_with_notes": shard_sequences_with_notes,
                "num_notes": shard_notes,
                "flat_sequence_ratio": float(shard_flat / shard_sequences_with_notes) if shard_sequences_with_notes else 0.0,
                "two_bins_or_less_ratio": float(shard_two_bins_or_less / shard_sequences_with_notes) if shard_sequences_with_notes else 0.0,
                "low_std_ratio": float(shard_low_std / shard_sequences_with_notes) if shard_sequences_with_notes else 0.0,
                "dominant_sequence_ratio": float(shard_dominant / shard_sequences_with_notes) if shard_sequences_with_notes else 0.0,
                "mean_unique_velocity_bins": float(np.mean(shard_unique_bins)) if shard_unique_bins else 0.0,
                "mean_velocity_std": float(np.mean(shard_std)) if shard_std else 0.0,
                "mean_velocity_entropy_bits": float(np.mean(shard_entropy)) if shard_entropy else 0.0,
            }
        )

    flattest_sequences = sorted(
        flattest_sequences,
        key=lambda item: (
            item["num_unique_velocity_bins"],
            item["velocity_std"],
            -item["dominant_fraction"],
            item["velocity_entropy_bits"],
            -item["num_notes"],
        ),
    )[:top_k]
    most_expressive_sequences = sorted(
        most_expressive_sequences,
        key=lambda item: (
            -item["num_unique_velocity_bins"],
            -item["velocity_entropy_bits"],
            -item["velocity_std"],
            item["dominant_fraction"],
            -item["num_notes"],
        ),
    )[:top_k]

    flattest_shards = sorted(
        shard_reports,
        key=lambda item: (
            -item["flat_sequence_ratio"],
            -item["two_bins_or_less_ratio"],
            -item["low_std_ratio"],
            item["mean_unique_velocity_bins"],
        ),
    )[:top_k]
    most_expressive_shards = sorted(
        shard_reports,
        key=lambda item: (
            -item["mean_unique_velocity_bins"],
            -item["mean_velocity_entropy_bits"],
            -item["mean_velocity_std"],
            item["flat_sequence_ratio"],
        ),
    )[:top_k]

    sequences_with_notes_den = max(1, total_sequences_with_notes)
    report = {
        "split": split_name,
        "path": str(split_path),
        "num_shards_scanned": len(shard_reports),
        "num_sequences_scanned": total_sequences,
        "num_sequences_with_notes": total_sequences_with_notes,
        "num_empty_sequences": empty_sequences,
        "num_note_events": total_notes,
        "mean_compact_tokens_per_sequence": float(total_compact_tokens / max(1, total_sequences)),
        "flat_sequence_ratio": float(flat_sequences / sequences_with_notes_den),
        "two_bins_or_less_ratio": float(two_bins_or_less_sequences / sequences_with_notes_den),
        "low_std_sequence_ratio": float(low_std_sequences / sequences_with_notes_den),
        "dominant_sequence_ratio": float(dominant_sequences / sequences_with_notes_den),
        "overall_velocity_histogram": overall_histogram.astype(int).tolist(),
        "notes_per_sequence_summary": quantile_summary(notes_per_sequence_values),
        "unique_velocity_bins_summary": quantile_summary(unique_bins_values),
        "velocity_std_summary": quantile_summary(std_values),
        "velocity_entropy_bits_summary": quantile_summary(entropy_values),
        "dominant_fraction_summary": quantile_summary(dominant_fraction_values),
        "velocity_change_ratio_summary": quantile_summary(change_ratio_values),
        "flattest_shards": flattest_shards,
        "most_expressive_shards": most_expressive_shards,
        "flattest_sequences": flattest_sequences,
        "most_expressive_sequences": most_expressive_sequences,
    }
    return report


def print_report(report: dict) -> None:
    print()
    print(f"Split: {report['split']}")
    print(f"Path: {report['path']}")
    print(
        "Scanned:",
        f"{report['num_shards_scanned']} shards,",
        f"{report['num_sequences_scanned']} sequences,",
        f"{report['num_note_events']} note events",
    )
    print(
        "Ratios:",
        f"flat={report['flat_sequence_ratio']:.3f},",
        f"<=2 bins={report['two_bins_or_less_ratio']:.3f},",
        f"low_std={report['low_std_sequence_ratio']:.3f},",
        f"dominant={report['dominant_sequence_ratio']:.3f}",
    )

    unique_summary = report["unique_velocity_bins_summary"]
    std_summary = report["velocity_std_summary"]
    entropy_summary = report["velocity_entropy_bits_summary"]
    notes_summary = report["notes_per_sequence_summary"]
    print(
        "Unique bins per sequence:",
        f"median={unique_summary['median']:.2f},",
        f"p75={unique_summary['p75']:.2f},",
        f"max={unique_summary['max']:.2f}",
    )
    print(
        "Velocity std per sequence:",
        f"median={std_summary['median']:.2f},",
        f"p75={std_summary['p75']:.2f},",
        f"max={std_summary['max']:.2f}",
    )
    print(
        "Velocity entropy per sequence:",
        f"median={entropy_summary['median']:.2f},",
        f"p75={entropy_summary['p75']:.2f},",
        f"max={entropy_summary['max']:.2f}",
    )
    print(
        "Notes per sequence:",
        f"median={notes_summary['median']:.1f},",
        f"p75={notes_summary['p75']:.1f},",
        f"max={notes_summary['max']:.1f}",
    )

    print("Flattest shards:")
    for shard in report["flattest_shards"]:
        print(
            "  ",
            os.path.basename(shard["shard"]),
            f"flat={shard['flat_sequence_ratio']:.3f}",
            f"<=2bins={shard['two_bins_or_less_ratio']:.3f}",
            f"mean_unique={shard['mean_unique_velocity_bins']:.2f}",
            f"mean_entropy={shard['mean_velocity_entropy_bits']:.2f}",
        )

    print("Most expressive shards:")
    for shard in report["most_expressive_shards"]:
        print(
            "  ",
            os.path.basename(shard["shard"]),
            f"mean_unique={shard['mean_unique_velocity_bins']:.2f}",
            f"mean_entropy={shard['mean_velocity_entropy_bits']:.2f}",
            f"mean_std={shard['mean_velocity_std']:.2f}",
            f"flat={shard['flat_sequence_ratio']:.3f}",
        )

    print("Flattest sequences:")
    for seq in report["flattest_sequences"]:
        print(
            "  ",
            os.path.basename(seq["shard"]),
            f"row={seq['row_index']}",
            f"notes={seq['num_notes']}",
            f"unique={seq['num_unique_velocity_bins']}",
            f"dom={seq['dominant_fraction']:.3f}",
            f"std={seq['velocity_std']:.2f}",
            f"entropy={seq['velocity_entropy_bits']:.2f}",
        )

    print("Most expressive sequences:")
    for seq in report["most_expressive_sequences"]:
        print(
            "  ",
            os.path.basename(seq["shard"]),
            f"row={seq['row_index']}",
            f"notes={seq['num_notes']}",
            f"unique={seq['num_unique_velocity_bins']}",
            f"dom={seq['dominant_fraction']:.3f}",
            f"std={seq['velocity_std']:.2f}",
            f"entropy={seq['velocity_entropy_bits']:.2f}",
        )


def main() -> None:
    args = parse_args()
    split_paths = resolve_split_paths(args.dataset_path, args.split)
    if not split_paths:
        raise ValueError(f"No valid splits found under {args.dataset_path}")

    reports = []
    for split_name, split_path in split_paths:
        report = analyze_split(
            split_name,
            split_path,
            max_shards=args.max_shards,
            max_sequences=args.max_sequences,
            top_k=args.top_k,
            dominant_threshold=args.dominant_threshold,
            low_std_threshold=args.low_std_threshold,
        )
        reports.append(report)
        print_report(report)

    payload = {"dataset_path": str(Path(args.dataset_path).expanduser().resolve()), "reports": reports}
    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print()
        print(f"Saved JSON report to {output_path}")


if __name__ == "__main__":
    main()
