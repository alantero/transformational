#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mido
import numpy as np
import pretty_midi

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit velocity diversity directly from MIDI files. "
            "Accepts a CSV with a filepath column, a directory of .mid/.midi files, or a single MIDI file."
        )
    )
    parser.add_argument("source", type=str, help="CSV, MIDI directory, or single MIDI file")
    parser.add_argument("--max_files", type=int, default=0, help="Optional cap on the number of MIDI files to inspect")
    parser.add_argument("--sample_stride", type=int, default=1, help="Keep one file every N files after sorting")
    parser.add_argument("--offset", type=int, default=0, help="Start file selection from this offset")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--top_k", type=int, default=10, help="How many flat/expressive files and tracks to print")
    parser.add_argument("--velocity_bins", type=int, default=32, help="Number of velocity bins for model-aligned metrics")
    parser.add_argument("--dominant_threshold", type=float, default=0.9, help="Threshold for calling one velocity bin dominant")
    parser.add_argument("--low_std_threshold", type=float, default=1.0, help="Threshold for low velocity std in binned units")
    parser.add_argument("--include_drums", action="store_true", help="Include drum tracks; default matches the non-drum pretraining flow")
    parser.add_argument("--json_output", type=str, default=None, help="Optional JSON file with the full report")
    parser.add_argument("--progress_bar", choices=["auto", "always", "never"], default="auto", help="Show tqdm progress bars")
    return parser.parse_args()


def should_enable_progress_bar(mode: str) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    return os.isatty(2)


def create_progress_bar(*, enabled: bool, total: int, desc: str):
    if not enabled or tqdm is None:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True)


def velocity_to_bin(velocity: int, velocity_bins: int) -> int:
    step = 128 // velocity_bins
    return max(0, min(velocity_bins - 1, int(velocity) // step))


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


def build_velocity_stats(velocities: list[int], *, velocity_bins: int, dominant_threshold: float, low_std_threshold: float) -> dict:
    if not velocities:
        return {
            "num_notes": 0,
            "num_unique_exact_velocities": 0,
            "num_unique_velocity_bins": 0,
            "dominant_velocity_bin": -1,
            "dominant_fraction": 0.0,
            "exact_velocity_mean": 0.0,
            "exact_velocity_std": 0.0,
            "binned_velocity_mean": 0.0,
            "binned_velocity_std": 0.0,
            "binned_velocity_range": 0,
            "binned_velocity_entropy_bits": 0.0,
            "binned_velocity_change_ratio": 0.0,
            "is_flat_exact": False,
            "is_flat_binned": False,
            "is_two_bins_or_less": False,
            "is_low_std": False,
            "is_dominant": False,
            "histogram": [0] * velocity_bins,
        }

    exact = np.asarray(velocities, dtype=np.int64)
    binned = np.asarray([velocity_to_bin(v, velocity_bins) for v in velocities], dtype=np.int64)
    counts = np.bincount(binned, minlength=velocity_bins)
    probs = counts[counts > 0] / counts.sum()
    dominant_bin = int(np.argmax(counts))
    dominant_fraction = float(counts[dominant_bin] / counts.sum())
    binned_std = float(np.std(binned))

    return {
        "num_notes": int(exact.size),
        "num_unique_exact_velocities": int(np.unique(exact).size),
        "num_unique_velocity_bins": int(np.count_nonzero(counts)),
        "dominant_velocity_bin": dominant_bin,
        "dominant_fraction": dominant_fraction,
        "exact_velocity_mean": float(np.mean(exact)),
        "exact_velocity_std": float(np.std(exact)),
        "binned_velocity_mean": float(np.mean(binned)),
        "binned_velocity_std": binned_std,
        "binned_velocity_range": int(np.max(binned) - np.min(binned)),
        "binned_velocity_entropy_bits": float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0,
        "binned_velocity_change_ratio": float(np.mean(binned[1:] != binned[:-1])) if binned.size > 1 else 0.0,
        "is_flat_exact": int(np.unique(exact).size) == 1,
        "is_flat_binned": int(np.count_nonzero(counts)) == 1,
        "is_two_bins_or_less": int(np.count_nonzero(counts)) <= 2,
        "is_low_std": binned_std <= low_std_threshold,
        "is_dominant": dominant_fraction >= dominant_threshold,
        "histogram": counts.astype(int).tolist(),
    }


def iter_midi_files(source: str) -> list[str]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() in {".mid", ".midi"}:
            return [str(path)]
        if path.suffix.lower() == ".csv":
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if "filepath" not in (reader.fieldnames or []):
                    raise ValueError(f"CSV {path} must contain a filepath column")
                return [row["filepath"] for row in reader if row.get("filepath")]
        raise ValueError(f"Unsupported file type for {path}")

    if not path.is_dir():
        raise ValueError(f"{path} is neither a supported file nor a directory")

    files = sorted(
        str(candidate)
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".mid", ".midi"}
    )
    return files


def select_files(files: list[str], *, offset: int, sample_stride: int, max_files: int) -> list[str]:
    selected = files[max(0, offset) :: max(1, sample_stride)]
    if max_files > 0:
        selected = selected[:max_files]
    return selected


def process_midi_file(path: str, *, velocity_bins: int, dominant_threshold: float, low_std_threshold: float, include_drums: bool) -> dict:
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        # Some files have data bytes outside 0-127; clip them and retry
        midi_obj = mido.MidiFile(path, clip=True)
        pm = pretty_midi.PrettyMIDI(midi_file=midi_obj)
    track_reports = []
    file_velocities: list[int] = []

    for instrument_index, instrument in enumerate(pm.instruments):
        if instrument.is_drum and not include_drums:
            continue
        velocities = [note.velocity for note in instrument.notes]
        if not velocities:
            continue

        stats = build_velocity_stats(
            velocities,
            velocity_bins=velocity_bins,
            dominant_threshold=dominant_threshold,
            low_std_threshold=low_std_threshold,
        )
        track_reports.append(
            {
                "path": path,
                "track_index": instrument_index,
                "program": int(instrument.program),
                "is_drum": bool(instrument.is_drum),
                "name": instrument.name or "",
                **stats,
            }
        )
        file_velocities.extend(velocities)

    file_stats = build_velocity_stats(
        file_velocities,
        velocity_bins=velocity_bins,
        dominant_threshold=dominant_threshold,
        low_std_threshold=low_std_threshold,
    )

    return {
        "path": path,
        "num_tracks_with_notes": len(track_reports),
        **file_stats,
        "tracks": track_reports,
    }


def analyze_files(
    files: list[str],
    *,
    workers: int,
    velocity_bins: int,
    dominant_threshold: float,
    low_std_threshold: float,
    include_drums: bool,
    top_k: int,
    progress_bar_enabled: bool,
) -> dict:
    total_files = 0
    failed_files = 0
    files_with_notes = 0
    total_tracks_with_notes = 0
    total_notes = 0

    file_flat = 0
    file_two_bins = 0
    file_low_std = 0
    file_dominant = 0

    track_flat = 0
    track_two_bins = 0
    track_low_std = 0
    track_dominant = 0

    file_histogram = np.zeros(velocity_bins, dtype=np.int64)
    track_histogram = np.zeros(velocity_bins, dtype=np.int64)

    file_unique_bins_values: list[float] = []
    file_std_values: list[float] = []
    file_entropy_values: list[float] = []
    file_notes_values: list[float] = []

    track_unique_bins_values: list[float] = []
    track_std_values: list[float] = []
    track_entropy_values: list[float] = []
    track_notes_values: list[float] = []

    file_reports: list[dict] = []
    track_reports: list[dict] = []
    failures: list[dict] = []

    progress_bar = create_progress_bar(enabled=progress_bar_enabled, total=len(files), desc="audit midi")

    def consume_report(path: str, report: dict | None, exc: Exception | None) -> None:
        nonlocal total_files, failed_files, files_with_notes, total_tracks_with_notes, total_notes
        nonlocal file_flat, file_two_bins, file_low_std, file_dominant
        nonlocal track_flat, track_two_bins, track_low_std, track_dominant

        total_files += 1
        if progress_bar is not None:
            progress_bar.update(1)

        if exc is not None:
            failed_files += 1
            failures.append({"path": path, "error": repr(exc)})
            if progress_bar is not None:
                progress_bar.set_postfix(failed=failed_files)
            return

        assert report is not None
        file_reports.append(report)
        num_notes = int(report["num_notes"])
        if num_notes == 0:
            return

        files_with_notes += 1
        total_notes += num_notes
        total_tracks_with_notes += int(report["num_tracks_with_notes"])
        file_histogram[:] = file_histogram + np.asarray(report["histogram"], dtype=np.int64)
        file_unique_bins_values.append(float(report["num_unique_velocity_bins"]))
        file_std_values.append(float(report["binned_velocity_std"]))
        file_entropy_values.append(float(report["binned_velocity_entropy_bits"]))
        file_notes_values.append(float(num_notes))

        if bool(report["is_flat_binned"]):
            file_flat += 1
        if bool(report["is_two_bins_or_less"]):
            file_two_bins += 1
        if bool(report["is_low_std"]):
            file_low_std += 1
        if bool(report["is_dominant"]):
            file_dominant += 1

        for track in report["tracks"]:
            track_reports.append(track)
            track_notes = int(track["num_notes"])
            track_histogram[:] = track_histogram + np.asarray(track["histogram"], dtype=np.int64)
            track_unique_bins_values.append(float(track["num_unique_velocity_bins"]))
            track_std_values.append(float(track["binned_velocity_std"]))
            track_entropy_values.append(float(track["binned_velocity_entropy_bits"]))
            track_notes_values.append(float(track_notes))

            if bool(track["is_flat_binned"]):
                track_flat += 1
            if bool(track["is_two_bins_or_less"]):
                track_two_bins += 1
            if bool(track["is_low_std"]):
                track_low_std += 1
            if bool(track["is_dominant"]):
                track_dominant += 1

        if progress_bar is not None:
            progress_bar.set_postfix(files_with_notes=files_with_notes, failed=failed_files)

    if workers <= 1:
        for path in files:
            try:
                report = process_midi_file(
                    path,
                    velocity_bins=velocity_bins,
                    dominant_threshold=dominant_threshold,
                    low_std_threshold=low_std_threshold,
                    include_drums=include_drums,
                )
                consume_report(path, report, None)
            except Exception as exc:
                consume_report(path, None, exc)
    else:
        try:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {
                    executor.submit(
                        process_midi_file,
                        path,
                        velocity_bins=velocity_bins,
                        dominant_threshold=dominant_threshold,
                        low_std_threshold=low_std_threshold,
                        include_drums=include_drums,
                    ): path
                    for path in files
                }

                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        report = future.result()
                        consume_report(path, report, None)
                    except Exception as exc:
                        consume_report(path, None, exc)
        except Exception as exc:
            print(
                f"Warning: parallel worker setup failed ({exc!r}); "
                "falling back to serial MIDI inspection."
            )
            for path in files:
                try:
                    report = process_midi_file(
                        path,
                        velocity_bins=velocity_bins,
                        dominant_threshold=dominant_threshold,
                        low_std_threshold=low_std_threshold,
                        include_drums=include_drums,
                    )
                    consume_report(path, report, None)
                except Exception as inner_exc:
                    consume_report(path, None, inner_exc)

    if progress_bar is not None:
        progress_bar.close()

    file_den = max(1, files_with_notes)
    track_den = max(1, len(track_reports))

    def flat_sort_key(item: dict) -> tuple:
        return (
            item["num_unique_velocity_bins"],
            item["binned_velocity_std"],
            -item["dominant_fraction"],
            item["binned_velocity_entropy_bits"],
            -item["num_notes"],
        )

    def expressive_sort_key(item: dict) -> tuple:
        return (
            -item["num_unique_velocity_bins"],
            -item["binned_velocity_entropy_bits"],
            -item["binned_velocity_std"],
            item["dominant_fraction"],
            -item["num_notes"],
        )

    return {
        "num_files_scanned": total_files,
        "num_failed_files": failed_files,
        "num_files_with_notes": files_with_notes,
        "num_tracks_with_notes": total_tracks_with_notes,
        "num_note_events": total_notes,
        "file_flat_ratio": float(file_flat / file_den),
        "file_two_bins_or_less_ratio": float(file_two_bins / file_den),
        "file_low_std_ratio": float(file_low_std / file_den),
        "file_dominant_ratio": float(file_dominant / file_den),
        "track_flat_ratio": float(track_flat / track_den),
        "track_two_bins_or_less_ratio": float(track_two_bins / track_den),
        "track_low_std_ratio": float(track_low_std / track_den),
        "track_dominant_ratio": float(track_dominant / track_den),
        "file_velocity_histogram": file_histogram.astype(int).tolist(),
        "track_velocity_histogram": track_histogram.astype(int).tolist(),
        "file_notes_summary": quantile_summary(file_notes_values),
        "file_unique_bins_summary": quantile_summary(file_unique_bins_values),
        "file_binned_std_summary": quantile_summary(file_std_values),
        "file_binned_entropy_summary": quantile_summary(file_entropy_values),
        "track_notes_summary": quantile_summary(track_notes_values),
        "track_unique_bins_summary": quantile_summary(track_unique_bins_values),
        "track_binned_std_summary": quantile_summary(track_std_values),
        "track_binned_entropy_summary": quantile_summary(track_entropy_values),
        "flattest_files": sorted(
            [
                {
                    "path": report["path"],
                    "num_notes": report["num_notes"],
                    "num_tracks_with_notes": report["num_tracks_with_notes"],
                    "num_unique_velocity_bins": report["num_unique_velocity_bins"],
                    "dominant_fraction": report["dominant_fraction"],
                    "binned_velocity_std": report["binned_velocity_std"],
                    "binned_velocity_entropy_bits": report["binned_velocity_entropy_bits"],
                }
                for report in file_reports
                if report["num_notes"] > 0
            ],
            key=flat_sort_key,
        )[:top_k],
        "most_expressive_files": sorted(
            [
                {
                    "path": report["path"],
                    "num_notes": report["num_notes"],
                    "num_tracks_with_notes": report["num_tracks_with_notes"],
                    "num_unique_velocity_bins": report["num_unique_velocity_bins"],
                    "dominant_fraction": report["dominant_fraction"],
                    "binned_velocity_std": report["binned_velocity_std"],
                    "binned_velocity_entropy_bits": report["binned_velocity_entropy_bits"],
                }
                for report in file_reports
                if report["num_notes"] > 0
            ],
            key=expressive_sort_key,
        )[:top_k],
        "flattest_tracks": sorted(
            [
                {
                    "path": track["path"],
                    "track_index": track["track_index"],
                    "program": track["program"],
                    "name": track["name"],
                    "num_notes": track["num_notes"],
                    "num_unique_velocity_bins": track["num_unique_velocity_bins"],
                    "dominant_fraction": track["dominant_fraction"],
                    "binned_velocity_std": track["binned_velocity_std"],
                    "binned_velocity_entropy_bits": track["binned_velocity_entropy_bits"],
                }
                for track in track_reports
            ],
            key=flat_sort_key,
        )[:top_k],
        "most_expressive_tracks": sorted(
            [
                {
                    "path": track["path"],
                    "track_index": track["track_index"],
                    "program": track["program"],
                    "name": track["name"],
                    "num_notes": track["num_notes"],
                    "num_unique_velocity_bins": track["num_unique_velocity_bins"],
                    "dominant_fraction": track["dominant_fraction"],
                    "binned_velocity_std": track["binned_velocity_std"],
                    "binned_velocity_entropy_bits": track["binned_velocity_entropy_bits"],
                }
                for track in track_reports
            ],
            key=expressive_sort_key,
        )[:top_k],
        "failures": failures[:top_k],
    }


def print_report(report: dict) -> None:
    print(f"Scanned {report['num_files_scanned']} files, failed {report['num_failed_files']}, files_with_notes {report['num_files_with_notes']}")
    print(f"Tracks with notes: {report['num_tracks_with_notes']}, total note events: {report['num_note_events']}")
    print(
        "File ratios:",
        f"flat={report['file_flat_ratio']:.3f}",
        f"<=2bins={report['file_two_bins_or_less_ratio']:.3f}",
        f"low_std={report['file_low_std_ratio']:.3f}",
        f"dominant={report['file_dominant_ratio']:.3f}",
    )
    print(
        "Track ratios:",
        f"flat={report['track_flat_ratio']:.3f}",
        f"<=2bins={report['track_two_bins_or_less_ratio']:.3f}",
        f"low_std={report['track_low_std_ratio']:.3f}",
        f"dominant={report['track_dominant_ratio']:.3f}",
    )

    file_unique = report["file_unique_bins_summary"]
    file_std = report["file_binned_std_summary"]
    track_unique = report["track_unique_bins_summary"]
    track_std = report["track_binned_std_summary"]

    print(
        "File unique bins:",
        f"median={file_unique['median']:.2f}",
        f"p75={file_unique['p75']:.2f}",
        f"max={file_unique['max']:.2f}",
    )
    print(
        "File binned std:",
        f"median={file_std['median']:.2f}",
        f"p75={file_std['p75']:.2f}",
        f"max={file_std['max']:.2f}",
    )
    print(
        "Track unique bins:",
        f"median={track_unique['median']:.2f}",
        f"p75={track_unique['p75']:.2f}",
        f"max={track_unique['max']:.2f}",
    )
    print(
        "Track binned std:",
        f"median={track_std['median']:.2f}",
        f"p75={track_std['p75']:.2f}",
        f"max={track_std['max']:.2f}",
    )

    print("Flattest files:")
    for item in report["flattest_files"]:
        print(
            "  ",
            item["path"],
            f"notes={item['num_notes']}",
            f"tracks={item['num_tracks_with_notes']}",
            f"unique_bins={item['num_unique_velocity_bins']}",
            f"dom={item['dominant_fraction']:.3f}",
            f"std={item['binned_velocity_std']:.2f}",
            f"entropy={item['binned_velocity_entropy_bits']:.2f}",
        )

    print("Most expressive files:")
    for item in report["most_expressive_files"]:
        print(
            "  ",
            item["path"],
            f"notes={item['num_notes']}",
            f"tracks={item['num_tracks_with_notes']}",
            f"unique_bins={item['num_unique_velocity_bins']}",
            f"dom={item['dominant_fraction']:.3f}",
            f"std={item['binned_velocity_std']:.2f}",
            f"entropy={item['binned_velocity_entropy_bits']:.2f}",
        )

    print("Flattest tracks:")
    for item in report["flattest_tracks"]:
        print(
            "  ",
            item["path"],
            f"track={item['track_index']}",
            f"program={item['program']}",
            f"notes={item['num_notes']}",
            f"unique_bins={item['num_unique_velocity_bins']}",
            f"dom={item['dominant_fraction']:.3f}",
            f"std={item['binned_velocity_std']:.2f}",
            f"entropy={item['binned_velocity_entropy_bits']:.2f}",
        )

    print("Most expressive tracks:")
    for item in report["most_expressive_tracks"]:
        print(
            "  ",
            item["path"],
            f"track={item['track_index']}",
            f"program={item['program']}",
            f"notes={item['num_notes']}",
            f"unique_bins={item['num_unique_velocity_bins']}",
            f"dom={item['dominant_fraction']:.3f}",
            f"std={item['binned_velocity_std']:.2f}",
            f"entropy={item['binned_velocity_entropy_bits']:.2f}",
        )

    if report["failures"]:
        print("Failures:")
        for item in report["failures"]:
            print("  ", item["path"], item["error"])


def main() -> None:
    args = parse_args()
    files = iter_midi_files(args.source)
    files = select_files(
        files,
        offset=args.offset,
        sample_stride=args.sample_stride,
        max_files=args.max_files,
    )
    if not files:
        raise ValueError("No MIDI files selected for analysis")

    print(
        f"Selected {len(files)} MIDI files from {args.source} "
        f"(offset={max(0, args.offset)}, stride={max(1, args.sample_stride)}, max_files={args.max_files or 'all'})"
    )
    report = analyze_files(
        files,
        workers=args.workers,
        velocity_bins=args.velocity_bins,
        dominant_threshold=args.dominant_threshold,
        low_std_threshold=args.low_std_threshold,
        include_drums=args.include_drums,
        top_k=args.top_k,
        progress_bar_enabled=should_enable_progress_bar(args.progress_bar),
    )
    print_report(report)

    payload = {"source": str(Path(args.source).expanduser().resolve()), "selected_files": len(files), "report": report}
    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"Saved JSON report to {output_path}")


if __name__ == "__main__":
    main()
