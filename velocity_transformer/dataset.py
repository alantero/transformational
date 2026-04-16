from __future__ import annotations

import json
import os
import time
from bisect import bisect_right
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .data_utils import IGNORE_INDEX, compact_sequence_for_velocity_prediction
from .vocab import pad_token

MANIFEST_VERSION = 1
DEFAULT_MANIFEST_FILENAME = ".velocity_shard_manifest.json"


class ShardedMIDIVelocityDataset(Dataset):
    """
    Lazy dataset over one or more t5-midi .pt shards.

    Each original sequence already contains explicit set_velocity_* tokens.
    This dataset removes them on the fly and turns the following note_on
    positions into velocity-bin classification targets.
    """

    def __init__(
        self,
        path_or_dir: str,
        *,
        min_notes_per_sequence: int = 1,
        min_unique_velocity_bins: int = 0,
        default_velocity_bin: int | None = None,
        max_retry_samples: int = 8,
        use_manifest_cache: bool = True,
        manifest_path: str | None = None,
        progress_label: str | None = None,
        log_every_n_shards: int = 25,
        shard_offset: int = 0,
        shard_stride: int = 1,
        max_shards: int = 0,
    ) -> None:
        self.min_notes_per_sequence = min_notes_per_sequence
        self.min_unique_velocity_bins = min_unique_velocity_bins
        self.default_velocity_bin = default_velocity_bin
        self.max_retry_samples = max_retry_samples
        self.use_manifest_cache = use_manifest_cache
        self.progress_label = progress_label or os.path.basename(os.path.abspath(path_or_dir))
        self.log_every_n_shards = max(1, log_every_n_shards)
        self.source_path = os.path.abspath(path_or_dir)
        self.shard_offset = max(0, shard_offset)
        self.shard_stride = max(1, shard_stride)
        self.max_shards = max(0, max_shards)

        if os.path.isdir(path_or_dir):
            shard_paths = sorted(
                os.path.join(path_or_dir, filename)
                for filename in os.listdir(path_or_dir)
                if filename.endswith(".pt") and not filename.startswith(".tmp_")
            )
        elif os.path.isfile(path_or_dir) and path_or_dir.endswith(".pt"):
            shard_paths = [path_or_dir]
        else:
            raise ValueError(f"{path_or_dir} is neither a .pt shard nor a directory of shards")

        shard_paths = shard_paths[self.shard_offset :: self.shard_stride]
        if self.max_shards > 0:
            shard_paths = shard_paths[: self.max_shards]
        self.shard_paths = shard_paths

        if not self.shard_paths:
            raise ValueError(f"No .pt shards found under {path_or_dir}")

        self.manifest_path = self._resolve_manifest_path(manifest_path)
        self.shard_row_counts = self._load_or_build_index()
        self.cumulative_sizes: list[int] = []
        total = 0
        for row_count in self.shard_row_counts:
            total += row_count
            self.cumulative_sizes.append(total)
        self.total_sequences = total

        self._current_shard_idx: int | None = None
        self._current_shard: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _resolve_manifest_path(self, manifest_path: str | None) -> Path | None:
        if not self.use_manifest_cache:
            return None
        if manifest_path is not None:
            return Path(manifest_path).expanduser().resolve()
        if os.path.isdir(self.source_path):
            return Path(self.source_path) / DEFAULT_MANIFEST_FILENAME
        return None

    def _stat_mtime_ns(self, shard_path: str) -> int:
        stat = os.stat(shard_path)
        return int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))

    def _current_manifest_entries(self) -> list[dict[str, int | str]]:
        return [
            {
                "name": os.path.basename(shard_path),
                "size_bytes": int(os.path.getsize(shard_path)),
                "mtime_ns": self._stat_mtime_ns(shard_path),
            }
            for shard_path in self.shard_paths
        ]

    def _load_manifest_row_counts(self) -> list[int] | None:
        if self.manifest_path is None or not self.manifest_path.exists():
            return None

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[index:{self.progress_label}] warning: could not read manifest {self.manifest_path}: {exc}")
            return None

        if payload.get("version") != MANIFEST_VERSION:
            return None

        saved_entries = payload.get("shards")
        if not isinstance(saved_entries, list) or len(saved_entries) != len(self.shard_paths):
            return None

        current_entries = self._current_manifest_entries()
        row_counts: list[int] = []
        for current, saved in zip(current_entries, saved_entries):
            if (
                saved.get("name") != current["name"]
                or int(saved.get("size_bytes", -1)) != current["size_bytes"]
                or int(saved.get("mtime_ns", -1)) != current["mtime_ns"]
            ):
                return None
            row_count = int(saved.get("num_rows", -1))
            if row_count < 0:
                return None
            row_counts.append(row_count)

        print(
            f"[index:{self.progress_label}] loaded manifest {self.manifest_path} "
            f"for {len(row_counts)} shards, total sequences: {sum(row_counts)}"
        )
        return row_counts

    def _write_manifest(self, row_counts: list[int]) -> None:
        if self.manifest_path is None:
            return

        payload = {
            "version": MANIFEST_VERSION,
            "created_at_unix": time.time(),
            "source_path": self.source_path,
            "total_sequences": int(sum(row_counts)),
            "shards": [],
        }
        for meta, row_count in zip(self._current_manifest_entries(), row_counts):
            payload["shards"].append({**meta, "num_rows": int(row_count)})

        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.manifest_path.with_name(f"{self.manifest_path.name}.tmp-{os.getpid()}")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, self.manifest_path)
            print(f"[index:{self.progress_label}] wrote manifest to {self.manifest_path}")
        except OSError as exc:
            print(
                f"[index:{self.progress_label}] warning: could not write manifest "
                f"{self.manifest_path}: {exc}"
            )

    def _build_row_counts_from_shards(self) -> list[int]:
        print(
            f"[index:{self.progress_label}] indexing {len(self.shard_paths)} shard(s) under {self.source_path} "
            f"(offset={self.shard_offset}, stride={self.shard_stride}, max_shards={self.max_shards or 'all'})"
        )
        row_counts: list[int] = []
        total = 0
        for shard_number, shard_path in enumerate(self.shard_paths, start=1):
            shard = torch.load(shard_path, map_location="cpu")
            if shard.dim() != 2:
                raise ValueError(f"Shard {shard_path} must contain a 2-D padded tensor")
            row_count = int(shard.size(0))
            row_counts.append(row_count)
            total += row_count
            del shard

            if (
                shard_number == 1
                or shard_number == len(self.shard_paths)
                or shard_number % self.log_every_n_shards == 0
            ):
                print(
                    f"[index:{self.progress_label}] {shard_number}/{len(self.shard_paths)} shards "
                    f"indexed, total sequences so far: {total}"
                )

        print(
            f"[index:{self.progress_label}] completed index: {len(self.shard_paths)} shards, "
            f"total sequences: {total}"
        )
        return row_counts

    def _load_or_build_index(self) -> list[int]:
        if self.use_manifest_cache:
            row_counts = self._load_manifest_row_counts()
            if row_counts is not None:
                return row_counts
            if self.manifest_path is not None:
                print(
                    f"[index:{self.progress_label}] manifest missing or stale; "
                    "falling back to direct shard indexing"
                )

        row_counts = self._build_row_counts_from_shards()
        if self.use_manifest_cache:
            self._write_manifest(row_counts)
        return row_counts

    def _load_shard(self, shard_idx: int) -> None:
        if shard_idx == self._current_shard_idx:
            return
        self._current_shard = torch.load(self.shard_paths[shard_idx], map_location="cpu")
        self._current_shard_idx = shard_idx

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        shard_idx = bisect_right(self.cumulative_sizes, idx)
        local_idx = idx if shard_idx == 0 else idx - self.cumulative_sizes[shard_idx - 1]
        return shard_idx, local_idx

    def _prepare_example(self, idx: int) -> dict[str, torch.Tensor] | None:
        shard_idx, local_idx = self._resolve_index(idx)
        self._load_shard(shard_idx)
        assert self._current_shard is not None
        sequence = self._current_shard[local_idx]

        compact_tokens, labels, note_on_positions = compact_sequence_for_velocity_prediction(
            sequence, default_velocity_bin=self.default_velocity_bin
        )
        supervised_labels = [label for label in labels if label != IGNORE_INDEX]
        if len(supervised_labels) < self.min_notes_per_sequence or not compact_tokens:
            return None
        if self.min_unique_velocity_bins > 0:
            if len(set(supervised_labels)) < self.min_unique_velocity_bins:
                return None

        input_ids = torch.tensor(compact_tokens, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        probe_idx = idx
        for _ in range(self.max_retry_samples):
            example = self._prepare_example(probe_idx)
            if example is not None:
                return example
            probe_idx = (probe_idx + 1) % len(self)

        raise RuntimeError(
            "Could not find a valid sample near index "
            f"{idx}. Consider lowering min_notes_per_sequence."
        )


class VelocityPredictionCollator:
    def __init__(self, pad_token_id: int = pad_token, ignore_index: int = IGNORE_INDEX) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=self.ignore_index),
        }
