from __future__ import annotations

import os
from bisect import bisect_right

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .data_utils import IGNORE_INDEX, compact_sequence_for_velocity_prediction
from .vocab import pad_token


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
        default_velocity_bin: int | None = None,
        max_retry_samples: int = 8,
    ) -> None:
        self.min_notes_per_sequence = min_notes_per_sequence
        self.default_velocity_bin = default_velocity_bin
        self.max_retry_samples = max_retry_samples

        if os.path.isdir(path_or_dir):
            self.shard_paths = sorted(
                os.path.join(path_or_dir, filename)
                for filename in os.listdir(path_or_dir)
                if filename.endswith(".pt") and not filename.startswith(".tmp_")
            )
        elif os.path.isfile(path_or_dir) and path_or_dir.endswith(".pt"):
            self.shard_paths = [path_or_dir]
        else:
            raise ValueError(f"{path_or_dir} is neither a .pt shard nor a directory of shards")

        if not self.shard_paths:
            raise ValueError(f"No .pt shards found under {path_or_dir}")

        self.cumulative_sizes: list[int] = []
        total = 0
        for shard_path in self.shard_paths:
            shard = torch.load(shard_path, map_location="cpu")
            if shard.dim() != 2:
                raise ValueError(f"Shard {shard_path} must contain a 2-D padded tensor")
            total += shard.size(0)
            self.cumulative_sizes.append(total)
            del shard

        self._current_shard_idx: int | None = None
        self._current_shard: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

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
        supervised_notes = sum(label != IGNORE_INDEX for label in labels)
        if supervised_notes < self.min_notes_per_sequence or not compact_tokens:
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
