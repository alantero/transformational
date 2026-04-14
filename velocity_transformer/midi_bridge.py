from __future__ import annotations

import sys
from pathlib import Path


def _resolve_repo_path(repo_path: str | None) -> Path:
    if repo_path is not None:
        path = Path(repo_path).expanduser().resolve()
    else:
        path = (Path(__file__).resolve().parents[2] / "t5-midi").resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find t5-midi repository at {path}. Pass --t5_midi_repo explicitly."
        )
    return path


def load_t5_midi_modules(repo_path: str | None = None):
    path = _resolve_repo_path(repo_path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import tokenizer  # type: ignore
    import vocabulary  # type: ignore

    return tokenizer, vocabulary


def midi_file_to_token_ids(midi_path: str, repo_path: str | None = None) -> list[int]:
    tokenizer, _ = load_t5_midi_modules(repo_path)
    tensor, _, _ = tokenizer.midi_parser(fname=midi_path)
    return tensor.tolist()


def token_ids_to_pretty_midi(token_ids: list[int], repo_path: str | None = None, *, name: str = "velocity") -> object:
    tokenizer, _ = load_t5_midi_modules(repo_path)
    return tokenizer.list_parser(index_list=token_ids, fname=name)
