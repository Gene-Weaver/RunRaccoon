"""History: one JSON object per committed step, appended to files/wandb-history.jsonl.

    {"_step": 12, "_runtime": 341.2, "_timestamp": 1790272076.2, "train/loss": 0.41, ...}

Rows that arrive late for a step that was already written (Ultralytics logs its final plots
this way) are appended as extra rows with the same `_step`; readers merge them.
NaN / inf are written as the bare tokens Python's json module round-trips.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Iterator


class HistoryWriter:
    def __init__(self, path: Path, append: bool = False):
        self.path = path
        self._fh = open(path, "a" if append else "w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        line = json.dumps(row, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def iter_rows(path: Path, offset: int = 0) -> Iterator[tuple[dict, int]]:
    """Yield (row, end_offset) for each complete line from byte `offset` on. A partial last line
    (being written right now) is left for the next call."""
    try:
        fh = open(path, "rb")
    except OSError:
        return
    with fh:
        fh.seek(offset)
        pos = offset
        for raw in fh:
            if not raw.endswith(b"\n"):
                break
            pos += len(raw)
            try:
                yield json.loads(raw), pos
            except ValueError:
                continue


def read_rows(path: Path) -> list[dict]:
    return merge_rows(row for row, _ in iter_rows(path))


def merge_rows(rows: Iterable[dict]) -> list[dict]:
    """Combine rows that share a `_step` (keeps first-seen order)."""
    by_step: dict = {}
    order = []
    for row in rows:
        step = row.get("_step")
        if step in by_step:
            by_step[step].update({k: v for k, v in row.items() if k not in ("_runtime", "_timestamp")})
        else:
            by_step[step] = dict(row)
            order.append(step)
    return [by_step[s] for s in order]
