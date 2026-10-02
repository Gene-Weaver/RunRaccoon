"""`Summary` - one value per metric for the whole run (wandb-summary.json).

By default each key holds the last logged value. `run.define_metric("val/loss", summary="min")`
switches a key to its min / max / mean / first instead, exactly like wandb.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Iterator

from runraccoon.utils import to_builtin


class Summary:
    def __init__(self, on_change: Callable[[], None] | None = None):
        self._items: dict[str, Any] = {}
        self._stats: dict[str, tuple[float, int]] = {}    # running (sum, count) for "mean"
        self._lock = threading.RLock()
        self._on_change = on_change

    # --- used by Run.log --------------------------------------------------------
    def _observe(self, key: str, value: Any, mode: str | None) -> None:
        """Fold one logged value into the summary according to its define_metric mode."""
        if mode is None or mode in ("last", "none") or not isinstance(value, (int, float)) or isinstance(value, bool):
            if mode != "none":
                self._items[key] = value
            return
        old = self._items.get(key)
        if mode == "min":
            self._items[key] = value if not isinstance(old, (int, float)) else min(old, value)
        elif mode == "max":
            self._items[key] = value if not isinstance(old, (int, float)) else max(old, value)
        elif mode == "first":
            self._items.setdefault(key, value)
        elif mode == "mean":
            s, n = self._stats.get(key, (0.0, 0))
            self._stats[key] = (s + value, n + 1)
            self._items[key] = (s + value) / (n + 1)
        else:
            self._items[key] = value

    def _observe_row(self, row: dict, modes: dict[str, str]) -> None:
        with self._lock:
            for k, v in row.items():
                self._observe(k, v, modes.get(k))

    def _load(self, data: dict) -> None:
        with self._lock:
            self._items.update(data)

    def _attach(self, on_change: Callable[[], None]) -> None:
        self._on_change = on_change

    # --- user-facing dict API ---------------------------------------------------
    def update(self, d: dict | None = None, **kwargs: Any) -> None:
        with self._lock:
            for k, v in {**(d or {}), **kwargs}.items():
                self._items[str(k)] = to_builtin(v)
        if self._on_change:
            self._on_change()

    def __setitem__(self, key: str, value: Any) -> None:
        self.update({key: value})

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._items[key]
        if self._on_change:
            self._on_change()

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._items[key]
        except KeyError:
            raise AttributeError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return self._items.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def keys(self):
        return self._items.keys()

    def items(self):
        return self._items.items()

    def as_dict(self) -> dict:
        with self._lock:
            return dict(self._items)

    def __repr__(self) -> str:
        return f"Summary({self._items!r})"
