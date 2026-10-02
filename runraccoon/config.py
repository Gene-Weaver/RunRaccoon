"""`Config` - the run's hyperparameters, with both `config.lr` and `config["lr"]` access.

Persisted to files/config.yaml in wandb's format, where every key is wrapped as `{value: ...}`
and RunRaccoon's own bookkeeping lives under `_wandb`.
"""
from __future__ import annotations

import argparse
import threading
from typing import Any, Callable, Iterator, Mapping

from runraccoon.utils import to_builtin


def _to_mapping(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, argparse.Namespace):
        return vars(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):      # e.g. HF / OmegaConf-like objects
        return dict(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise TypeError(f"config must be a dict, argparse.Namespace or object, got {type(obj).__name__}")


class Config:
    """Dict-like hyperparameter store. Every change is written straight to config.yaml."""

    def __init__(self, data: Any = None, on_change: Callable[[], None] | None = None):
        object.__setattr__(self, "_items", {})
        object.__setattr__(self, "_on_change", on_change)
        object.__setattr__(self, "_lock", threading.RLock())
        if data is not None:
            self.update(data)

    # --- mutation -------------------------------------------------------------
    def update(self, d: Any = None, allow_val_change: bool = True, **kwargs: Any) -> None:
        with self._lock:
            new = _to_mapping(d)
            new.update(kwargs)
            self._items.update({str(k): to_builtin(v) for k, v in new.items()})
        self._changed()

    def setdefaults(self, d: Any) -> None:
        with self._lock:
            for k, v in _to_mapping(d).items():
                self._items.setdefault(str(k), to_builtin(v))
        self._changed()

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[str(key)] = to_builtin(value)
        self._changed()

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._items[key]
        self._changed()

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _attach(self, on_change: Callable[[], None]) -> None:
        object.__setattr__(self, "_on_change", on_change)

    # --- access ---------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __getattr__(self, key: str) -> Any:
        try:
            return self._items[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r}") from None

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

    def values(self):
        return self._items.values()

    def as_dict(self) -> dict:
        with self._lock:
            return dict(self._items)

    def __repr__(self) -> str:
        return f"Config({self._items!r})"
