"""Small helpers shared across RunRaccoon: JSON-safe values, atomic writes, hashing."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import random
import string
import tempfile
from pathlib import Path
from typing import Any

_ID_ALPHABET = string.ascii_lowercase + string.digits


def generate_id(length: int = 8) -> str:
    """A wandb-style run id: 8 lowercase alphanumerics."""
    return "".join(random.SystemRandom().choice(_ID_ALPHABET) for _ in range(length))


def timestamp_slug(t: float | None = None) -> str:
    """`20260924_083909` - the timestamp wandb puts in run directory names."""
    return _dt.datetime.fromtimestamp(t if t is not None else _dt.datetime.now().timestamp()).strftime("%Y%m%d_%H%M%S")


def iso_utc(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    """Write via a temp file + rename so concurrent readers (the dashboard) never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int | None = None) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, default=str))


def to_builtin(value: Any) -> Any:
    """Convert numpy / torch / pathlib values into plain Python so they serialize cleanly.

    Unknown objects become their `str()`; nothing here ever raises.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_builtin(v) for v in value]
    # numpy scalars / arrays and torch tensors, without importing either
    item = getattr(value, "item", None)
    shape = getattr(value, "shape", None)
    if shape is not None and callable(getattr(value, "tolist", None)):
        if hasattr(value, "detach"):  # torch tensor
            value = value.detach().cpu()
        if shape == () or shape == []:
            return to_builtin(value.item())
        return to_builtin(value.tolist())
    if callable(item):
        try:
            return to_builtin(item())
        except Exception:
            pass
    if hasattr(value, "__dict__") and type(value).__module__ == "argparse":
        return to_builtin(vars(value))
    return str(value)


def as_number(value: Any) -> float | int | bool | None:
    """Return `value` as a Python number if it is a scalar number (incl. numpy/torch 0-d), else None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    shape = getattr(value, "shape", None)
    if shape is not None and tuple(shape) in ((), (1,)) and callable(getattr(value, "item", None)):
        try:
            v = value.item()
        except Exception:
            return None
        return v if isinstance(v, (int, float, bool)) else None
    if shape is None and type(value).__module__ == "numpy" and callable(getattr(value, "item", None)):
        try:
            v = value.item()
        except (ValueError, TypeError):
            return None
        return v if isinstance(v, (int, float, bool)) else None
    return None


def json_safe(value: Any) -> Any:
    """Like `to_builtin` but also replaces NaN/inf with None (for strict JSON consumers like browsers)."""
    value = to_builtin(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def safe_relpath(key: str) -> str:
    """A media key -> a relative path. `/` keeps meaning "subdirectory" (as in wandb); other path
    hazards are neutralized."""
    parts = [p for p in str(key).replace("\\", "/").split("/") if p not in ("", ".", "..")]
    cleaned = ["".join("_" if c in '\0:*?"<>|' else c for c in p) for p in parts]
    return "/".join(cleaned) or "media"


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def format_value(v: Any) -> str:
    """Compact, human-friendly number formatting for summaries and labels."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if not math.isfinite(v):
        return str(v)
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e5 or a < 1e-3:
        return f"{v:.3g}"
    if a >= 100:
        return f"{v:,.1f}"
    return f"{v:.4g}"
