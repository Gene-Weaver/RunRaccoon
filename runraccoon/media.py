"""Loggable media: `Image`, `Table`, `Histogram`, and file-backed `Video` / `Audio` / `Html`.

Each object snapshots its data when constructed (like wandb), and is written to disk when it is
logged, because only then are its key and step known:

    files/media/images/<key>_<step>_<sha256[:20]>.<ext>
    files/media/table/<key>_<step>_<sha256[:20]>.table.json

The history row / summary then holds a small JSON reference, e.g.
    {"_type": "image-file", "path": "media/images/qc/sheet_7_ab12....jpg", "width": 640, ...}
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from runraccoon.utils import json_safe, safe_relpath, sha256_bytes, sha256_file, to_builtin

log = logging.getLogger("runraccoon")
_warned: set[str] = set()


def _warn_once(tag: str, message: str) -> None:
    if tag not in _warned:
        _warned.add(tag)
        log.warning(message)


class Media:
    """Base class. Subclasses implement `_bind(files_dir, key, step) -> dict`."""

    _type = "media"
    _subdir = "media"

    def _bind(self, files_dir: Path, key: str, step: int) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _target(files_dir: Path, subdir: str, key: str, step: int, digest: str, ext: str) -> Path:
        rel = f"{safe_relpath(key)}_{step}_{digest[:20]}.{ext}"
        path = files_dir / "media" / subdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------------------------------- Image
class Image(Media):
    """An image from a file path, PIL image, numpy array, torch tensor, or matplotlib figure.

    Arrays may be HW, HWC or CHW; float data in [0, 1] (or [-1, 1]) is scaled to 0-255, any other
    float range is min-max normalized. `boxes`, `masks` and `classes` are accepted for wandb
    compatibility but are not drawn - render overlays yourself before logging.
    """

    _type = "image-file"

    def __init__(self, data_or_path: Any, mode: str | None = None, caption: str | None = None,
                 grouping: int | None = None, classes: Any = None, boxes: Any = None, masks: Any = None,
                 file_type: str | None = None, normalize: bool = True, **_ignored: Any):
        self.caption = None if caption is None else str(caption)
        self._path: Path | None = None
        self._bytes: bytes | None = None
        self._format = (file_type or "png").lower().lstrip(".")
        self.width = self.height = None
        if boxes is not None or masks is not None:
            _warn_once("overlays", "Image(boxes=/masks=) overlays are not drawn; "
                                   "the plain image is saved.")

        if isinstance(data_or_path, (str, os.PathLike)):
            self._path = Path(data_or_path)
            if not self._path.is_file():
                raise FileNotFoundError(f"Image path does not exist: {self._path}")
            self._format = (file_type or self._path.suffix.lstrip(".") or "png").lower()
            self.width, self.height = _probe_size(self._path)
            return

        pil = _to_pil(data_or_path, mode=mode, normalize=normalize)
        self.width, self.height = pil.size
        if self._format in ("jpg", "jpeg") and pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG" if self._format in ("jpg", "jpeg") else self._format.upper(), quality=90)
        self._bytes = buf.getvalue()

    @property
    def format(self) -> str:
        return "jpg" if self._format == "jpeg" else self._format

    def _bind(self, files_dir: Path, key: str, step: int) -> dict:
        if self._path is not None:
            digest = sha256_file(self._path)
            target = self._target(files_dir, "images", key, step, digest, self.format)
            if not target.exists():
                shutil.copyfile(self._path, target)
            size = target.stat().st_size
        else:
            assert self._bytes is not None
            digest = sha256_bytes(self._bytes)
            target = self._target(files_dir, "images", key, step, digest, self.format)
            if not target.exists():
                target.write_bytes(self._bytes)
            size = len(self._bytes)
        ref = {"_type": self._type, "path": target.relative_to(files_dir).as_posix(), "format": self.format,
               "width": self.width, "height": self.height, "sha256": digest, "size": size}
        if self.caption is not None:
            ref["caption"] = self.caption
        return ref


def bind_image_list(images: Sequence[Image], files_dir: Path, key: str, step: int) -> dict:
    """A list of Images under one key, stored the way wandb stores it ("images/separated")."""
    refs = [img._bind(files_dir, key, step) for img in images]
    out = {"_type": "images/separated", "count": len(refs), "filenames": [r["path"] for r in refs],
           "width": refs[0]["width"] if refs else None, "height": refs[0]["height"] if refs else None,
           "format": refs[0]["format"] if refs else None}
    captions = [r.get("caption") for r in refs]
    if any(c is not None for c in captions):
        out["captions"] = captions
    return out


def _probe_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            return im.size
    except Exception:
        return None, None


def _to_pil(data: Any, mode: str | None = None, normalize: bool = True):
    from PIL import Image as PILImage

    if isinstance(data, PILImage.Image):
        return data.convert(mode) if mode else data
    if hasattr(data, "savefig"):                              # matplotlib Figure
        buf = io.BytesIO()
        data.savefig(buf, format="png", bbox_inches="tight", dpi=data.get_dpi())
        buf.seek(0)
        return PILImage.open(buf).convert("RGBA")
    if type(data).__module__.startswith("plotly"):
        raise TypeError("plotly figures are not supported by RunRaccoon.Image; use matplotlib or an image file")

    import numpy as np

    is_tensor = hasattr(data, "detach") and hasattr(data, "cpu")
    arr = data.detach().cpu().numpy() if is_tensor else np.asarray(data)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and (is_tensor or arr.shape[-1] not in (1, 3, 4)):
        arr = np.transpose(arr, (1, 2, 0))                    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError(f"Image arrays must be HW, HWC or CHW; got shape {arr.shape}")

    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr.astype(np.float64))
        lo, hi = (float(arr.min()), float(arr.max())) if arr.size else (0.0, 1.0)
        if lo >= 0.0 and hi <= 1.0:
            arr = arr * 255.0
        elif lo >= -1.0 and hi <= 1.0:
            arr = (arr + 1.0) * 127.5
        elif normalize and hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(arr)
    return img.convert(mode) if mode else img


# ---------------------------------------------------------------------------------------- Table
class Table(Media):
    """A small table of rows. Saved as `{"columns": [...], "data": [[...], ...]}` (wandb's format)."""

    _type = "table-file"

    def __init__(self, columns: Sequence[str] | None = None, data: Iterable[Sequence[Any]] | None = None,
                 rows: Iterable[Sequence[Any]] | None = None, dataframe: Any = None, **_ignored: Any):
        if dataframe is not None:
            columns = [str(c) for c in dataframe.columns]
            data = dataframe.values.tolist()
        rows_in = [list(r) for r in (data if data is not None else rows or [])]
        if columns is None:
            columns = [f"col{i}" for i in range(len(rows_in[0]) if rows_in else 0)]
        self.columns: list[str] = [str(c) for c in columns]
        self.data: list[list[Any]] = [to_builtin(r) for r in rows_in]

    def add_data(self, *row: Any) -> None:
        if len(row) != len(self.columns):
            raise ValueError(f"row has {len(row)} values but the table has {len(self.columns)} columns")
        self.data.append(to_builtin(list(row)))

    def add_column(self, name: str, data: Sequence[Any]) -> None:
        if self.data and len(data) != len(self.data):
            raise ValueError("new column length does not match the number of rows")
        self.columns.append(str(name))
        if not self.data:
            self.data = [[v] for v in to_builtin(list(data))]
        else:
            for r, v in zip(self.data, data):
                r.append(to_builtin(v))

    def get_column(self, name: str, convert_to: str | None = None) -> list:
        i = self.columns.index(name)
        col = [r[i] for r in self.data]
        if convert_to == "numpy":
            import numpy as np
            return np.asarray(col)
        return col

    def to_json_obj(self) -> dict:
        return {"columns": self.columns, "data": json_safe(self.data)}

    def _bind(self, files_dir: Path, key: str, step: int, extra: dict | None = None) -> dict:
        payload = json.dumps(self.to_json_obj()).encode("utf-8")
        digest = sha256_bytes(payload)
        target = self._target(files_dir, "table", key, step, digest, "table.json")
        if not target.exists():
            target.write_bytes(payload)
        ref = {"_type": self._type, "path": target.relative_to(files_dir).as_posix(), "sha256": digest,
               "size": len(payload), "ncols": len(self.columns), "nrows": len(self.data)}
        if extra:
            ref.update(extra)
        return ref


# ------------------------------------------------------------------------------------ Histogram
class Histogram(Media):
    """A histogram, stored inline in the history row (no file), as wandb does."""

    _type = "histogram"

    def __init__(self, sequence: Any = None, np_histogram: tuple | None = None, num_bins: int = 64):
        import numpy as np

        if np_histogram is not None:
            counts, bins = np_histogram
        else:
            values = np.asarray(to_builtin(sequence), dtype=float).ravel()
            values = values[np.isfinite(values)]
            counts, bins = np.histogram(values, bins=min(num_bins, 512)) if values.size else ([], [])
        self.histogram = [int(c) for c in np.asarray(counts).tolist()]
        self.bins = [float(b) for b in np.asarray(bins).tolist()]

    def _bind(self, files_dir: Path, key: str, step: int) -> dict:
        return {"_type": self._type, "values": self.histogram, "bins": self.bins}


# --------------------------------------------------------------------------- file-backed media
class _FileMedia(Media):
    """Video / Audio / Html from a file path (or html text): copied into media/<kind>/."""

    _kind = "files"

    def __init__(self, data_or_path: Any, caption: str | None = None, format: str | None = None, **_ignored: Any):
        self.caption = caption
        self._text: str | None = None
        self._path: Path | None = None
        if isinstance(data_or_path, (str, os.PathLike)) and Path(data_or_path).is_file():
            self._path = Path(data_or_path)
            self._ext = (format or self._path.suffix.lstrip(".") or "bin").lower()
        elif isinstance(self, Html) and isinstance(data_or_path, str):
            self._text, self._ext = data_or_path, "html"
        elif hasattr(data_or_path, "read"):                       # file-like
            self._text_bytes = data_or_path.read()
            self._ext = (format or "bin").lower()
        else:
            raise TypeError(f"RunRaccoon.{type(self).__name__} needs a file path "
                            f"(in-memory arrays are not supported locally)")

    def _bind(self, files_dir: Path, key: str, step: int) -> dict:
        if self._path is not None:
            digest = sha256_file(self._path)
            target = self._target(files_dir, self._kind, key, step, digest, self._ext)
            if not target.exists():
                shutil.copyfile(self._path, target)
        else:
            payload = self._text.encode("utf-8") if self._text is not None else self._text_bytes
            digest = sha256_bytes(payload)
            target = self._target(files_dir, self._kind, key, step, digest, self._ext)
            if not target.exists():
                target.write_bytes(payload)
        ref = {"_type": f"{self._kind.rstrip('s')}-file", "path": target.relative_to(files_dir).as_posix(),
               "sha256": digest, "size": target.stat().st_size}
        if self.caption is not None:
            ref["caption"] = str(self.caption)
        return ref


class Video(_FileMedia):
    _kind = "videos"


class Audio(_FileMedia):
    _kind = "audio"


class Html(_FileMedia):
    _kind = "html"
