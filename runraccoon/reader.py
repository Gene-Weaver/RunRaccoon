"""`RunData` - read a run directory back from disk (used by the plot renderer, dashboard and CLI).

Everything RunRaccoon shows is derived from the files a run writes; nothing talks to the
training process directly. That is what lets plots be re-rendered later (`runraccoon replot`)
and lets one dashboard watch runs from many processes.
"""
from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any

from runraccoon.history import read_rows
from runraccoon.paths import RunPaths


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_config_yaml(path: Path) -> dict:
    """config.yaml -> {key: value}, unwrapping wandb's `{value: ...}` envelopes."""
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ImportError, ValueError):
        return {}
    except Exception:          # yaml.YAMLError
        return {}
    return {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}


_SERIES_SUFFIX = re.compile(r"^(?P<base>.+?)[_\-](?:epoch|step|iter)[_\-]?(?P<n>\d+)$", re.I)


def media_series_key(key: str) -> tuple[str, int | None]:
    """`val/inputs_epoch9` -> ("val/inputs", 9). Scripts that put the epoch in the key name log a new
    key every epoch; folding them back into one series gives them a step slider and a GIF."""
    m = _SERIES_SUFFIX.match(key)
    return (m.group("base"), int(m.group("n"))) if m else (key, None)


class RunData:
    def __init__(self, paths: RunPaths | str | Path, rows: list[dict] | None = None):
        self.paths = paths if isinstance(paths, RunPaths) else RunPaths.from_run_dir(paths)
        if rows is not None:              # already-parsed history (the dashboard keeps its own cache)
            self.__dict__["rows"] = rows

    @cached_property
    def config_all(self) -> dict:
        return read_config_yaml(self.paths.config)

    @property
    def config(self) -> dict:
        """User config, without RunRaccoon's `_wandb` bookkeeping."""
        return {k: v for k, v in self.config_all.items() if k != "_wandb"}

    @property
    def wandb_internal(self) -> dict:
        v = self.config_all.get("_wandb")
        return v if isinstance(v, dict) else {}

    @property
    def metric_defs(self) -> list[dict]:
        return [m for m in self.wandb_internal.get("m", []) or [] if isinstance(m, dict)]

    @cached_property
    def summary(self) -> dict:
        return _read_json(self.paths.summary)

    @cached_property
    def metadata(self) -> dict:
        return _read_json(self.paths.metadata)

    @cached_property
    def rows(self) -> list[dict]:
        return read_rows(self.paths.history)

    @property
    def name(self) -> str:
        return self.wandb_internal.get("run_name") or self.paths.run_id

    @property
    def project(self) -> str | None:
        return self.wandb_internal.get("project")

    def media_refs(self) -> dict[str, list[dict[str, Any]]]:
        """{key: [{"step", "path", "caption", ...}, ...]} for every logged image. Keys that carry the
        epoch in their name (`val/preds_epoch12`) are folded into one series (`val/preds`) whose step
        is that number."""
        out: dict[str, list[dict]] = {}
        for row in self.rows:
            for raw_key, val in row.items():
                key, n = media_series_key(raw_key)
                step = n if n is not None else row.get("_step")
                if not isinstance(val, dict):
                    continue
                t = val.get("_type")
                if t == "image-file":
                    ref = {"step": step, "path": val["path"], "caption": val.get("caption"),
                           "width": val.get("width"), "height": val.get("height")}
                    if isinstance(val.get("_runraccoon_sheet"), dict):
                        ref["boxes"] = val["_runraccoon_sheet"].get("boxes")      # ContactSheet tile positions
                        ref["qc"] = val["_runraccoon_sheet"].get("qc")            # QCSheet overlay data (restylable)
                    out.setdefault(key, []).append(ref)
                elif t == "images/separated":
                    caps = val.get("captions") or [None] * len(val.get("filenames", []))
                    for i, (fn, cap) in enumerate(zip(val.get("filenames", []), caps)):
                        out.setdefault(key, []).append({"step": step, "path": fn, "caption": cap, "index": i})
        return out

    def chart_refs(self) -> dict[str, dict]:
        """Latest custom chart (table + drawing spec) per key."""
        out = {}
        for row in self.rows:
            for key, val in row.items():
                if isinstance(val, dict) and val.get("_type") == "table-file" and "_runraccoon_chart" in val:
                    out[key[:-len("_table")] if key.endswith("_table") else key] = {**val, "step": row.get("_step")}
        return out
