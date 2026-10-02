"""Machine-wide index of runs, so one dashboard can show every run - live and past.

Each run owns one small JSON file:  ~/.runraccoon/runs/<run_id>.json   (RUNRACCOON_HOME to move it)

    {"id", "name", "project", "group", "run_dir", "pid", "host", "started", "heartbeat",
     "state": "running" | "finished", "exit_code", "step", ...}

A live run refreshes `heartbeat` every few seconds. `status()` turns a record into what the
dashboard shows: running / finished / failed / crashed (process gone without finishing) /
unresponsive (process alive but silent).
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from runraccoon.utils import atomic_write_json

HEARTBEAT_EVERY_S = 5.0
STALE_AFTER_S = 120.0


def home() -> Path:
    return Path(os.environ.get("RUNRACCOON_HOME", Path.home() / ".runraccoon")).expanduser()


def runs_dir() -> Path:
    d = home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_path(run_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    return runs_dir() / f"{safe}.json"


def write(record: dict) -> None:
    try:
        atomic_write_json(_record_path(record["id"]), record, indent=1)
    except OSError:
        pass   # the registry is a convenience; never break a training run over it


def update(run_id: str, **fields: Any) -> None:
    rec = read(run_id) or {"id": run_id}
    rec.update(fields)
    write(rec)


def read(run_id: str) -> dict | None:
    try:
        return json.loads(_record_path(run_id).read_text())
    except (OSError, ValueError):
        return None


def forget(run_id: str) -> bool:
    try:
        _record_path(run_id).unlink()
        return True
    except OSError:
        return False


def all_records() -> list[dict]:
    out = []
    for p in runs_dir().glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def status(rec: dict, now: float | None = None) -> str:
    now = time.time() if now is None else now
    if rec.get("state") == "finished":
        return "finished" if not rec.get("exit_code") else "failed"
    if not Path(rec.get("run_dir", "")).exists():
        return "missing"
    if rec.get("host") not in (None, socket.gethostname()):
        # Another machine (e.g. a shared Dropbox folder): trust only the heartbeat.
        return "running" if now - rec.get("heartbeat", 0) < STALE_AFTER_S else "unknown"
    pid = rec.get("pid")
    if not pid or not _pid_alive(int(pid)):
        return "crashed"
    return "running" if now - rec.get("heartbeat", 0) < STALE_AFTER_S else "unresponsive"


def listing() -> list[dict]:
    """All runs with a computed `status`, active first, then newest first."""
    now = time.time()
    recs = []
    for r in all_records():
        r = dict(r)
        r["status"] = status(r, now)
        recs.append(r)
    recs.sort(key=lambda r: (r["status"] not in ("running", "unresponsive"), -float(r.get("started", 0))))
    return recs


def register_existing(run_dir: Path) -> dict | None:
    """Add a run directory written earlier (or moved) to the index."""
    from runraccoon.paths import RunPaths
    from runraccoon.reader import RunData

    paths = RunPaths.from_run_dir(run_dir)
    if not paths.history.exists() and not paths.summary.exists():
        return None
    data = RunData(paths)
    meta = data.metadata
    cfg_wandb = data.wandb_internal
    rec = {"id": paths.run_id, "name": cfg_wandb.get("run_name") or paths.run_id,
           "project": cfg_wandb.get("project"), "group": cfg_wandb.get("group"), "run_dir": str(paths.run_dir),
           "host": meta.get("host"), "pid": None, "started": paths.run_dir.stat().st_mtime,
           "heartbeat": 0, "state": "finished", "exit_code": data.summary.get("_runraccoon_exit_code", 0),
           "step": data.summary.get("_step")}
    write(rec)
    return rec
