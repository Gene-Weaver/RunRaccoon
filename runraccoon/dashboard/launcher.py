"""Start the shared dashboard if it is not already running.

Every run calls `ensure_dashboard()` on init. The first one starts a detached server process
(`python -m runraccoon dashboard --idle-exit 3600`) that outlives it; later runs just find it.
The server exits by itself after an hour with no active runs. It binds to 127.0.0.1 only.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from runraccoon import registry

log = logging.getLogger("runraccoon")
_PKG_PARENT = str(Path(__file__).resolve().parents[2])


def ping(port: int, timeout: float = 0.4) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data if data.get("app") == "runraccoon" else None
    except Exception:
        return None


def ensure_dashboard(port: int, wait_s: float = 3.0) -> str | None:
    url = f"http://127.0.0.1:{port}/"
    if ping(port):
        return url
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_PKG_PARENT, env.get("PYTHONPATH", "")) if p)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    logfile = registry.home() / "dashboard.log"
    try:
        with open(logfile, "a", encoding="utf-8") as fh:
            subprocess.Popen([sys.executable, "-m", "runraccoon", "dashboard", "--port", str(port),
                              "--idle-exit", "3600", "--no-browser"],
                             stdout=fh, stderr=fh, stdin=subprocess.DEVNULL, env=env, close_fds=True,
                             start_new_session=True, cwd=str(registry.home()))
    except OSError as e:
        log.warning("could not start the dashboard: %s", e)
        return None
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if ping(port, timeout=0.2):
            return url
        time.sleep(0.15)
    log.warning("dashboard did not start on port %d (port in use? see %s); "
                "set RUNRACCOON_PORT to another port or run `runraccoon dashboard`", port, logfile)
    return None
