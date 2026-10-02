"""`PlotScheduler` - decides when to re-render the live progress figure, and runs the renderer
in a subprocess so training never waits on matplotlib.

A re-render is requested when a committed row contains a plottable number and either the logged
`epoch` changed, or (when no `epoch` is logged) a new step was committed. Requests are coalesced:
at most one render runs at a time, and at most one per `plot_every_s` seconds. `finish()` waits
for the final render so all figures exist when the training script exits.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("runraccoon")
_PKG_PARENT = str(Path(__file__).resolve().parents[2])


def _render_cmd(run_dir: Path, final: bool, formats: tuple, status: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "runraccoon", "render", str(run_dir)]
    if final:
        cmd += ["--final", "--formats", ",".join(formats)]
    if status:
        cmd += ["--status", status]
    return cmd


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_PKG_PARENT, env.get("PYTHONPATH", "")) if p)
    env["MPLBACKEND"] = "Agg"
    env.pop("RUNRACCOON_SHIM", None)
    return env


class PlotScheduler:
    def __init__(self, run_dir: Path, log_path: Path, every_s: float, enabled: bool):
        self.run_dir, self.log_path, self.every_s, self.enabled = run_dir, log_path, every_s, enabled
        self._dirty = False
        self._last_epoch = object()
        self._last_start = 0.0
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="runraccoon-plots", daemon=True)
        if enabled:
            self._thread.start()

    def notify(self, row: dict) -> None:
        """Called after every committed history row."""
        if not self.enabled:
            return
        if not any(not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool)
                   for k, v in row.items()):
            return
        epoch = row.get("epoch")
        if epoch is not None:
            if epoch == self._last_epoch:
                return
            self._last_epoch = epoch
        self._dirty = True

    def _spawn(self, final: bool = False, formats: tuple = ("png",), status: str | None = None):
        with open(self.log_path, "a", encoding="utf-8") as fh:
            return subprocess.Popen(_render_cmd(self.run_dir, final, formats, status), stdout=fh, stderr=fh,
                                    stdin=subprocess.DEVNULL, env=_env(), close_fds=True)

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                if self._proc is not None and self._proc.poll() is None:
                    continue
                self._proc = None
                if self._dirty and time.time() - self._last_start >= self.every_s:
                    self._dirty = False
                    self._last_start = time.time()
                    try:
                        self._proc = self._spawn()
                    except OSError as e:
                        log.warning("could not start the plot renderer: %s", e)

    def finish(self, final: bool, formats: tuple, status: str, timeout: float) -> bool:
        """Stop live rendering; optionally run the final render and wait for it."""
        self._stop.set()
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.wait(timeout=min(60.0, timeout))
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
            if not final:
                return True
            try:
                proc = self._spawn(final=True, formats=formats, status=status)
                proc.wait(timeout=timeout)
                return proc.returncode == 0
            except subprocess.TimeoutExpired:
                proc.kill()
                log.warning("final plot rendering timed out after %.0fs", timeout)
            except OSError as e:
                log.warning("could not run the final plot renderer: %s", e)
            return False
