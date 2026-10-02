"""`PlotScheduler` - decides when to re-render the live progress figure, and runs the renderer
in a subprocess so training never waits on matplotlib.

A re-render is requested when a committed row contains a plottable number and either the logged
`epoch` changed, or (when no `epoch` is logged) a new step was committed. Requests are coalesced:
at most one render runs at a time, and at most one per `plot_every_s` seconds. `finish()` waits
for the final render so all figures exist when the training script exits.

Spawning is fork-safe. Training frameworks fork worker processes (PyTorch DataLoader) from the
main thread at any moment. `subprocess.Popen` from a background thread opens an internal pipe
and blocks until it closes; a worker forked during that window inherits the pipe and keeps it
open for its whole life, which blocks the thread forever (and anything waiting on it).
`os.posix_spawn` uses no such pipe, so the renderer is started with it, outside any lock.
"""
from __future__ import annotations

import logging
import os
import signal
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


class _Child:
    """A spawned process: os.posix_spawn where available (fork-race free), Popen otherwise."""

    def __init__(self, argv: list[str], log_path: Path):
        self.pid = None
        self.proc = None
        self.returncode = None
        if hasattr(os, "posix_spawn"):
            actions = [(os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
                       (os.POSIX_SPAWN_OPEN, 1, str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644),
                       (os.POSIX_SPAWN_DUP2, 1, 2)]
            self.pid = os.posix_spawn(argv[0], argv, _env(), file_actions=actions)
        else:                                                  # pragma: no cover - Windows
            with open(log_path, "a", encoding="utf-8") as fh:
                self.proc = subprocess.Popen(argv, stdout=fh, stderr=fh, stdin=subprocess.DEVNULL, env=_env())

    def poll(self) -> int | None:
        """Exit code if finished (reaping it), else None. Never blocks."""
        if self.returncode is not None:
            return self.returncode
        if self.proc is not None:
            self.returncode = self.proc.poll()
            return self.returncode
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            self.returncode = 0
            return self.returncode
        if pid:
            self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout: float) -> int | None:
        deadline = time.time() + timeout
        while self.poll() is None:
            if time.time() > deadline:
                return None
            time.sleep(0.1)
        return self.returncode

    def kill(self) -> None:
        try:
            if self.proc is not None:
                self.proc.kill()
            elif self.pid:
                os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass
        self.wait(5)


class PlotScheduler:
    def __init__(self, run_dir: Path, log_path: Path, every_s: float, enabled: bool):
        self.run_dir, self.log_path, self.every_s, self.enabled = run_dir, log_path, every_s, enabled
        self._dirty = False
        self._last_epoch = object()
        self._last_start = 0.0
        self._child: _Child | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()                  # guards state only; never held while spawning or waiting
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

    def _spawn(self, final: bool = False, formats: tuple = ("png",), status: str | None = None) -> _Child:
        return _Child(_render_cmd(self.run_dir, final, formats, status), self.log_path)

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                if self._child is not None and self._child.poll() is None:
                    continue
                self._child = None
                due = self._dirty and time.time() - self._last_start >= self.every_s
                if due:
                    self._dirty = False
                    self._last_start = time.time()
            if due and not self._stop.is_set():
                try:
                    child = self._spawn()
                except OSError as e:
                    log.warning("could not start the plot renderer: %s", e)
                    continue
                with self._lock:
                    self._child = child

    def finish(self, final: bool, formats: tuple, status: str, timeout: float) -> bool:
        """Stop live rendering; optionally run the final render and wait for it."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            child, self._child = self._child, None
        if child is not None and child.wait(min(60.0, timeout)) is None:
            child.kill()
        if not final:
            return True
        try:
            proc = self._spawn(final=True, formats=formats, status=status)
        except OSError as e:
            log.warning("could not run the final plot renderer: %s", e)
            return False
        code = proc.wait(timeout)
        if code is None:
            proc.kill()
            log.warning("final plot rendering timed out after %.0fs", timeout)
            return False
        return code == 0
