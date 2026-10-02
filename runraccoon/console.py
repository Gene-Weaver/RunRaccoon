"""`ConsoleCapture` - tee stdout/stderr into files/output.log, like wandb does.

Progress bars that redraw with carriage returns (tqdm, Ultralytics) are collapsed to their final
state, and ANSI color codes are stripped, so output.log stays readable.
"""
from __future__ import annotations

import re
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")


class _Tee:
    def __init__(self, stream: TextIO, sink: "ConsoleCapture"):
        self._stream, self._sink, self._line = stream, sink, ""

    def write(self, text: Any) -> int:
        n = self._stream.write(text)
        try:
            self._sink._feed(self, text if isinstance(text, str) else text.decode("utf-8", "replace"))
        except Exception:
            pass
        return n if n is not None else len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:      # isatty, fileno, encoding, buffer, ...
        return getattr(self._stream, name)


class ConsoleCapture:
    def __init__(self, path: Path, append: bool = False):
        self._fh = open(path, "a" if append else "w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        self._out, self._err = _Tee(sys.stdout, self), _Tee(sys.stderr, self)
        sys.stdout, sys.stderr = self._out, self._err

    def _feed(self, tee: _Tee, text: str) -> None:
        with self._lock:
            buf = tee._line + text
            *complete, tee._line = buf.split("\n")
            for line in complete:
                line = line.rstrip("\r").rsplit("\r", 1)[-1]
                self._fh.write(_ANSI.sub("", line) + "\n")
            # an un-terminated line that is being redrawn: keep only the latest frame
            if "\r" in tee._line:
                tee._line = tee._line.rsplit("\r", 1)[-1]

    def close(self) -> None:
        with self._lock:
            for tee in (self._out, self._err):
                if tee._line.strip():
                    self._fh.write(_ANSI.sub("", tee._line) + "\n")
                tee._line = ""
            self._fh.close()
        # Only restore if nobody else replaced the streams after us.
        if sys.stdout is self._out:
            sys.stdout = self._orig_out
        if sys.stderr is self._err:
            sys.stderr = self._orig_err
