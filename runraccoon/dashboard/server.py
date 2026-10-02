"""The dashboard server: standard library only, read-only, bound to 127.0.0.1.

    GET /                              the single-page app (static/index.html)
    GET /api/ping                      {"app": "runraccoon", "version": ...}
    GET /api/runs                      every registered run with its live status
    GET /api/runs/<id>                 config, summary, metadata, list of rendered plots
    GET /api/runs/<id>/panels?v=<n>    chart panels (same grouping as the PNGs); {"unchanged": true} if v is current
    GET /api/runs/<id>/media           logged images per key, per step
    GET /api/runs/<id>/log?lines=400   tail of output.log
    GET /files/<id>/<path>             a file from the run's files/ folder (images, plots)
"""
from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from runraccoon import registry
from runraccoon._version import __version__
from runraccoon.history import iter_rows, merge_rows
from runraccoon.panels import build_sections, x_label_hint
from runraccoon.paths import RunPaths
from runraccoon.reader import RunData, read_config_yaml
from runraccoon.settings import DEFAULT_PORT
from runraccoon.utils import json_safe

STATIC = Path(__file__).with_name("static")
MAX_POINTS = 1500


class _RunCache:
    """Incrementally parsed history for one run directory."""

    def __init__(self, run_dir: Path):
        self.paths = RunPaths.from_run_dir(run_dir)
        self.offset = 0
        self.raw: list[dict] = []
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        self.panels_json: dict | None = None
        self.panels_version = -1

    def refresh(self) -> int:
        with self.lock:
            new = False
            for row, end in iter_rows(self.paths.history, self.offset):
                self.raw.append(row)
                self.offset = end
                new = True
            if new or not self.rows:
                self.rows = merge_rows(self.raw)
            return self.offset

    def panels(self) -> dict:
        version = self.refresh()
        with self.lock:
            if self.panels_json is None or self.panels_version != version:
                cfg = read_config_yaml(self.paths.config)
                internal = cfg.get("_wandb") if isinstance(cfg.get("_wandb"), dict) else {}
                user_cfg = {k: v for k, v in cfg.items() if k != "_wandb"}
                sections = build_sections(self.rows, internal.get("m") or [], live=False,
                                          x_label_hint=x_label_hint(self.rows, user_cfg))
                self.panels_json = {"version": version, "sections": [
                    {"name": s.name, "panels": [p.to_json(MAX_POINTS) for p in s.panels]} for s in sections]}
                self.panels_version = version
            return self.panels_json


class Dashboard:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, idle_exit_s: float | None = None):
        self.host, self.port, self.idle_exit_s = host, port, idle_exit_s
        self._caches: dict[str, _RunCache] = {}
        self._lock = threading.Lock()
        self.httpd: ThreadingHTTPServer | None = None
        self._last_active = time.time()

    def cache(self, run_id: str) -> _RunCache | None:
        rec = registry.read(run_id)
        if not rec or not rec.get("run_dir"):
            return None
        with self._lock:
            c = self._caches.get(run_id)
            if c is None or str(c.paths.run_dir) != str(Path(rec["run_dir"]).resolve()):
                c = self._caches[run_id] = _RunCache(Path(rec["run_dir"]))
            return c

    def serve(self) -> None:
        dashboard = self

        class Handler(_Handler):
            app = dashboard

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        if self.idle_exit_s:
            threading.Thread(target=self._idle_watch, daemon=True).start()
        self.httpd.serve_forever(poll_interval=0.5)

    def shutdown(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def _idle_watch(self) -> None:
        while True:
            time.sleep(30)
            if any(r["status"] in ("running", "unresponsive") for r in registry.listing()):
                self._last_active = time.time()
            elif time.time() - self._last_active > self.idle_exit_s:
                print("no active runs; dashboard exiting", flush=True)
                threading.Thread(target=self.shutdown, daemon=True).start()
                return


class _Handler(BaseHTTPRequestHandler):
    app: Dashboard
    server_version = f"RunRaccoon/{__version__}"

    def log_message(self, fmt, *args):      # keep the console quiet
        pass

    # ----------------------------------------------------------------------------- plumbing
    def _send(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(json_safe(obj)).encode("utf-8"), "application/json")

    def _file(self, path: Path, cache: str = "no-cache") -> None:
        try:
            body = path.read_bytes()
        except OSError:
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, body, ctype, cache)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pragma: no cover
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except Exception:
                pass

    # ------------------------------------------------------------------------------- routes
    def _route(self) -> None:
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        q = parse_qs(url.query)

        if not parts:
            return self._file(STATIC / "index.html")
        if parts[0] == "static" and len(parts) == 2 and "/" not in parts[1] and parts[1] != "..":
            return self._file(STATIC / parts[1])
        if parts[0] == "favicon.ico":
            return self._file(STATIC / "favicon.svg")
        if parts == ["api", "ping"]:
            return self._json({"app": "runraccoon", "version": __version__, "pid": os.getpid()})
        if parts == ["api", "runs"]:
            return self._json({"runs": [_public(r) for r in registry.listing()], "now": time.time()})
        if parts[:2] == ["api", "runs"] and len(parts) >= 3:
            return self._run_api(parts[2], parts[3:], q)
        if parts[0] == "files" and len(parts) >= 3:
            return self._run_file(parts[1], parts[2:])
        return self._json({"error": "not found"}, 404)

    def _run_api(self, run_id: str, rest: list[str], q: dict) -> None:
        cache = self.app.cache(run_id)
        if cache is None:
            return self._json({"error": f"unknown run {run_id}"}, 404)
        if not rest:
            rec = registry.read(run_id) or {}
            rec["status"] = registry.status(rec)
            data = RunData(cache.paths)
            summary = {k: v for k, v in data.summary.items() if not isinstance(v, dict)}
            meta = data.metadata
            plots = sorted(
                ({"path": p.relative_to(cache.paths.files).as_posix(), "mtime": p.stat().st_mtime}
                 for p in cache.paths.plots.rglob("*") if p.suffix.lower() in (".png", ".svg", ".jpg")),
                key=lambda d: (not d["path"].endswith("run_summary.png"), d["path"] != "plots/progress.png", d["path"]))
            return self._json({"run": _public(rec), "config": data.config, "summary": summary,
                               "internal": data.wandb_internal, "plots": plots,
                               "metadata": {k: meta.get(k) for k in ("host", "program", "args", "python", "gpu",
                                                                      "gpu_count", "git", "startedAt", "executable")
                                            if k in meta}})
        if rest == ["panels"]:
            have = q.get("v", [None])[0]
            version = cache.refresh()
            if have is not None and str(version) == have:
                return self._json({"version": version, "unchanged": True})
            return self._json(cache.panels())
        if rest == ["media"]:
            cache.refresh()
            with cache.lock:
                rows = list(cache.rows)
            return self._json({"media": RunData(cache.paths, rows=rows).media_refs()})
        if rest == ["log"]:
            n = int(q.get("lines", ["400"])[0])
            try:
                with open(cache.paths.output_log, encoding="utf-8", errors="replace") as fh:
                    lines = list(deque(fh, maxlen=max(1, min(n, 5000))))
            except OSError:
                lines = []
            return self._json({"lines": [ln.rstrip("\n") for ln in lines]})
        return self._json({"error": "not found"}, 404)

    def _run_file(self, run_id: str, rel: list[str]) -> None:
        cache = self.app.cache(run_id)
        if cache is None:
            return self._json({"error": "unknown run"}, 404)
        base = cache.paths.files.resolve()
        target = (base / "/".join(rel)).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return self._json({"error": "forbidden"}, 403)
        if not target.is_file():
            return self._json({"error": "not found"}, 404)
        return self._file(target, cache="max-age=60" if "/media/" in target.as_posix() else "no-cache")


def _public(rec: dict) -> dict:
    keys = ("id", "name", "project", "group", "job_type", "tags", "status", "state", "started", "finished",
            "heartbeat", "step", "runtime", "run_dir", "exit_code", "host", "pid", "resumed", "program")
    return {k: rec.get(k) for k in keys}


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT, idle_exit_s: float | None = None, open_browser: bool = True) -> None:
    app = Dashboard(host, port, idle_exit_s)
    url = f"http://{host}:{port}/"
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    print(f"RunRaccoon dashboard on {url}  (Ctrl+C to stop)", flush=True)
    try:
        app.serve()
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f"could not bind {host}:{port}: {e}", flush=True)
        raise SystemExit(1)
