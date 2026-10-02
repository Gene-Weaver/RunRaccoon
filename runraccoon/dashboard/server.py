"""The dashboard server: standard library only, read-only, bound to 127.0.0.1.

    GET /                              the single-page app (static/index.html)
    GET /api/ping                      {"app": "runraccoon", "version": ...}
    GET /api/runs                      every registered run with its live status
    GET /api/runs/<id>                 config, summary, metadata, list of rendered plots
    GET /api/runs/<id>/panels?v=<n>    chart panels (same grouping as the PNGs); {"unchanged": true} if v is current
    GET /api/runs/<id>/media           logged images per key, per step
    GET /api/runs/<id>/log?lines=400   tail of output.log
    GET /files/<id>/<path>             a file from the run's files/ folder (images, plots)
    GET  /api/runs/<id>/gifs/config    GIF settings, overlay style, color features, job status
    GET  /api/runs/<id>/gifs/preview   one frame rendered with ?options=<json> (live preview)
    POST /api/runs/<id>/gifs/render    regenerate the GIFs with {"options": {...}, "style": {...}}
    POST /api/runs/<id>/prefs          {"starred": bool, "hidden": bool} (stored in the run index)

The only write action is GIF regeneration (into the run's plots/gifs). It requires the custom
header `X-RunRaccoon: 1` and a localhost Host header, so other web pages cannot trigger it.
"""
from __future__ import annotations

import io
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
from runraccoon.panels import best_checkpoint, build_sections, estimate_end, training_rows, ultralytics_version, x_label_hint
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

    def estimated_end(self) -> float | None:
        self.refresh()
        cfg = read_config_yaml(self.paths.config)
        with self.lock:
            return estimate_end(self.rows, cfg.get("epochs"))

    def panels(self) -> dict:
        version = self.refresh()
        with self.lock:
            if self.panels_json is None or self.panels_version != version:
                cfg = read_config_yaml(self.paths.config)
                internal = cfg.get("_wandb") if isinstance(cfg.get("_wandb"), dict) else {}
                user_cfg = {k: v for k, v in cfg.items() if k != "_wandb"}
                hint = x_label_hint(self.rows, user_cfg)
                sections = build_sections(training_rows(self.rows, user_cfg), internal.get("m") or [], live=False,
                                          x_label_hint=hint)
                try:
                    reqs = self.paths.requirements.read_text(encoding="utf-8")
                except OSError:
                    reqs = ""
                best = best_checkpoint(self.rows, internal.get("m") or [], ultralytics_version(reqs))
                best_row = best["row"] if best else None
                self.panels_json = {"version": version, "sections": [
                    {"name": s.name, "panels": [p.to_json(MAX_POINTS, best_row) for p in s.panels]} for s in sections]}
                if best:
                    where = (f"epoch {best_row['epoch']}" if isinstance(best_row.get("epoch"), (int, float))
                             else f"{hint or 'step'} {best_row.get('_step')}")
                    self.panels_json["checkpoint"] = {"step": best_row.get("_step"), "where": where,
                                                      "value": best["value"], "rule": best["rule"],
                                                      "source": best["source"]}
                self.panels_version = version
            return self.panels_json


class Dashboard:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, idle_exit_s: float | None = None):
        self.host, self.port, self.idle_exit_s = host, port, idle_exit_s
        self._caches: dict[str, _RunCache] = {}
        self._lock = threading.Lock()
        self.gif_jobs: dict[str, dict] = {}
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

    def do_POST(self):
        try:
            host = (self.headers.get("Host") or "").split(":")[0]
            if self.headers.get("X-RunRaccoon") != "1" or host not in ("127.0.0.1", "localhost", "::1", "[::1]"):
                return self._json({"error": "forbidden"}, 403)
            parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "prefs":
                length = min(int(self.headers.get("Content-Length") or 0), 10_000)
                body = json.loads(self.rfile.read(length) or b"{}")
                if registry.read(parts[2]) is None:
                    return self._json({"error": "unknown run"}, 404)
                prefs = {k: bool(body[k]) for k in ("starred", "hidden") if k in body}
                registry.update(parts[2], **prefs)
                return self._json({"ok": True, **prefs})
            if len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3:] == ["gifs", "render"]:
                length = min(int(self.headers.get("Content-Length") or 0), 1_000_000)
                body = json.loads(self.rfile.read(length) or b"{}")
                return self._gif_render(parts[2], body)
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pragma: no cover
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    # --------------------------------------------------------------------------- GIF control
    def _gif_options(self, cache: "_RunCache") -> dict:
        from runraccoon.plotting.gifs import saved_options
        return vars(saved_options(RunData(cache.paths)))

    def _gif_sources(self, cache: "_RunCache"):
        """(animatable keys, latest QC doc per restylable key)."""
        cache.refresh()
        with cache.lock:
            rows = list(cache.rows)
        refs = RunData(cache.paths, rows=rows).media_refs()
        keys, docs = [], {}
        for key, items in refs.items():
            if len({r["step"] for r in items}) >= 2:
                keys.append(key)
                last = max(items, key=lambda r: r["step"])
                if last.get("qc") and all(r.get("qc") for r in items):
                    try:
                        docs[key] = json.loads((cache.paths.files / last["qc"]).read_text())
                    except (OSError, ValueError):
                        pass
        return keys, docs

    def _gif_config(self, run_id: str, cache: "_RunCache") -> None:
        from runraccoon.qc import qc_features, resolve_style
        opts = self._gif_options(cache)
        keys, docs = self._gif_sources(cache)
        feats, seen = [], set()
        for doc in docs.values():
            for f in qc_features(doc.get("meta", {}), doc.get("tiles", [])):
                if f["id"] not in seen:
                    seen.add(f["id"])
                    feats.append(f)
        style = resolve_style(opts.get("style"))
        for f in feats:                                  # show the user's chosen colors, not the defaults
            f["color"] = style["colors"].get(f["id"], f["color"])
        self._json({"options": opts, "style": style, "features": feats, "keys": keys,
                    "restylable": sorted(docs), "job": self.app.gif_jobs.get(run_id)})

    def _gif_preview(self, cache: "_RunCache", q: dict) -> None:
        from PIL import Image as PILImage

        from runraccoon.plotting.gifs import GifOptions, compose_frame
        try:
            req = json.loads(q.get("options", ["{}"])[0])
        except ValueError:
            req = {}
        opts = GifOptions.from_dict({**self._gif_options(cache), **req})
        keys, docs = self._gif_sources(cache)
        W, H = opts.canvas
        if docs:
            key = sorted(docs)[0]
            doc = docs[key]
            panel = 0 if opts.panels and len(doc.get("tiles", [])) else None
            from runraccoon.qc import draw_overlays, render_qc_sheet
            if panel is None:
                img, _ = render_qc_sheet(doc, cache.paths.files, opts.style)
            else:
                tile = doc["tiles"][0]
                base = PILImage.open(cache.paths.files / tile["base"]).convert("RGB")
                k = min(W * 0.95 / base.width, H * 0.85 / base.height)
                base = base.resize((max(1, round(base.width * k)), max(1, round(base.height * k))), PILImage.Resampling.LANCZOS)
                img = draw_overlays(base, tile, doc.get("meta", {}), opts.style, cache.paths.files)
            title = f"{key} - panel 1" if panel is not None else key
            step = doc.get("step")
        elif keys:
            refs = RunData(cache.paths).media_refs()[keys[0]]
            last = max(refs, key=lambda r: r["step"])
            img, title, step = PILImage.open(cache.paths.files / last["path"]), keys[0], last["step"]
        else:
            return self._json({"error": "nothing to preview"}, 404)
        bg = tuple(int(c) for c in img.convert("RGB").getpixel((0, 0)))
        frame = compose_frame(img, title, f"step {step}", 1.0, opts, bg)
        frame.thumbnail((960, 960))
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        self._send(200, buf.getvalue(), "image/png")

    def _gif_render(self, run_id: str, body: dict) -> None:
        cache = self.app.cache(run_id)
        if cache is None:
            return self._json({"error": "unknown run"}, 404)
        job = self.app.gif_jobs.get(run_id)
        if job and job.get("state") == "running":
            return self._json({"job": job}, 409)
        allowed = {"seconds_per_frame", "orientation", "resolution", "hold_last_s", "panels", "label", "keys",
                   "max_frames", "min_frames"}
        options = {k: v for k, v in (body.get("options") or {}).items() if k in allowed}
        options["style"] = body.get("style") or None
        options["resolution"] = int(max(360, min(2160, int(options.get("resolution") or 1080))))
        options["seconds_per_frame"] = float(max(0.02, min(5.0, float(options.get("seconds_per_frame") or 0.25))))
        merged = {**self._gif_options(cache), **options}
        gif_dir = cache.paths.plots / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        (gif_dir / "options.json").write_text(json.dumps(merged, indent=2))
        job = {"state": "running", "done": 0, "total": 0, "started": time.time(), "error": None}
        self.app.gif_jobs[run_id] = job

        def work():
            from runraccoon.panels import x_label_hint
            from runraccoon.plotting.gifs import GifOptions, render_gifs
            try:
                data = RunData(cache.paths)
                written = render_gifs(data, GifOptions.from_dict(merged), x_label_hint(data.rows, data.config) or "step",
                                      progress=lambda d, t: job.update(done=d, total=t))
                job.update(state="done", written=len(written), finished=time.time())
            except Exception as e:
                job.update(state="error", error=f"{type(e).__name__}: {e}")

        threading.Thread(target=work, name=f"gifs-{run_id}", daemon=True).start()
        return self._json({"job": job}, 202)

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
            runs = []
            for r in registry.listing():
                pub = _public(r)
                if r["status"] in ("running", "unresponsive"):
                    try:
                        c = self.app.cache(r["id"])
                        pub["est_end"] = c.estimated_end() if c else None
                    except Exception:
                        pub["est_end"] = None
                else:
                    pub["ended"] = r.get("finished") or r.get("heartbeat")
                runs.append(pub)
            return self._json({"runs": runs, "now": time.time()})
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
            figures = [{"path": p.relative_to(cache.paths.files).as_posix(), "mtime": p.stat().st_mtime,
                        "size": p.stat().st_size}
                       for p in cache.paths.plots.rglob("*") if not p.name.startswith(".")
                       and p.suffix.lower() in (".png", ".svg", ".jpg", ".gif")]
            plots = sorted((f for f in figures if not f["path"].endswith(".gif")),
                           key=lambda d: (not d["path"].endswith("run_summary.png"), d["path"] != "plots/progress.png", d["path"]))
            # full-image GIFs first, then each image's panel GIFs in order
            gifs = sorted((f for f in figures if f["path"].endswith(".gif")),
                          key=lambda d: (d["path"].rsplit("/", 1)[0] if "/panel_" in d["path"] else d["path"][:-4],
                                         "/panel_" in d["path"], d["path"]))
            return self._json({"run": _public(rec), "config": data.config, "summary": summary,
                               "internal": data.wandb_internal, "plots": plots, "gifs": gifs,
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
        if rest == ["gifs", "config"]:
            return self._gif_config(run_id, cache)
        if rest == ["gifs", "preview"]:
            return self._gif_preview(cache, q)
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
            "heartbeat", "step", "runtime", "run_dir", "exit_code", "host", "pid", "resumed", "program",
            "starred", "hidden")
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
