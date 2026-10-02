"""`Run` - one training run. Created by `runraccoon.init()`; mirrors `wandb.sdk.wandb_run.Run`.

Step semantics follow wandb:
  * `log(d)`                  merges `d` into the current step and commits it (step += 1)
  * `log(d, commit=False)`    merges `d` into the current step, keeps it open
  * `log(d, step=n)`          if n > current: commits the open step, then opens step n.
                              if n == current: merges into it. Committed by a later step,
                              `commit=True`, or `finish()`.
  * a step that was already committed (n < current) is still recorded (wandb would drop it):
    it is appended as an extra row with that `_step`, and readers merge it.
"""
from __future__ import annotations

import fnmatch
import logging
import math
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from runraccoon import registry
from runraccoon._version import __version__
from runraccoon.config import Config
from runraccoon.history import HistoryWriter, iter_rows
from runraccoon.media import Histogram, Image, Media, bind_image_list
from runraccoon.paths import RunPaths
from runraccoon.plot import CustomChart
from runraccoon.settings import Settings
from runraccoon.summary import Summary
from runraccoon.utils import as_number, atomic_write_json, atomic_write_text, format_duration, format_value, to_builtin

log = logging.getLogger("runraccoon")


def _say(settings: Settings, msg: str) -> None:
    if not settings.quiet:
        tag = "\033[1;38;5;245mrunraccoon\033[0m" if sys.__stderr__ and sys.__stderr__.isatty() else "runraccoon"
        print(f"{tag}: {msg}", file=sys.stderr, flush=True)


class Run:
    """A local run. Use `runraccoon.init()` rather than constructing this directly."""

    def __init__(self, *, project: str, name: str, run_id: str, base_dir: str, group: str | None = None,
                 job_type: str | None = None, tags: Sequence[str] = (), notes: str | None = None,
                 config: Any = None, settings: Settings | None = None, disabled: bool = False,
                 resume_from: Path | None = None):
        self.settings = settings or Settings()
        self.id, self.name, self.project = run_id, name, project
        self.entity = "local"
        self.group, self.job_type, self.tags, self.notes = group, job_type, tuple(tags or ()), notes
        self.disabled = disabled
        self.resumed = resume_from is not None
        self.start_time = time.time()
        self._runtime_offset = 0.0
        self._lock = threading.RLock()
        self._step = 0
        self._pending: dict[str, Any] = {}
        self._metric_defs: list[dict] = []
        self._finished = False
        self._exit_code: int | None = None
        self._last_summary_write = 0.0
        self._console = None
        self._history: HistoryWriter | None = None
        self._scheduler = None
        self._heartbeat_stop = threading.Event()
        self._log_handler: logging.Handler | None = None
        self._dashboard_url: str | None = None

        self.config = Config()
        self.summary = Summary()
        if disabled:
            self.paths = None
            self.config.update(config or {})
            return

        from runraccoon.utils import timestamp_slug
        self.paths = RunPaths.create(base_dir, self.settings.dirname, timestamp_slug(self.start_time), run_id)
        self._setup_debug_log()
        log.info("RunRaccoon %s: run %s (%s/%s) in %s", __version__, run_id, project, name, self.paths.run_dir)

        if resume_from is not None:
            self._load_previous(resume_from)
        self.config.update(config or {})
        self.config._attach(self._write_config)
        self.summary._attach(self._summary_changed)
        self._write_config()

        self._history = HistoryWriter(self.paths.history, append=resume_from is not None)
        if self.settings.console != "off":
            from runraccoon.console import ConsoleCapture
            self._console = ConsoleCapture(self.paths.output_log)
        self.paths.link_latest()
        self._metadata_thread = threading.Thread(target=self._write_metadata, name="runraccoon-metadata", daemon=True)
        self._metadata_thread.start()

        from runraccoon.plotting.scheduler import PlotScheduler
        self._scheduler = PlotScheduler(self.paths.run_dir, self.paths.logs / "plots.log",
                                        self.settings.plot_every_s, self.settings.live_plots)
        self._register()
        threading.Thread(target=self._heartbeat, name="runraccoon-heartbeat", daemon=True).start()
        if self.settings.dashboard:
            from runraccoon.dashboard.launcher import ensure_dashboard
            self._dashboard_url = ensure_dashboard(self.settings.dashboard_port)
        verb = "resumed" if self.resumed else "started"
        _say(self.settings, f"{verb} run {name} ({run_id}) -> {self.paths.run_dir}")
        if self._dashboard_url:
            _say(self.settings, f"live dashboard -> {self.url}")

    # =================================================================== wandb-compatible API
    @property
    def dir(self) -> str:
        """The run's files/ directory (wandb.run.dir)."""
        return str(self.paths.files) if self.paths else os.getcwd()

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.id}"

    @property
    def url(self) -> str | None:
        if self._dashboard_url:
            return f"{self._dashboard_url}#run={self.id}"
        return self.paths.run_dir.as_uri() if self.paths else None

    @property
    def step(self) -> int:
        return self._step

    @property
    def offline(self) -> bool:
        return True

    @property
    def sweep_id(self) -> None:
        return None

    @property
    def project_name(self) -> str:
        return self.project

    def log(self, data: Mapping[str, Any], step: int | None = None, commit: bool | None = None,
            sync: bool | None = None) -> None:
        if self._finished:
            log.warning("log() called after finish(); ignored")
            return
        if not isinstance(data, Mapping):
            raise TypeError(f"run.log() expects a dict, got {type(data).__name__}")
        with self._lock:
            if step is not None:
                step = int(step)
                if step < self._step:
                    if not self.disabled:
                        self._write_row(self._process(data, step), step, late=True)
                    return
                if step > self._step:
                    self._commit()
                    self._step = step
            if self.disabled:
                if commit or (commit is None and step is None):
                    self._step += 1
                return
            self._pending.update(self._process(data, self._step))
            if commit or (commit is None and step is None):
                self._commit()

    def define_metric(self, name: str, step_metric: str | None = None, step_sync: bool | None = None,
                      hidden: bool | None = None, summary: str | None = None, goal: str | None = None,
                      overwrite: bool | None = None, **_: Any) -> dict:
        """Choose a metric's x-axis (`step_metric`), summary ("min"/"max"/"mean"/"last"/"first"/"none"),
        whether it is plotted (`hidden`), and which direction is better (`goal`: "minimize"/"maximize")."""
        spec = {"name": name}
        if step_metric:
            spec["step_metric"] = step_metric
        if hidden:
            spec["hidden"] = True
        if summary:
            spec["summary"] = summary.split(",")[0].strip()
        if goal:
            spec["goal"] = {"minimize": "min", "maximize": "max"}.get(goal, goal)
        elif summary in ("min", "max"):
            spec["goal"] = summary
        with self._lock:
            self._metric_defs = [d for d in self._metric_defs if d["name"] != name] + [spec]
        if not self.disabled:
            self._write_config()
        return spec

    def save(self, glob_str: str | os.PathLike, base_path: str | None = None, policy: str = "live") -> list[str]:
        """Copy file(s) matching `glob_str` into the run's files/ folder."""
        if self.disabled:
            return []
        import glob as _glob
        pattern = os.fspath(glob_str)
        saved = []
        for src in _glob.glob(pattern, recursive=True):
            if not os.path.isfile(src):
                continue
            rel = os.path.relpath(src, base_path) if base_path else os.path.basename(src)
            dst = self.paths.files / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            saved.append(str(dst))
        return saved

    def log_artifact(self, artifact_or_path: Any, name: str | None = None, type: str | None = None,
                     aliases: Sequence[str] | None = None, tags: Sequence[str] | None = None) -> Any:
        from runraccoon.artifacts import Artifact
        art = artifact_or_path
        if not isinstance(art, Artifact):
            p = Path(artifact_or_path)
            art = Artifact(name or p.name, type or "unspecified")
            art.add_dir(p) if p.is_dir() else art.add_file(p)
        if not self.disabled:
            where = art._save(self.paths.artifacts, copy=self.settings.artifact_copy, aliases=list(aliases or []))
            log.info("artifact %s (%s) -> %s", art.name, art.type, where)
        return art

    def log_model(self, path: str | os.PathLike, name: str | None = None, aliases: Sequence[str] | None = None) -> Any:
        return self.log_artifact(path, name=name, type="model", aliases=aliases)

    def use_artifact(self, artifact_or_name: Any, type: str | None = None, aliases: Sequence[str] | None = None,
                     use_as: str | None = None) -> Any:
        from runraccoon.artifacts import Artifact
        if isinstance(artifact_or_name, Artifact):
            return artifact_or_name
        name = str(artifact_or_name).split("/")[-1].split(":")[0]
        if self.paths and (self.paths.artifacts / name).is_dir():
            art = Artifact(name, type or "unspecified")
            art.logged_path = self.paths.artifacts / name
            return art
        raise RuntimeError(f"RunRaccoon is local-only: artifact {artifact_or_name!r} cannot be fetched from a server.")

    def watch(self, *args: Any, **kwargs: Any) -> None:
        """No-op: gradient/parameter histograms are not tracked locally."""

    def unwatch(self, *args: Any, **kwargs: Any) -> None:
        pass

    def alert(self, title: str, text: str, level: Any = None, wait_duration: Any = None) -> None:
        _say(self.settings, f"ALERT [{getattr(level, 'value', level) or 'INFO'}] {title}: {text}")
        log.warning("alert: %s: %s", title, text)

    def log_code(self, root: str = ".", name: str | None = None, include_fn: Any = None, exclude_fn: Any = None,
                 **_: Any) -> None:
        """Copy the training script into files/code/ (only the entry-point script)."""
        main = getattr(sys.modules.get("__main__"), "__file__", None)
        if main and self.paths:
            dst = self.paths.files / "code" / os.path.basename(main)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(main, dst)

    def mark_preempting(self) -> None:
        pass

    def status(self) -> str:
        return "finished" if self._finished else "running"

    def finish(self, exit_code: int | None = None, quiet: bool | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        code = exit_code if exit_code is not None else (self._exit_code or 0)
        if self.disabled:
            _clear_module_run(self)
            return
        with self._lock:
            self._commit()
        self._heartbeat_stop.set()
        if self._history:
            self._history.close()
        self._write_summary(final=True)
        self._write_config()
        self._metadata_thread.join(timeout=15)

        status = "finished" if code == 0 else "failed"
        ok = self._scheduler.finish(final=self.settings.final_plots, formats=tuple(self.settings.plot_formats),
                                    status=status, timeout=self.settings.final_plot_timeout_s)
        registry.update(self.id, state="finished", exit_code=code, heartbeat=time.time(), step=self._step,
                        finished=time.time(), runtime=self._runtime())
        if not (quiet or self.settings.quiet):
            self._print_footer(plots_ok=ok)
        if self._console is not None:
            self._console.close()
            self._console = None
        log.info("finished with exit code %s", code)
        if self._log_handler is not None:
            log.removeHandler(self._log_handler)
            self._log_handler.close()
        _clear_module_run(self)

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(exit_code=1 if exc_type else 0)
        return False

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"<runraccoon.Run {self.path} name={self.name!r} step={self._step}>"

    # ========================================================================= internals
    def _runtime(self) -> float:
        return self._runtime_offset + (time.time() - self.start_time)

    def _summary_mode(self, key: str) -> str | None:
        mode = None
        for d in self._metric_defs:
            n = d["name"]
            if n == key:
                return d.get("summary")
            if any(c in n for c in "*?[") and fnmatch.fnmatchcase(key, n):
                mode = d.get("summary", mode)
        return mode

    def _process(self, data: Mapping[str, Any], step: int, prefix: str = "") -> dict:
        """User values -> JSON-ready values; media is written to disk here (key + step are known)."""
        out: dict[str, Any] = {}
        files = self.paths.files
        for k, v in data.items():
            key = f"{prefix}{k}"
            try:
                if isinstance(v, CustomChart):
                    out[f"{key}_table"] = v._bind(files, key, step)
                elif isinstance(v, Media):
                    out[key] = v._bind(files, key, step)
                elif isinstance(v, (list, tuple)) and v and all(isinstance(i, Image) for i in v):
                    out[key] = bind_image_list(v, files, key, step)
                elif isinstance(v, Mapping):
                    out.update(self._process(v, step, prefix=f"{key}."))
                elif (n := as_number(v)) is not None:
                    out[key] = n
                elif v is None or isinstance(v, str):
                    out[key] = v
                elif hasattr(v, "__len__") or hasattr(v, "shape"):
                    out[key] = Histogram(v)._bind(files, key, step)
                else:
                    out[key] = str(v)
            except Exception as e:      # never let one bad value kill a training run
                log.warning("could not log %r: %s", key, e)
        return out

    def _commit(self) -> None:
        if not self._pending:
            return
        row, self._pending = self._pending, {}
        self._write_row(row, self._step)
        self._step += 1

    def _write_row(self, values: dict, step: int, late: bool = False) -> None:
        if not values:
            return
        now = time.time()
        row = {**values, "_step": step, "_runtime": round(self._runtime(), 3), "_timestamp": now}
        self._history.write(row)
        self.summary._observe_row({k: v for k, v in values.items()}, {k: self._summary_mode(k) for k in values})
        if not late or step >= self._step - 1:
            self.summary._load({"_step": step})
        self._summary_changed()
        if not late:
            self._scheduler.notify(row)

    def _summary_changed(self) -> None:
        if time.time() - self._last_summary_write >= 1.0:
            self._write_summary()

    def _write_summary(self, final: bool = False) -> None:
        if self.paths is None:
            return
        self._last_summary_write = time.time()
        data = self.summary.as_dict()
        runtime = self._runtime()
        data.update({"_runtime": round(runtime, 3), "_timestamp": time.time(), "_wandb": {"runtime": int(runtime)}})
        data.setdefault("_step", max(0, self._step - 1))
        try:
            atomic_write_text(self.paths.summary, _dumps(data))
        except OSError as e:
            log.warning("could not write summary: %s", e)

    def _write_config(self) -> None:
        if self.paths is None:
            return
        import yaml
        internal = {"cli_version": f"runraccoon-{__version__}", "python_version": sys.version.split()[0],
                    "start_time": self.start_time, "project": self.project, "run_name": self.name,
                    "group": self.group, "job_type": self.job_type, "tags": list(self.tags), "notes": self.notes,
                    "m": self._metric_defs, "live_metrics": list(self.settings.live_metrics or []) or None,
                    "runraccoon": True}
        doc = {"_wandb": {"value": internal}}
        for k, v in self.config.as_dict().items():
            doc[k] = {"value": v}
        try:
            atomic_write_text(self.paths.config, yaml.safe_dump(to_builtin(doc), sort_keys=False, default_flow_style=False))
        except Exception as e:
            log.warning("could not write config.yaml: %s", e)

    def _write_metadata(self) -> None:
        from runraccoon import metadata
        try:
            atomic_write_json(self.paths.metadata, metadata.collect(str(self.paths.root.parent), self.start_time), indent=2)
            atomic_write_text(self.paths.requirements, metadata.requirements())
        except Exception as e:
            log.info("metadata collection failed: %s", e)

    def _setup_debug_log(self) -> None:
        handler = logging.FileHandler(self.paths.debug_log, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)s:%(process)d %(message)s"))
        handler.setLevel(logging.DEBUG)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        self._log_handler = handler

    def _register(self) -> None:
        registry.write({"id": self.id, "name": self.name, "project": self.project, "group": self.group,
                        "job_type": self.job_type, "tags": list(self.tags), "run_dir": str(self.paths.run_dir),
                        "pid": os.getpid(), "host": socket.gethostname(), "started": self.start_time,
                        "heartbeat": time.time(), "state": "running", "exit_code": None, "step": self._step,
                        "resumed": self.resumed, "program": getattr(sys.modules.get("__main__"), "__file__", None)})

    def _heartbeat(self) -> None:
        while not self._heartbeat_stop.wait(registry.HEARTBEAT_EVERY_S):
            registry.update(self.id, heartbeat=time.time(), step=self._step, runtime=self._runtime(), state="running")
            if time.time() - self._last_summary_write > 5:
                self._write_summary()

    def _load_previous(self, prev_dir: Path) -> None:
        """Resume: carry config, summary, history and media over from the previous directory."""
        from runraccoon.reader import RunData
        prev = RunData(prev_dir)
        self.config.update(prev.config)
        internal = prev.wandb_internal
        self._metric_defs = list(internal.get("m") or [])
        summary = {k: v for k, v in prev.summary.items() if k != "_wandb"}
        self._runtime_offset = float(summary.get("_runtime", 0.0) or 0.0)
        self.summary._load({k: v for k, v in summary.items() if not k.startswith("_")})
        last_step = -1
        with open(self.paths.history, "w", encoding="utf-8") as out:
            for row, _ in iter_rows(prev.paths.history):
                out.write(_dumps(row) + "\n")
                if isinstance(row.get("_step"), int):
                    last_step = max(last_step, row["_step"])
        self._step = last_step + 1
        _link_tree(prev.paths.media, self.paths.media)
        log.info("resumed from %s at step %d", prev_dir, self._step)

    def _print_footer(self, plots_ok: bool) -> None:
        from runraccoon.panels import build_sections, flat_panels
        from runraccoon.reader import RunData
        try:
            rows = RunData(self.paths).rows
            panels = flat_panels(build_sections(rows, self._metric_defs, live=True))[:12]
        except Exception:
            panels = []
        lines = [f"run {self.name} ({self.id}) finished in {format_duration(self._runtime())}"]
        if panels:
            lines.append("history:")
            width = max(len(p.series[0].key) for p in panels)
            for p in panels:
                for s in p.series:
                    lines.append(f"  {s.key:<{width}}  {_sparkline(s.y)}  {format_value(s.y[-1]) if s.y else ''}")
        scalars = {k: v for k, v in self.summary.as_dict().items()
                   if not k.startswith("_") and isinstance(v, (int, float, str)) and not isinstance(v, bool)}
        if scalars:
            lines.append("summary:")
            width = min(40, max(len(k) for k in scalars))
            for k in list(scalars)[:30]:
                lines.append(f"  {k:<{width}}  {format_value(scalars[k])}")
            if len(scalars) > 30:
                lines.append(f"  ... {len(scalars) - 30} more in wandb-summary.json")
        lines.append(f"plots  -> {self.paths.plots}" + ("" if plots_ok else "   (rendering failed: see logs/plots.log)"))
        lines.append(f"files  -> {self.paths.files}")
        if self._dashboard_url:
            lines.append(f"dashboard -> {self.url}")
        for line in lines:
            _say(self.settings, line)


# ------------------------------------------------------------------------------- helpers
def _dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, default=str)


_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float], width: int = 24) -> str:
    vals = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return ""
    if len(vals) > width:
        chunk = len(vals) / width
        vals = [sum(vals[int(i * chunk):max(int(i * chunk) + 1, int((i + 1) * chunk))]) /
                max(1, len(vals[int(i * chunk):max(int(i * chunk) + 1, int((i + 1) * chunk))])) for i in range(width)]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return _SPARK[3] * len(vals)
    return "".join(_SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def _link_tree(src: Path, dst: Path) -> None:
    """Hard-link (or copy) every file under src into dst."""
    if not src.is_dir():
        return
    for p in src.rglob("*"):
        if p.is_file():
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            try:
                os.link(p, target)
            except OSError:
                shutil.copy2(p, target)


def _clear_module_run(run: Run) -> None:
    from runraccoon import sdk
    sdk._on_run_finished(run)
