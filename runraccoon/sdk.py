"""Module-level API: `init`, `log`, `finish`, ... - the functions you call as `wandb.<name>`.

`runraccoon.run`, `runraccoon.config` and `runraccoon.summary` always point at the active run
(or are None / empty placeholders when there is none), exactly like their wandb counterparts.
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from runraccoon.config import Config
from runraccoon.paths import find_run_dirs
from runraccoon.run import Run
from runraccoon.settings import Settings
from runraccoon.summary import Summary
from runraccoon.utils import generate_id

log = logging.getLogger("runraccoon")
_lock = threading.RLock()
_active: list[Run] = []
_hooks_installed = False

# WANDB_MODE as the *user* set it, before the shim (which sets it to "disabled" for subprocesses).
_USER_WANDB_MODE = os.environ.get("WANDB_MODE") if not os.environ.get("RUNRACCOON_SET_WANDB_MODE") else None

_ADJ = ["brisk", "clever", "curious", "dusky", "gentle", "hidden", "lively", "mellow", "nimble", "quiet",
        "rustling", "silver", "sly", "spry", "starlit", "sunny", "tidy", "wild", "witty", "zesty"]
_NOUN = ["acorn", "bandit", "burrow", "creek", "fern", "hollow", "lantern", "maple", "meadow", "moss",
         "otter", "pebble", "pine", "raccoon", "ridge", "river", "rowan", "thicket", "trail", "willow"]


def _module():
    return sys.modules["runraccoon"]


def _friendly_name(run_id: str) -> str:
    h = int(hashlib.sha1(run_id.encode()).hexdigest(), 16)
    return f"{_ADJ[h % len(_ADJ)]}-{_NOUN[(h // 97) % len(_NOUN)]}-{h % 1000}"


def _set_module_run(run: Run | None) -> None:
    mod = _module()
    mod.run = run
    mod.config = run.config if run is not None else Config()
    mod.summary = run.summary if run is not None else Summary()


def _on_run_finished(run: Run) -> None:
    with _lock:
        if run in _active:
            _active.remove(run)
        if getattr(_module(), "run", None) is run:
            _set_module_run(_active[-1] if _active else None)


def _install_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    previous = sys.excepthook

    def excepthook(exc_type, exc, tb):
        for r in list(_active):
            r._exit_code = 255 if issubclass(exc_type, KeyboardInterrupt) else 1
        previous(exc_type, exc, tb)

    sys.excepthook = excepthook

    @atexit.register
    def _finish_all() -> None:
        for r in list(_active):
            try:
                r.finish()
            except Exception as e:      # pragma: no cover - best effort at interpreter exit
                print(f"runraccoon: error while finishing run {r.id}: {e}", file=sys.stderr)


def _filter_config(config: Any, include: Sequence[str] | None, exclude: Sequence[str] | None) -> Any:
    if config is None or (not include and not exclude):
        return config
    from runraccoon.config import _to_mapping
    d = _to_mapping(config)
    if include:
        d = {k: v for k, v in d.items() if k in include}
    if exclude:
        d = {k: v for k, v in d.items() if k not in exclude}
    return d


# ======================================================================================= init
def init(project: str | None = None, entity: str | None = None, name: str | None = None,
         config: Any = None, dir: str | os.PathLike | None = None, id: str | None = None,
         group: str | None = None, job_type: str | None = None, tags: Sequence[str] | None = None,
         notes: str | None = None, mode: str | None = None, resume: bool | str | None = None,
         reinit: bool | str | None = None, settings: Settings | Mapping | None = None,
         config_exclude_keys: Sequence[str] | None = None, config_include_keys: Sequence[str] | None = None,
         allow_val_change: bool | None = None, save_code: bool | None = None, **_ignored: Any) -> Run:
    """Start a run. Same signature as `wandb.init`; `entity` and cloud-only options are ignored.

    Environment defaults honored: WANDB_PROJECT, WANDB_NAME, WANDB_RUN_ID, WANDB_RUN_GROUP,
    WANDB_JOB_TYPE, WANDB_TAGS, WANDB_NOTES, WANDB_DIR, WANDB_RESUME, WANDB_MODE (=disabled).
    """
    env = os.environ
    s = Settings()
    if settings is not None:
        s.update(settings if isinstance(settings, Settings) else dict(settings))
    mode = (mode or s.extra.get("mode") or env.get("RUNRACCOON_MODE") or _USER_WANDB_MODE or "local").lower()
    disabled = mode in ("disabled", "dryrun-disabled")
    if mode in ("online", "shared"):
        log.info("RunRaccoon is local-only; mode=%r is treated as local", mode)

    with _lock:
        current = getattr(_module(), "run", None)
        if current is not None and not current._finished:
            if reinit == "create_new":
                pass
            elif reinit in (False, "return_previous"):
                return current
            else:                                   # default / True / "finish_previous"
                current.finish()

    project = project or env.get("WANDB_PROJECT") or "uncategorized"
    base_dir = os.fspath(dir) if dir else env.get("WANDB_DIR") or os.getcwd()
    run_id = id or env.get("WANDB_RUN_ID")
    resume = resume if resume is not None else env.get("WANDB_RESUME")
    tags = tags if tags is not None else [t for t in env.get("WANDB_TAGS", "").split(",") if t]

    resume_from = None
    root = Path(base_dir).expanduser().resolve() / s.dirname
    if resume in (True, "auto", "allow", "must") and run_id is None and resume != "must":
        latest = root / "latest-run"
        if latest.exists():
            run_id = latest.resolve().name.split("-", 2)[-1]
    if run_id:
        previous = find_run_dirs(root, run_id)
        if previous and resume not in ("never",):
            resume_from = previous[-1]
        elif previous and resume == "never":
            raise ValueError(f"run id {run_id!r} already exists in {root} and resume='never'")
        elif resume == "must":
            raise ValueError(f"resume='must' but no run with id {run_id!r} exists in {root}")
    run_id = run_id or generate_id()

    if not name and resume_from is not None:
        try:
            from runraccoon.reader import RunData
            name = RunData(resume_from).wandb_internal.get("run_name")
            project = RunData(resume_from).project or project
        except Exception:
            pass
    name = name or env.get("WANDB_NAME") or _friendly_name(run_id)

    run = Run(project=project, name=str(name), run_id=run_id, base_dir=base_dir,
              group=group or env.get("WANDB_RUN_GROUP"), job_type=job_type or env.get("WANDB_JOB_TYPE"),
              tags=tags, notes=notes or env.get("WANDB_NOTES"),
              config=_filter_config(config, config_include_keys, config_exclude_keys),
              settings=s, disabled=disabled, resume_from=resume_from)
    if save_code and not disabled:
        run.log_code()
    with _lock:
        _active.append(run)
        _set_module_run(run)
    _install_hooks()
    return run


# ================================================================== module-level shortcuts
def _require_run(fn: str) -> Run:
    run = getattr(_module(), "run", None)
    if run is None:
        raise RuntimeError(f"You must call runraccoon.init() before runraccoon.{fn}()")
    return run


def log(data: Mapping[str, Any], step: int | None = None, commit: bool | None = None, sync: bool | None = None) -> None:
    _require_run("log").log(data, step=step, commit=commit)


def finish(exit_code: int | None = None, quiet: bool | None = None) -> None:
    run = getattr(_module(), "run", None)
    if run is not None:
        run.finish(exit_code=exit_code, quiet=quiet)


def define_metric(name: str, step_metric: str | None = None, **kwargs: Any) -> dict:
    return _require_run("define_metric").define_metric(name, step_metric=step_metric, **kwargs)


def save(glob_str: str, base_path: str | None = None, policy: str = "live") -> list[str]:
    return _require_run("save").save(glob_str, base_path=base_path, policy=policy)


def log_artifact(artifact_or_path: Any, name: str | None = None, type: str | None = None,
                 aliases: Sequence[str] | None = None, **kwargs: Any) -> Any:
    return _require_run("log_artifact").log_artifact(artifact_or_path, name=name, type=type, aliases=aliases)


def log_model(path: str, name: str | None = None, aliases: Sequence[str] | None = None) -> Any:
    return _require_run("log_model").log_model(path, name=name, aliases=aliases)


def use_artifact(artifact_or_name: Any, type: str | None = None, aliases: Sequence[str] | None = None,
                 use_as: str | None = None) -> Any:
    return _require_run("use_artifact").use_artifact(artifact_or_name, type=type, aliases=aliases)


def alert(title: str, text: str, level: Any = None, wait_duration: Any = None) -> None:
    _require_run("alert").alert(title, text, level=level)


def watch(*args: Any, **kwargs: Any) -> None:
    """No-op (gradient histograms are not tracked locally)."""


def unwatch(*args: Any, **kwargs: Any) -> None:
    pass


def login(*args: Any, **kwargs: Any) -> bool:
    """No-op: RunRaccoon never talks to a server, so there is nothing to log in to."""
    return True


def setup(*args: Any, **kwargs: Any) -> None:
    pass


def teardown(*args: Any, **kwargs: Any) -> None:
    finish()


def termlog(msg: str = "", **_: Any) -> None:
    print(f"runraccoon: {msg}", file=sys.stderr)


def termwarn(msg: str = "", **_: Any) -> None:
    print(f"runraccoon: WARNING {msg}", file=sys.stderr)


def termerror(msg: str = "", **_: Any) -> None:
    print(f"runraccoon: ERROR {msg}", file=sys.stderr)


class Api:
    """The wandb public (cloud) API does not exist locally."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError("RunRaccoon is local-only: wandb.Api() (the cloud query API) is not available. "
                           "Read runs from disk with runraccoon.RunData('<run dir>') instead.")


class AlertLevel:
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
