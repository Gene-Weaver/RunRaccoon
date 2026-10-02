"""`Settings` - accepts every wandb.Settings keyword, uses the ones that matter locally.

RunRaccoon-specific options (all optional, all overridable by environment variables):

    dashboard        auto-start the localhost dashboard on init      RUNRACCOON_DASHBOARD=0/1
    dashboard_port   port for the dashboard                          RUNRACCOON_PORT=8473
    live_plots       re-render files/plots/progress.png during runs  RUNRACCOON_LIVE_PLOTS=0/1
    final_plots      render the summary figures in finish()          RUNRACCOON_FINAL_PLOTS=0/1
    plot_every_s     minimum seconds between live re-renders         RUNRACCOON_PLOT_EVERY_S=10
    plot_formats     formats for the final figures                   RUNRACCOON_PLOT_FORMATS=png,pdf
    live_metrics     glob patterns that pick the live-plot metrics   RUNRACCOON_LIVE_METRICS=train/*,val/*
    dirname          run folder inside the output dir ("runraccoon")   RUNRACCOON_DIRNAME=runraccoon
    console          "wrap" tees stdout/stderr to output.log, "off"  RUNRACCOON_CONSOLE=off
    artifact_copy    copy artifact files instead of referencing them RUNRACCOON_ARTIFACT_COPY=0/1
    quiet            suppress RunRaccoon's own console messages      RUNRACCOON_QUIET=0/1
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any


DEFAULT_PORT = 8473      # dashboard port; override with RUNRACCOON_PORT


def default_port() -> int:
    return int(os.environ.get("RUNRACCOON_PORT", DEFAULT_PORT))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_list(name: str, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(p.strip() for p in raw.split(",") if p.strip())


@dataclass
class Settings:
    dashboard: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_DASHBOARD", True))
    dashboard_port: int = field(default_factory=default_port)
    live_plots: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_LIVE_PLOTS", True))
    final_plots: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_FINAL_PLOTS", True))
    plot_every_s: float = field(default_factory=lambda: float(os.environ.get("RUNRACCOON_PLOT_EVERY_S", "10")))
    plot_formats: tuple = field(default_factory=lambda: _env_list("RUNRACCOON_PLOT_FORMATS", ("png",)))
    live_metrics: tuple | None = field(default_factory=lambda: _env_list("RUNRACCOON_LIVE_METRICS", None))
    dirname: str = field(default_factory=lambda: os.environ.get("RUNRACCOON_DIRNAME", "runraccoon"))
    console: str = field(default_factory=lambda: os.environ.get("RUNRACCOON_CONSOLE", "wrap"))
    artifact_copy: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_ARTIFACT_COPY", False))
    quiet: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_QUIET", False))
    final_plot_timeout_s: float = 300.0
    # wandb.Settings keywords that are accepted and ignored (start_method, mode, ...), kept for inspection
    extra: dict = field(default_factory=dict)

    def __init__(self, **kwargs: Any):
        # Hand-written so unknown wandb keywords are kept in `extra` instead of raising TypeError.
        for f in fields(self):
            if f.name != "extra":
                default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
                setattr(self, f.name, default)
        self.extra = {}
        self.update(**kwargs)

    def update(self, other: "Settings | dict | None" = None, **kwargs: Any) -> "Settings":
        items = dict(vars(other)) if isinstance(other, Settings) else dict(other or {})
        if isinstance(other, Settings):
            items.pop("extra", None)
            items.update(other.extra)
        items.update(kwargs)
        known = {f.name for f in fields(self)} - {"extra"}
        for key, value in items.items():
            if key in known:
                if key in ("plot_formats", "live_metrics") and isinstance(value, str):
                    value = tuple(p.strip() for p in value.split(",") if p.strip())
                elif key in ("plot_formats", "live_metrics") and value is not None:
                    value = tuple(value)
                setattr(self, key, value)
            else:
                self.extra[key] = value
        return self

    def __getattr__(self, name: str) -> Any:
        # wandb.Settings is a bag of many fields; reading an unknown one returns None, not an error.
        extra = self.__dict__.get("extra", {})
        if name in extra:
            return extra[name]
        if name.startswith("__"):
            raise AttributeError(name)
        return None
