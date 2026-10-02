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

  Training-progress GIFs (made in finish() from images logged at several steps):
    gifs                   make GIFs at all                             RUNRACCOON_GIFS=0/1
    gif_seconds_per_frame  time each step is shown                      RUNRACCOON_GIF_SECONDS=0.25
    gif_orientation        "landscape" (1920x1080) / "portrait"         RUNRACCOON_GIF_ORIENTATION=portrait
    gif_resolution         short side of every frame, px                RUNRACCOON_GIF_RESOLUTION=1080
    gif_hold_last_s        pause on the final frame before looping      RUNRACCOON_GIF_HOLD=1.5
    gif_panels             also one GIF per panel of a contact sheet    RUNRACCOON_GIF_PANELS=0/1
    gif_label              "<key> - epoch N" banner + progress bar      RUNRACCOON_GIF_LABEL=0/1
    gif_keys               glob patterns of image keys to animate       RUNRACCOON_GIF_KEYS=qc/*
    gif_workers            parallel workers (default: cores/threads-2)  RUNRACCOON_GIF_WORKERS=16

  Automatic QC contact sheet (Ultralytics training; predictions + ground truth on fixed val images):
    qc_sheet               make it at all                               RUNRACCOON_QC_SHEET=0/1
    qc_images              number of val images                         RUNRACCOON_QC_IMAGES=12
    qc_every               every N epochs (the last epoch always)       RUNRACCOON_QC_EVERY=1
    qc_cols                grid columns (0 = automatic)                 RUNRACCOON_QC_COLS=4
    qc_tile_px             tile width, px                               RUNRACCOON_QC_TILE_PX=720
    qc_conf                prediction confidence threshold              RUNRACCOON_QC_CONF=0.25
    qc_key                 media key it is logged under                 RUNRACCOON_QC_KEY=runraccoon/qc_contact_sheet
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
    gifs: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_GIFS", True))
    gif_seconds_per_frame: float = field(default_factory=lambda: float(os.environ.get("RUNRACCOON_GIF_SECONDS", "0.25")))
    gif_orientation: str = field(default_factory=lambda: os.environ.get("RUNRACCOON_GIF_ORIENTATION", "landscape"))
    gif_resolution: int = field(default_factory=lambda: int(os.environ.get("RUNRACCOON_GIF_RESOLUTION", "1080")))
    gif_hold_last_s: float = field(default_factory=lambda: float(os.environ.get("RUNRACCOON_GIF_HOLD", "1.5")))
    gif_panels: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_GIF_PANELS", True))
    gif_label: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_GIF_LABEL", True))
    gif_keys: tuple | None = field(default_factory=lambda: _env_list("RUNRACCOON_GIF_KEYS", None))
    gif_workers: int | None = field(default_factory=lambda: int(os.environ["RUNRACCOON_GIF_WORKERS"])
                                    if os.environ.get("RUNRACCOON_GIF_WORKERS") else None)
    qc_sheet: bool = field(default_factory=lambda: _env_bool("RUNRACCOON_QC_SHEET", True))
    qc_images: int = field(default_factory=lambda: int(os.environ.get("RUNRACCOON_QC_IMAGES", "12")))
    qc_every: int = field(default_factory=lambda: int(os.environ.get("RUNRACCOON_QC_EVERY", "1")))
    qc_cols: int = field(default_factory=lambda: int(os.environ.get("RUNRACCOON_QC_COLS", "4")))
    qc_tile_px: int = field(default_factory=lambda: int(os.environ.get("RUNRACCOON_QC_TILE_PX", "720")))
    qc_conf: float = field(default_factory=lambda: float(os.environ.get("RUNRACCOON_QC_CONF", "0.25")))
    qc_key: str = field(default_factory=lambda: os.environ.get("RUNRACCOON_QC_KEY", "runraccoon/qc_contact_sheet"))
    final_plot_timeout_s: float = 900.0
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
                if key in ("plot_formats", "live_metrics", "gif_keys") and isinstance(value, str):
                    value = tuple(p.strip() for p in value.split(",") if p.strip())
                elif key in ("plot_formats", "live_metrics", "gif_keys") and value is not None:
                    value = tuple(value)
                setattr(self, key, value)
            else:
                self.extra[key] = value
        return self

    def gif_options(self) -> dict:
        """The GIF settings as stored in config.yaml (read back by the renderer and `runraccoon gif`)."""
        return {"enabled": bool(self.gifs), "seconds_per_frame": float(self.gif_seconds_per_frame),
                "orientation": self.gif_orientation, "resolution": int(self.gif_resolution),
                "hold_last_s": float(self.gif_hold_last_s), "panels": bool(self.gif_panels),
                "label": bool(self.gif_label), "keys": list(self.gif_keys) if self.gif_keys else None,
                "workers": int(self.gif_workers) if self.gif_workers else None}

    def __getattr__(self, name: str) -> Any:
        # wandb.Settings is a bag of many fields; reading an unknown one returns None, not an error.
        extra = self.__dict__.get("extra", {})
        if name in extra:
            return extra[name]
        if name.startswith("__"):
            raise AttributeError(name)
        return None
