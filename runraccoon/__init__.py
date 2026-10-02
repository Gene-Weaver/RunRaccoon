"""RunRaccoon - local-only, drop-in replacement for the parts of wandb you actually look at.

    import runraccoon as wandb

    wandb.init(project="my-project", name="baseline", config={"lr": 1e-3})
    for epoch in range(epochs):
        wandb.log({"train/loss": tl, "val/loss": vl, "val/acc": acc, "epoch": epoch})
        wandb.log({"qc/samples": wandb.Image(grid)})
    wandb.finish()

Everything is written under <dir>/runraccoon/run-<timestamp>-<id>/ in wandb's layout, plus rendered
figures in files/plots/. Nothing ever leaves the machine.
"""
from __future__ import annotations

import logging as _logging
import os as _os

from runraccoon._version import __version__
from runraccoon.artifacts import Artifact
from runraccoon.config import Config
from runraccoon.media import Audio, Histogram, Html, Image, Table, Video
from runraccoon import plot
from runraccoon.plot import plot_table
from runraccoon.reader import RunData
from runraccoon.run import Run
from runraccoon.sdk import (AlertLevel, Api, alert, define_metric, finish, init, log, log_artifact, log_model,
                            login, save, setup, teardown, termerror, termlog, termwarn, unwatch, use_artifact,
                            watch)
from runraccoon.settings import Settings
from runraccoon.shim import install_as_wandb, uninstall_as_wandb
from runraccoon.summary import Summary

# Active-run handles, replaced by init()/finish() (same as wandb.run / wandb.config / wandb.summary).
run: Run | None = None
config: Config = Config()
summary: Summary = Summary()

__all__ = ["__version__", "init", "log", "finish", "define_metric", "save", "log_artifact", "log_model",
           "use_artifact", "alert", "watch", "unwatch", "login", "setup", "teardown", "Run", "Settings",
           "Config", "Summary", "Image", "Table", "Histogram", "Video", "Audio", "Html", "Artifact", "plot",
           "plot_table", "RunData", "Api", "AlertLevel", "run", "config", "summary", "install_as_wandb",
           "uninstall_as_wandb"]

_log = _logging.getLogger("runraccoon")
if not _log.handlers:
    _console = _logging.StreamHandler()
    _console.setLevel(_logging.WARNING)
    _console.setFormatter(_logging.Formatter("runraccoon: %(message)s"))
    _log.addHandler(_console)
    _log.propagate = False

if _os.environ.get("RUNRACCOON_SHIM", "1").strip().lower() not in ("0", "false", "no", "off"):
    install_as_wandb()
