"""Make `import wandb` resolve to RunRaccoon, so libraries with a built-in wandb integration
(Ultralytics, ...) log locally and can never reach the cloud.

Installed automatically by `import runraccoon` (set RUNRACCOON_SHIM=0 to opt out). Import
RunRaccoon before the library creates its trainer; Ultralytics, for example, imports wandb
lazily when training starts, so `import runraccoon as wandb` at the top of the script is enough.

Child processes (e.g. multi-GPU DDP workers) get WANDB_MODE=disabled so that if they import the
real wandb package it stays offline.
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("runraccoon")


def install_as_wandb() -> None:
    module = sys.modules["runraccoon"]
    existing = sys.modules.get("wandb")
    if existing is module:
        return
    if existing is not None:
        log.warning("the real wandb package was imported before runraccoon; code that already "
                    "holds a reference to it is not redirected. Import runraccoon first.")
    sys.modules["wandb"] = module
    # Common submodule paths some code imports directly.
    sys.modules.setdefault("wandb.plot", module.plot)
    if not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["RUNRACCOON_SET_WANDB_MODE"] = "1"
    os.environ.setdefault("WANDB_SILENT", "true")


def uninstall_as_wandb() -> None:
    if sys.modules.get("wandb") is sys.modules.get("runraccoon"):
        del sys.modules["wandb"]
        sys.modules.pop("wandb.plot", None)
    if os.environ.pop("RUNRACCOON_SET_WANDB_MODE", None):
        os.environ.pop("WANDB_MODE", None)
