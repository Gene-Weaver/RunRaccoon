"""System metadata for wandb-metadata.json and the pip snapshot for requirements.txt.

Collected in a background thread at init so `nvidia-smi` / `git` never delay training.
"""
from __future__ import annotations

import getpass
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from runraccoon.utils import iso_utc


def _run(cmd: list[str], cwd: str | None = None, timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _program() -> Path | None:
    main = sys.modules.get("__main__")
    f = getattr(main, "__file__", None)
    return Path(f).resolve() if f else None


def _git(program: Path | None) -> dict:
    cwd = str(program.parent) if program else os.getcwd()
    root = _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    if not root:
        return {}
    info = {"root": root, "commit": _run(["git", "rev-parse", "HEAD"], cwd=cwd),
            "remote": _run(["git", "config", "--get", "remote.origin.url"], cwd=cwd)}
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd)
    info["dirty"] = bool(dirty)
    return {k: v for k, v in info.items() if v is not None}


def _gpus() -> dict:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,uuid,driver_version",
                "--format=csv,noheader,nounits"])
    if not out:
        return {}
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memoryTotal": str(int(float(parts[1]) * 1024 * 1024)),
                         "uuid": parts[2], "driver": parts[3] if len(parts) > 3 else None})
    info: dict[str, Any] = {"gpu": gpus[0]["name"] if gpus else None, "gpu_count": len(gpus), "gpu_nvidia": gpus}
    header = _run(["nvidia-smi"])
    if header and "CUDA Version:" in header:
        info["cudaVersion"] = header.split("CUDA Version:")[1].split()[0]
    return info


def _memory_total() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def collect(root: str, started_at: float) -> dict:
    program = _program()
    git = _git(program)
    meta: dict[str, Any] = {
        "os": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "startedAt": iso_utc(started_at),
        "args": sys.argv[1:],
        "program": str(program) if program else "<interactive>",
        "codePathLocal": os.path.relpath(program, os.getcwd()) if program else None,
        "host": socket.gethostname(),
        "username": getpass.getuser(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "root": root,
        "cpu_count_logical": os.cpu_count(),
        "memory": {"total": str(_memory_total())} if _memory_total() else {},
        "runraccoon": True,
    }
    if program and git.get("root"):
        meta["codePath"] = os.path.relpath(program, git["root"])
        meta["git"] = {k: git[k] for k in ("commit", "remote", "dirty") if k in git}
    try:
        du = shutil.disk_usage(root)
        meta["disk"] = {"/": {"total": str(du.total), "used": str(du.used)}}
    except OSError:
        pass
    meta.update(_gpus())
    return meta


def requirements() -> str:
    """`name==version` for every installed distribution, sorted (what wandb writes)."""
    try:
        from importlib import metadata as md
    except ImportError:  # pragma: no cover
        return ""
    seen = {}
    for dist in md.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if name and name.lower() not in seen:
            seen[name.lower()] = f"{name}=={dist.version}"
    return "\n".join(seen[k] for k in sorted(seen)) + "\n"
