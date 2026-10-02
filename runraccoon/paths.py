"""`RunPaths` - every file and folder a run writes, in wandb's layout.

`<dir>` is the output directory the training script passes as `init(dir=...)` (default: the
current directory). The run folder is named `runraccoon` (Settings.dirname) instead of `wandb`.

    <dir>/runraccoon/
        debug.log                          -> latest run's logs/debug.log
        latest-run                         -> run-20260924_083909-t0rb91yy
        run-20260924_083909-t0rb91yy/
            files/
                config.yaml                wandb format: {key: {value: ...}}
                wandb-summary.json         latest/best value of every key
                wandb-metadata.json        host, GPU, git, args, ...
                wandb-history.jsonl        one JSON row per committed step
                output.log                 captured stdout/stderr
                requirements.txt           pip freeze of the environment
                media/images/<key>_<step>_<sha20>.<ext>
                media/table/<key>_<step>_<sha20>.table.json
                plots/                     RunRaccoon's rendered figures (see README)
            logs/debug.log
            artifacts/<name>/manifest.json
            tmp/
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path        # <dir>/runraccoon
    run_dir: Path     # <dir>/runraccoon/run-<timestamp>-<id>

    @classmethod
    def create(cls, base_dir: str | os.PathLike, dirname: str, timestamp: str, run_id: str) -> "RunPaths":
        root = Path(base_dir).expanduser().resolve() / dirname
        paths = cls(root=root, run_dir=root / f"run-{timestamp}-{run_id}")
        for d in (paths.files, paths.logs, paths.tmp, paths.plots):
            d.mkdir(parents=True, exist_ok=True)
        return paths

    @classmethod
    def from_run_dir(cls, run_dir: str | os.PathLike) -> "RunPaths":
        """Open an existing run directory (accepts the run dir itself or its files/ folder)."""
        run_dir = Path(run_dir).expanduser().resolve()
        if run_dir.name == "files":
            run_dir = run_dir.parent
        if run_dir.name == "latest-run" or run_dir.is_symlink():
            run_dir = run_dir.resolve()
        return cls(root=run_dir.parent, run_dir=run_dir)

    # --------------------------------------------------------------------- folders
    @property
    def files(self) -> Path:
        return self.run_dir / "files"

    @property
    def logs(self) -> Path:
        return self.run_dir / "logs"

    @property
    def tmp(self) -> Path:
        return self.run_dir / "tmp"

    @property
    def media(self) -> Path:
        return self.files / "media"

    @property
    def plots(self) -> Path:
        return self.files / "plots"

    @property
    def artifacts(self) -> Path:
        return self.run_dir / "artifacts"

    # ----------------------------------------------------------------------- files
    @property
    def config(self) -> Path:
        return self.files / "config.yaml"

    @property
    def summary(self) -> Path:
        return self.files / "wandb-summary.json"

    @property
    def metadata(self) -> Path:
        return self.files / "wandb-metadata.json"

    @property
    def history(self) -> Path:
        return self.files / "wandb-history.jsonl"

    @property
    def output_log(self) -> Path:
        return self.files / "output.log"

    @property
    def requirements(self) -> Path:
        return self.files / "requirements.txt"

    @property
    def debug_log(self) -> Path:
        return self.logs / "debug.log"

    @property
    def run_id(self) -> str:
        return self.run_dir.name.split("-", 2)[-1]

    # ---------------------------------------------------------------------- links
    def link_latest(self) -> None:
        """Point `latest-run` and `debug.log` at this run, as wandb does (copies if symlinks fail)."""
        _relink(self.root / "latest-run", self.run_dir.name, self.run_dir, is_dir=True)
        _relink(self.root / "debug.log", os.path.join(self.run_dir.name, "logs", "debug.log"), self.debug_log)


def _relink(link: Path, relative_target: str, absolute_target: Path, is_dir: bool = False) -> None:
    try:
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        link.symlink_to(relative_target, target_is_directory=is_dir)
    except OSError:
        if not is_dir and absolute_target.exists():
            try:
                shutil.copyfile(absolute_target, link)
            except OSError:
                pass


def find_run_dirs(root: str | os.PathLike, run_id: str) -> list[Path]:
    """All `run-*-<id>` directories under a wandb root, oldest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob(f"run-*-{run_id}") if p.is_dir() and not p.is_symlink())
