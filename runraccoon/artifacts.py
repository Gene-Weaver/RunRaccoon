"""`Artifact` - a named bundle of files (e.g. model weights), recorded locally.

By default files are *referenced*, not copied: logging an artifact writes
    <run_dir>/artifacts/<name>/manifest.json
listing each file's absolute path, size and sha256. Large checkpoints are not duplicated.
Set `Settings(artifact_copy=True)` (or RUNRACCOON_ARTIFACT_COPY=1) to copy them instead.
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from runraccoon.utils import atomic_write_json, sha256_file, to_builtin

log = logging.getLogger("runraccoon")


class Artifact:
    def __init__(self, name: str, type: str, description: str | None = None, metadata: dict | None = None,
                 incremental: bool = False, use_as: str | None = None, **_ignored: Any):
        self.name = str(name).replace("/", "-")
        self.type = type
        self.description = description
        self.metadata = to_builtin(metadata or {})
        self.aliases: list[str] = []
        self._entries: list[dict] = []              # {"name", "path"} or {"name", "ref"}
        self._staging: Path | None = None
        self.logged_path: Path | None = None

    def add_file(self, local_path: str | os.PathLike, name: str | None = None, **_ignored: Any) -> None:
        path = Path(local_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Artifact.add_file: {path} does not exist")
        self._entries.append({"name": name or path.name, "path": str(path)})

    def add_dir(self, local_path: str | os.PathLike, name: str | None = None, **_ignored: Any) -> None:
        root = Path(local_path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Artifact.add_dir: {root} is not a directory")
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                self._entries.append({"name": f"{name}/{rel}" if name else rel, "path": str(p)})

    def add_reference(self, uri: str, name: str | None = None, **_ignored: Any) -> None:
        self._entries.append({"name": name or uri.rstrip("/").split("/")[-1], "ref": uri})

    def add(self, obj: Any, name: str) -> None:
        """Add a Table / media object; it is written into the artifact folder when logged."""
        self._entries.append({"name": name, "object": obj})

    @contextlib.contextmanager
    def new_file(self, name: str, mode: str = "w", encoding: str | None = None) -> Iterator[Any]:
        if self._staging is None:
            self._staging = Path(tempfile.mkdtemp(prefix="runraccoon-artifact-"))
        path = self._staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding=encoding if "b" not in mode else None) as fh:
            yield fh
        self._entries.append({"name": name, "path": str(path), "staged": True})

    # --- called by Run.log_artifact ------------------------------------------------------------
    def _save(self, artifacts_dir: Path, copy: bool, aliases: list[str] | None = None) -> Path:
        self.aliases = list(dict.fromkeys(["latest", *(aliases or [])]))
        target = artifacts_dir / self.name
        target.mkdir(parents=True, exist_ok=True)
        manifest = []
        for e in self._entries:
            if "ref" in e:
                manifest.append({"name": e["name"], "ref": e["ref"]})
                continue
            if "object" in e:
                obj = e["object"]
                from runraccoon.media import Table
                if isinstance(obj, Table):
                    import json
                    out = target / f"{e['name']}.table.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(obj.to_json_obj()))
                    manifest.append({"name": e["name"], "path": str(out), "copied": True})
                continue
            src = Path(e["path"])
            entry = {"name": e["name"], "size": src.stat().st_size, "sha256": sha256_file(src)}
            if copy or e.get("staged"):
                dst = target / e["name"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                entry.update(path=str(dst), copied=True)
            else:
                entry.update(path=str(src), copied=False)
            manifest.append(entry)
        atomic_write_json(target / "manifest.json", {
            "name": self.name, "type": self.type, "description": self.description, "metadata": self.metadata,
            "aliases": self.aliases, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": manifest}, indent=2)
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None
        self.logged_path = target
        return target

    # --- cloud-only operations ------------------------------------------------------------------
    def download(self, *args: Any, **kwargs: Any) -> str:
        if self.logged_path is not None:
            return str(self.logged_path)
        raise RuntimeError("RunRaccoon is local-only: artifacts cannot be downloaded from a server.")

    def wait(self, *args: Any, **kwargs: Any) -> "Artifact":
        return self

    def __repr__(self) -> str:
        return f"Artifact(name={self.name!r}, type={self.type!r}, files={len(self._entries)})"
