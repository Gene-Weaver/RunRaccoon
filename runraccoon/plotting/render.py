"""Render every RunRaccoon figure for a run directory, from the files on disk.

    live  -> files/plots/progress.png                       (refreshed every epoch while training)
    final -> files/plots/progress.png
             files/plots/summary/run_summary.<fmt>           report card: headline numbers + key curves
             files/plots/summary/<section>.<fmt>             every chart, one figure per section
             files/plots/charts/<key>.<fmt>                  custom charts (PR curves, confusion matrix, ...)
             files/plots/history.csv                         every scalar, one row per step

Runs in its own process (`python -m runraccoon render <run_dir>`) so plotting never competes
with training for the GIL and never touches the training script's matplotlib state.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

from runraccoon.panels import Panel, Section, build_sections, classify, flat_panels, guess_goal, x_label_hint
from runraccoon.reader import RunData
from runraccoon.utils import format_duration, format_value, safe_relpath

_PRIORITY = ["map50-95", "map", "f1", "acc", "iou", "dice", "auc", "psnr", "ssim", "precision", "recall"]


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_").lower() or "section"


def _save(fig, path: Path, fmt: str, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    fig.savefig(tmp, format=fmt, dpi=dpi)
    os.replace(tmp, path)
    return path


def _subtitle(data: RunData, rows: list[dict], panels: Sequence[Panel], status: str) -> str:
    parts = []
    if data.project:
        parts.append(data.project)
    parts.append(data.paths.run_id)
    if panels:                                  # progress along the x-axis the charts actually use
        p = panels[0]
        last = max((max(s.x) for s in p.series if s.x), default=None)
        if last is not None:
            parts.append(f"{p.x_label if p.x_label != '_step' else 'step'} {format_value(last) if last % 1 else int(last)}")
    runtime = data.summary.get("_runtime") or (rows[-1].get("_runtime") if rows else None)
    if isinstance(runtime, (int, float)):
        parts.append(format_duration(runtime))
    parts.append(status)
    return "   ·   ".join(parts)


def _priority(name: str) -> int:
    low = name.lower()
    for i, token in enumerate(_PRIORITY):
        if token in low:
            return i
    return len(_PRIORITY)


def headline_tiles(panels: Sequence[Panel], summary: dict) -> list[tuple[str, str, str]]:
    """Up to six (label, value, note) tiles: test results first, then best validation scores."""
    tiles: list[tuple[str, str, str]] = []
    test = []
    for k, v in summary.items():
        if k.startswith("_") or isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        role, base, _ = classify(k)
        if role == "test" and guess_goal(base) == "max":
            test.append((base.rsplit("/", 1)[-1], v))
    for name, v in sorted(test, key=lambda t: _priority(t[0]))[:4]:
        tiles.append((f"test · {name}", format_value(v), "held-out test set"))

    scored = [p for p in panels if p.goal == "max" and p.best()]
    scored.sort(key=lambda p: _priority(p.title.rsplit("/", 1)[-1]))
    errors = [p for p in panels if p.goal == "min" and not p.is_loss and p.best()]
    losses = [p for p in panels if p.is_loss and p.best() and any(s.role == "val" for s in p.series)]
    for p in [*scored, *errors, *losses]:
        if len(tiles) >= 6:
            break
        s, x, y = p.best()
        word = "best" if p.goal == "max" else "lowest"
        where = f"{s.role + ' ' if s.role else ''}@ {p.x_label if p.x_label != '_step' else 'step'} {int(x) if float(x).is_integer() else x}"
        tiles.append((f"{word} {p.title.rsplit('/', 1)[-1] if '/' in p.title and not s.role else p.title}",
                      format_value(y), where))
    return tiles


def write_history_csv(rows: list[dict], path: Path) -> Path:
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k, v in r.items():
            if k not in seen and (isinstance(v, (int, float, str)) or v is None) and k not in ("_timestamp",):
                seen.add(k)
                keys.append(k)
    lead = [k for k in ("_step", "_runtime", "epoch") if k in seen]
    keys = lead + [k for k in keys if k not in lead]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow(["" if r.get(k) is None or isinstance(r.get(k), dict) else r.get(k) for k in keys])
    return path


def render_charts(data: RunData, formats: Iterable[str], dpi: int) -> list[Path]:
    from runraccoon.plotting.figures import chart_figure
    out = []
    for key, ref in data.chart_refs().items():
        try:
            table = json.loads((data.paths.files / ref["path"]).read_text())
        except (OSError, ValueError):
            continue
        fig = chart_figure(table, ref.get("_runraccoon_chart", {}), key)
        if fig is None:
            continue
        for fmt in formats:
            out.append(_save(fig, data.paths.plots / "charts" / f"{safe_relpath(key)}.{fmt}", fmt, dpi))
    return out


def render_run(run_dir: str | Path, final: bool = False, formats: Sequence[str] = ("png",),
               status: str | None = None) -> list[Path]:
    from runraccoon.plotting import style
    from runraccoon.plotting.figures import progress_figure, report_figure, section_figure

    style.apply()
    data = RunData(run_dir)
    rows = data.rows
    hint = x_label_hint(rows, data.config)
    status = status or ("finished" if final else "running")
    title = data.name
    run_dir_s = str(data.paths.run_dir)
    live_patterns = data.wandb_internal.get("live_metrics") or None
    written: list[Path] = []

    live_sections = build_sections(rows, data.metric_defs, include=live_patterns, live=True, x_label_hint=hint)
    subtitle = _subtitle(data, rows, flat_panels(live_sections), status)
    fig = progress_figure(flat_panels(live_sections), title, subtitle, run_dir_s)
    if fig is not None:
        written.append(_save(fig, data.paths.plots / "progress.png", "png", 150 if not final else 200))

    if final:
        import shutil
        shutil.rmtree(data.paths.plots / "summary", ignore_errors=True)      # no stale sections
        sections: list[Section] = build_sections(rows, data.metric_defs, live=False, x_label_hint=hint)
        key_panels = [p for p in flat_panels(live_sections) if p.series]
        report = report_figure(key_panels[:8], headline_tiles(flat_panels(sections), data.summary), title,
                               subtitle, run_dir_s)
        for fmt in formats:
            if report is not None:
                written.append(_save(report, data.paths.plots / "summary" / f"run_summary.{fmt}", fmt, 200))
            for sec in sections:
                f = section_figure(sec.name, sec.panels, title, subtitle, run_dir_s)
                if f is not None:
                    written.append(_save(f, data.paths.plots / "summary" / f"{_slug(sec.name)}.{fmt}", fmt, 200))
        if rows:
            written.append(write_history_csv(rows, data.paths.plots / "history.csv"))
    written.extend(render_charts(data, formats if final else ("png",), 200))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="runraccoon render")
    ap.add_argument("run_dir")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--formats", default="png")
    ap.add_argument("--status", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    paths = render_run(args.run_dir, final=args.final, formats=[f for f in args.formats.split(",") if f],
                       status=args.status)
    print(f"rendered {len(paths)} files in {time.time() - t0:.1f}s")
    return 0
