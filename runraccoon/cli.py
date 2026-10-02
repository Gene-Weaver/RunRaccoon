"""`runraccoon` command line.

    runraccoon dashboard [--port 8473]     open the localhost dashboard (all runs, live + past)
    runraccoon ls [--all]                  list runs and their status
    runraccoon replot <run dir | id>       re-render every figure for a run (e.g. after a crash)
    runraccoon gif <run dir | id>          (re)make the training-progress GIFs of a run's QC images
    runraccoon register <dir>...           add existing run folders (moved / copied) to the dashboard
    runraccoon forget <id>...              remove runs from the dashboard index (files are kept)
    runraccoon gc                          forget runs whose folders no longer exist
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from runraccoon import registry
from runraccoon.settings import default_port
from runraccoon.utils import format_duration


def _resolve_run_dir(target: str) -> Path:
    p = Path(target).expanduser()
    if p.exists():
        if (p / "files").is_dir() or p.name == "files":
            return p
        for candidate in (p, p / "runraccoon", p / "wandb"):     # the run folder or the output dir
            if (candidate / "latest-run").exists():
                return (candidate / "latest-run").resolve()
        raise SystemExit(f"{p} is not a run directory (expected .../runraccoon/run-<time>-<id>)")
    rec = registry.read(target)
    if rec and rec.get("run_dir"):
        return Path(rec["run_dir"])
    raise SystemExit(f"no run directory or registered run id {target!r}")


def cmd_dashboard(args) -> int:
    from runraccoon.dashboard.launcher import ping
    from runraccoon.dashboard.server import serve
    if ping(args.port):
        url = f"http://127.0.0.1:{args.port}/"
        print(f"dashboard already running on {url}")
        if not args.no_browser:
            import webbrowser
            webbrowser.open(url)
        return 0
    serve(args.host, args.port, idle_exit_s=args.idle_exit, open_browser=not args.no_browser)
    return 0


def cmd_ls(args) -> int:
    runs = registry.listing()
    if not args.all:
        runs = runs[:25]
    if not runs:
        print("no runs registered yet")
        return 0
    rows = []
    for r in runs:
        started = _dt.datetime.fromtimestamp(r.get("started") or 0).strftime("%Y-%m-%d %H:%M")
        rt = format_duration(r["runtime"]) if isinstance(r.get("runtime"), (int, float)) else ""
        rows.append((r["status"], r.get("id", ""), str(r.get("project") or ""), str(r.get("name") or ""), started,
                     rt, str(r.get("step") if r.get("step") is not None else "")))
    head = ("STATUS", "ID", "PROJECT", "NAME", "STARTED", "RUNTIME", "STEP")
    widths = [max(len(head[i]), *(len(row[i]) for row in rows)) for i in range(len(head))]
    widths[3] = min(widths[3], 48)
    print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    for row in rows:
        print("  ".join(c[:w].ljust(w) for c, w in zip(row, widths)))
    return 0


def cmd_replot(args) -> int:
    from runraccoon.plotting.render import render_run
    run_dir = _resolve_run_dir(args.target)
    written = render_run(run_dir, final=True, formats=[f for f in args.formats.split(",") if f],
                         status=args.status, gifs=not args.no_gifs)
    for p in written:
        print(p)
    return 0


def cmd_gif(args) -> int:
    from runraccoon.panels import x_label_hint
    from runraccoon.plotting.gifs import render_gifs, saved_options
    from runraccoon.reader import RunData
    data = RunData(_resolve_run_dir(args.target))
    opts = saved_options(data)                       # run config + the dashboard's last regeneration settings
    if args.defaults:
        from runraccoon.plotting.gifs import GifOptions
        opts = GifOptions()
    for name in ("seconds_per_frame", "orientation", "resolution", "hold_last_s", "keys", "max_frames", "workers"):
        v = getattr(args, name)
        if v is not None:
            setattr(opts, name, tuple(p for p in v.split(",") if p) if name == "keys" else v)
    if args.no_panels:
        opts.panels = False
    if args.no_label:
        opts.label = False
    written = render_gifs(data, opts, x_word=x_label_hint(data.rows, data.config) or "step")
    for p in written:
        print(p)
    print(f"{len(written)} GIF(s) at {opts.canvas[0]}x{opts.canvas[1]}, {opts.seconds_per_frame}s per frame"
          if written else "no image key was logged at enough steps to animate")
    return 0


def cmd_render(args) -> int:                         # used internally by the training process
    from runraccoon.plotting.render import main
    argv = [args.run_dir, "--formats", args.formats] + (["--final"] if args.final else [])
    if args.status:
        argv += ["--status", args.status]
    return main(argv)


def cmd_register(args) -> int:
    n = 0
    for target in args.dirs:
        base = Path(target).expanduser()
        candidates = [base] if (base / "files").is_dir() else sorted(
            p for p in base.rglob("run-*") if p.is_dir() and (p / "files").is_dir() and not p.is_symlink())
        for c in candidates:
            rec = registry.register_existing(c)
            if rec:
                n += 1
                print(f"registered {rec['id']}  {c}")
    print(f"{n} run(s) registered")
    return 0


def cmd_forget(args) -> int:
    for rid in args.ids:
        print(("forgot " if registry.forget(rid) else "unknown run ") + rid)
    return 0


def cmd_gc(args) -> int:
    n = 0
    for r in registry.all_records():
        if not Path(r.get("run_dir", "")).exists():
            registry.forget(r["id"])
            n += 1
    print(f"forgot {n} run(s) whose folders are gone")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="runraccoon", description="Local-only experiment tracking (wandb drop-in).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dashboard", help="open the localhost dashboard")
    p.add_argument("--port", type=int, default=default_port(), help="default: $RUNRACCOON_PORT or 8473")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    p.add_argument("--idle-exit", type=float, default=None, help="exit after N seconds with no active runs")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("ls", help="list runs")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_ls)

    p = sub.add_parser("replot", help="re-render all figures for a run")
    p.add_argument("target", help="run directory, its runraccoon/ folder, the output dir, or a run id")
    p.add_argument("--formats", default="png", help="comma-separated, e.g. png,pdf,svg")
    p.add_argument("--status", default=None)
    p.add_argument("--no-gifs", action="store_true", help="figures only; leave the GIFs as they are")
    p.set_defaults(fn=cmd_replot)

    p = sub.add_parser("gif", help="(re)make training-progress GIFs from a run's QC images")
    p.add_argument("target", help="run directory, its runraccoon/ folder, the output dir, or a run id")
    p.add_argument("--seconds", dest="seconds_per_frame", type=float, default=None, help="seconds per step (default 0.25)")
    p.add_argument("--orientation", choices=("landscape", "portrait"), default=None)
    p.add_argument("--resolution", type=int, default=None, help="short side in px (default 1080)")
    p.add_argument("--hold", dest="hold_last_s", type=float, default=None, help="pause on the last frame, s (default 1.5)")
    p.add_argument("--keys", default=None, help="comma-separated glob patterns of image keys, e.g. qc/*")
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None, help="subsample longer runs (default 400)")
    p.add_argument("--workers", type=int, default=None, help="parallel workers (default: max(cores, threads) - 2)")
    p.add_argument("--defaults", action="store_true", help="ignore saved settings and use the defaults")
    p.add_argument("--no-panels", action="store_true", help="skip the per-panel GIFs")
    p.add_argument("--no-label", action="store_true", help="no title / step banner / progress bar")
    p.set_defaults(fn=cmd_gif)

    p = sub.add_parser("render", help=argparse.SUPPRESS)
    p.add_argument("run_dir")
    p.add_argument("--final", action="store_true")
    p.add_argument("--formats", default="png")
    p.add_argument("--status", default=None)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("register", help="add existing run folders to the dashboard")
    p.add_argument("dirs", nargs="+")
    p.set_defaults(fn=cmd_register)

    p = sub.add_parser("forget", help="remove runs from the dashboard index")
    p.add_argument("ids", nargs="+")
    p.set_defaults(fn=cmd_forget)

    p = sub.add_parser("gc", help="forget runs whose folders were deleted")
    p.set_defaults(fn=cmd_gc)

    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
