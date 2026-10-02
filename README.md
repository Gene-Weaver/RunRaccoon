# RunRaccoon 🦝

**Local-only, drop-in replacement for the parts of Weights & Biases you actually look at.**

RunRaccoon keeps wandb's API and file layout but has no cloud, no login, and no sync. Runs
are written to a `runraccoon/` folder inside the output directory your training script
already uses. Each run gets the files wandb would write, plus publication-ready figures:

- a **progress plot refreshed every epoch**,
- **QC images** saved exactly where wandb saves them,
- **summary figures** when the run ends,
- an optional **localhost dashboard** that shows every run on the machine, live and past.

```diff
- import wandb
+ import runraccoon as wandb
```

That one-line change is all most scripts need, including scripts that rely on Ultralytics'
built-in wandb integration.

![Run summary figure](https://raw.githubusercontent.com/Gene-Weaver/RunRaccoon/main/docs/images/run_summary.png)

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [What you get on disk](#what-you-get-on-disk)
- [The figures](#the-figures)
- [The dashboard](#the-dashboard)
- [Ultralytics / YOLO](#ultralytics--yolo)
- [Swapping an existing wandb script](#swapping-an-existing-wandb-script)
- [API compatibility](#api-compatibility)
- [Settings and environment variables](#settings-and-environment-variables)
- [Command line](#command-line)
- [How it works](#how-it-works)
- [Limitations](#limitations)
- [Development](#development)
- [Releasing to PyPI](#releasing-to-pypi)

---

## Install

### Option 1: from [PyPI](https://pypi.org/project/runraccoon/)

Install
```bash
pip install runraccoon
```

Upgrade
```bash
pip install --upgrade runraccoon
```

### Option 2: from source (editable)

```bash
git clone https://github.com/Gene-Weaver/RunRaccoon.git
cd RunRaccoon
pip install -e .              # into the environment you train in
```

Requirements: Python ≥ 3.9, `numpy`, `matplotlib`, `pillow`, `pyyaml`. The dashboard uses
only the standard library and works fully offline.

## Quick start

```python
import runraccoon as wandb

wandb.init(project="leaf-segmentation", name="unet-baseline", config={"lr": 3e-4, "epochs": 50},
           dir="outputs/unet-baseline")      # your output dir; RunRaccoon adds runraccoon/ inside it

for epoch in range(50):
    ...
    wandb.log({"train/loss": train_loss, "val/loss": val_loss, "val/iou": val_iou, "epoch": epoch})
    if epoch % 5 == 0:
        wandb.log({"qc/overlay": wandb.Image(overlay, caption=f"epoch {epoch}")}, commit=False)

wandb.summary["test/iou"] = test_iou
wandb.finish()
```

When the run starts, RunRaccoon prints where everything goes:

```text
runraccoon: started run unet-baseline (k3x9q2ab) -> outputs/unet-baseline/runraccoon/run-20261002_105640-k3x9q2ab
runraccoon: live dashboard -> http://127.0.0.1:8473/#run=k3x9q2ab
```

`finish()` prints a short wandb-style summary with sparklines and the paths to the plots.

Runnable examples are in [`examples/`](examples):

| Example | What it shows |
|---|---|
| [`minimal.py`](examples/minimal.py) | No framework needed |
| [`pytorch_loop.py`](examples/pytorch_loop.py) | A plain PyTorch loop |
| [`ultralytics_yolo.py`](examples/ultralytics_yolo.py) | YOLO training through Ultralytics' wandb integration |

## What you get on disk

Runs go in a `runraccoon/` folder inside the directory passed as `init(dir=...)`. Without
`dir`, they go in the current working directory. Ultralytics passes its own output folder
(`project/name`), so YOLO runs land next to `weights/` and `results.csv`.

Inside the run folder, the layout and file names are the same as wandb's:

```text
<dir>/runraccoon/
├── debug.log                         -> latest run's logs/debug.log
├── latest-run                        -> run-20260924_083909-t0rb91yy
└── run-20260924_083909-t0rb91yy/
    ├── files/
    │   ├── config.yaml               wandb format: {key: {value: ...}}
    │   ├── wandb-summary.json        final / best value of every key
    │   ├── wandb-metadata.json       host, GPUs, git commit, argv, python, ...
    │   ├── wandb-history.jsonl       every logged step, one JSON object per line
    │   ├── output.log                captured stdout/stderr (progress bars collapsed)
    │   ├── requirements.txt          pip freeze of the environment
    │   ├── media/
    │   │   ├── images/<key>_<step>_<sha20>.<ext>        e.g. qc/contact_sheet_73_fff6edfaacd84552d328.jpg
    │   │   └── table/<key>_<step>_<sha20>.table.json    e.g. curves/F1-Confidence(B)_table_302_....table.json
    │   └── plots/                    <- RunRaccoon's figures (next section)
    ├── logs/debug.log, logs/plots.log
    └── artifacts/<name>/manifest.json
```

wandb stores history in a binary `.wandb` file. RunRaccoon writes `wandb-history.jsonl`
instead, the plain-text name older wandb versions used. You can open it with any tool, or
use `plots/history.csv`.

## The figures

All figures go in `files/plots/`:

| File | When | What |
|---|---|---|
| `progress.png` | about once per epoch while training | The important metrics only: paired train/val/test losses, validation scores, your QC numbers. Learning rates, counters and model info are left out. |
| `summary/run_summary.png` | at `finish()` | A report card: headline numbers on top (test results, best validation scores and the epoch they occurred), key curves below. |
| `summary/<section>.png` | at `finish()` | Every chart, one figure per wandb panel section (`loss`, `performance`, `metrics`, `lr`, `val_px`, ...). |
| `charts/<key>.png` | when logged | Custom charts: `wandb.plot.*` and `plot_table`, e.g. Ultralytics' PR/F1 curves and confusion matrices. |
| `history.csv` | at `finish()` | Every scalar, one row per step, for Excel / pandas / R. |

How the plots are put together:

- **Pairing.** `train/loss`, `val/loss` and `test/loss` share one panel, and so do Keras-style
  `loss`/`val_loss` and `train_acc`/`val_acc`.
- **Colors and line styles.** Train is blue solid, val is orange dashed, test is green dotted.
  The colors are checked to be colorblind-safe, and the dashes keep the figure readable in
  grayscale print. Curves with many points are drawn solid and thinner so dashes don't turn
  into noise.
- **Best point.** For each metric whose direction is known, the best value is circled and
  labeled, e.g. `min 0.108 @ epoch 278`. Losses and errors are minimized; accuracy, mAP, IoU,
  F1 and similar are maximized. You can override this with `define_metric`.
- **Scales.** Losses that span more than ~30× switch to a log axis. Series longer than 400
  points get light smoothing drawn over the faint raw line.
- **Typography.** Text stays in vector form in PDF/SVG output (`pdf.fonttype 42`), so journal
  checks and Illustrator edits work. Add `plot_formats=("png", "pdf")` to get vector copies.

To re-render the figures at any time, for example after a crash or after changing settings:

```bash
runraccoon replot path/to/outputs/unet-baseline --formats png,pdf
```

## The dashboard

![Dashboard](https://raw.githubusercontent.com/Gene-Weaver/RunRaccoon/main/docs/images/dashboard.png)

The first run on the machine starts a small server on `http://127.0.0.1:8473` (localhost only).
Every later run, in any process or environment, registers with it.

**Left panel**
- *Active* lists every run currently training, so several GPUs or jobs each get their own entry.
- *Past* lists finished, failed, and crashed runs.

**Tabs for the selected run**
- **Charts:** the same panels as the figures, with hover tooltips, a smoothing slider, a
  metric filter, and a table view on every chart. Live runs update every few seconds.
- **Media:** every logged image with a step slider, like wandb's image panels.
- **Plots:** the rendered PNGs from `files/plots/`.
- **Summary / Config:** searchable key/value tables.
- **Logs:** the tail of `output.log`.

The auto-started server shuts itself down after an hour with no active runs. To browse past
runs at any time:

```bash
runraccoon dashboard            # opens your browser
```

The dashboard follows your system's light/dark setting. Set `RUNRACCOON_DASHBOARD=0` to turn
off the auto-start.

## Ultralytics / YOLO

Ultralytics' wandb callback runs `import wandb` internally. Importing RunRaccoon registers it
as the `wandb` module, so the callback logs to RunRaccoon instead:

- per-epoch losses, metrics and learning rates,
- `train_batch*.jpg`, `val_batch*_pred.jpg`, `labels.jpg` and `results.png`,
- confusion matrices and PR/F1/P/R curves (saved as tables and rendered to `plots/charts/`),
- the best weights, recorded as an artifact manifest.

```python
import runraccoon as wandb          # before model.train()
from ultralytics import YOLO, settings

settings.update({"wandb": True})
YOLO("yolo11n.pt").train(data="coco8.yaml", epochs=10, project="runs", name="exp")
```

Ultralytics' YOLO `resume=True` looks for a previous run in `save_dir/wandb/latest-run`. With
the `runraccoon/` folder, a resumed YOLO training therefore starts a new RunRaccoon run instead
of appending to the old one. Calling `wandb.init(id=..., resume="must")` yourself is not
affected. Set `RUNRACCOON_DIRNAME=wandb` if you need Ultralytics' automatic resume to find the
earlier run.

Importing RunRaccoon also sets `WANDB_MODE=disabled` for child processes. If something
launches the real wandb in a subprocess, such as multi-GPU DDP workers, it stays offline.

## Swapping an existing wandb script

Most scripts need only the import change. For example, here is
`Honey/annotation_app_build/train_yolo26_pose.py`:

```diff
-        import wandb
+        import runraccoon as wandb
```

Everything else in that script works unchanged:
- `wandb.init(..., dir=run_dir)`,
- the custom `EpochQC` callback logging `qc/contact_sheet` images and `val_px/*` at `step=epoch`,
- Ultralytics finishing the run, then `wandb.init(id=..., resume="must")` reopening it to add
  `test/*` to the summary.

Resuming creates a new `run-<time>-<same id>` directory, as wandb does. RunRaccoon carries the
earlier history and media into it, so the figures cover the whole run.

To switch every script in an environment without editing each import, add this near the
start of your entry point:

```python
import runraccoon  # noqa: F401  (registers itself as `wandb`)
```

From then on, any `import wandb` in that process returns RunRaccoon.

## API compatibility

| wandb | RunRaccoon |
|---|---|
| `init(project, name, config, dir, id, group, job_type, tags, notes, resume, reinit, mode, settings, ...)` | ✓ (`entity` and cloud-only options are accepted and ignored) |
| `log(data, step=, commit=)` | ✓ same step semantics; a late write to an already-committed step is kept, where wandb would drop it |
| `run.config`, `wandb.config.update(...)`, argparse `Namespace` | ✓ |
| `run.summary[...]`, `summary.update(...)` | ✓ |
| `define_metric(name, step_metric=, summary=, goal=, hidden=)` | ✓ also controls the plots (x-axis, best marker, hidden) |
| `Image` (path, PIL, numpy HW/HWC/CHW, torch tensor, matplotlib figure; `caption`) | ✓ (`boxes=`/`masks=` overlays are accepted but not drawn) |
| lists of `Image` under one key | ✓ (`images/separated`, as wandb) |
| `Table`, `plot.line`, `plot.line_series`, `plot.scatter`, `plot.bar`, `plot.histogram`, `plot.pr_curve`, `plot.roc_curve`, `plot.confusion_matrix`, `plot_table` | ✓ saved as tables and rendered to PNG |
| `Histogram`, raw arrays | ✓ stored in history (not plotted) |
| `Artifact`, `log_artifact`, `log_model` | ✓ local manifest (files referenced, optionally copied) |
| `save(glob)`, `alert`, `finish(exit_code)`, `run.dir`, `run.id`, `run.name`, `run.step`, context manager | ✓ |
| `login`, `watch`, `unwatch`, `Settings(...)` | accepted, no-op |
| `Video`, `Audio`, `Html` | ✓ from a file path |
| `Api()`, `use_artifact(...).download()` from a server, sweeps, reports | ✗ local-only; these raise a clear error |

## Settings and environment variables

Pass settings in code with `wandb.init(settings=wandb.Settings(plot_every_s=30))` (or a plain
dict), or set them through the environment:

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `dashboard` | `RUNRACCOON_DASHBOARD` | `1` | auto-start the localhost dashboard |
| `dashboard_port` | `RUNRACCOON_PORT` | `8473` | dashboard port (also used by `runraccoon dashboard`) |
| `live_plots` | `RUNRACCOON_LIVE_PLOTS` | `1` | refresh `progress.png` during training |
| `plot_every_s` | `RUNRACCOON_PLOT_EVERY_S` | `10` | minimum seconds between refreshes (at most once per epoch either way) |
| `final_plots` | `RUNRACCOON_FINAL_PLOTS` | `1` | render the summary figures in `finish()` |
| `plot_formats` | `RUNRACCOON_PLOT_FORMATS` | `png` | e.g. `png,pdf,svg` for the final figures |
| `live_metrics` | `RUNRACCOON_LIVE_METRICS` | auto | glob patterns choosing the progress-plot metrics, e.g. `train/*,val/*` |
| `dirname` | `RUNRACCOON_DIRNAME` | `runraccoon` | name of the run folder created inside the output dir |
| `console` | `RUNRACCOON_CONSOLE` | `wrap` | `off` to skip `output.log` capture |
| `artifact_copy` | `RUNRACCOON_ARTIFACT_COPY` | `0` | copy artifact files instead of referencing them |
| `quiet` | `RUNRACCOON_QUIET` | `0` | silence RunRaccoon's console messages |
| | `RUNRACCOON_HOME` | `~/.runraccoon` | where the run index and dashboard log live |
| | `RUNRACCOON_SHIM` | `1` | `0` to stop `import runraccoon` from registering as `wandb` |

The usual wandb variables are honored as defaults: `WANDB_PROJECT`, `WANDB_NAME`,
`WANDB_RUN_ID`, `WANDB_RUN_GROUP`, `WANDB_JOB_TYPE`, `WANDB_TAGS`, `WANDB_NOTES`, `WANDB_DIR`,
`WANDB_RESUME`, and `WANDB_MODE=disabled`, which turns logging off entirely.

## Command line

```text
runraccoon dashboard [--port 8473]     open the dashboard (all runs, live + past)
runraccoon ls [--all]                  list runs and their status
runraccoon replot <run dir | id>       re-render every figure for a run  [--formats png,pdf]
runraccoon register <folder>...        add existing/moved run folders to the dashboard
runraccoon forget <id>...              remove runs from the dashboard index (files are kept)
runraccoon gc                          forget runs whose folders were deleted
```

`python -m runraccoon ...` works the same way.

## How it works

```text
 training script                      runraccoon/run-<time>-<id>/files/
 runraccoon.init / log  ── appends ─▶  history · summary · config · media
        │                                     ▲              ▲
        │ spawns ≤ 1× per epoch               │ reads        │ reads
        ▼                                     │              │
 renderer subprocess (matplotlib) ────────────┘              │
        │ writes files/plots/*.png                           │
        │                                                    │
        └ heartbeat ─▶ ~/.runraccoon/runs/<id>.json ◀─ reads ─ dashboard server
                                                              127.0.0.1:8473
```

- **The training process only writes files.** Media is hashed and written when logged, and
  history rows are appended as each step is committed. Plotting runs in a short-lived
  subprocess, so it never competes with training for the GIL and never touches your
  script's matplotlib state.
- **One module decides what gets plotted.** `runraccoon/panels.py` handles key pairing,
  sections, the x-axis, and the better direction. The PNG renderer and the dashboard both use
  it, so they always agree.
- **The dashboard is read-only.** It reads the same files and the per-run heartbeat records.
  It therefore works across processes, Python environments, and runs that crashed.

The code is organized by job:

| Module | Job |
|---|---|
| `sdk.py` | Module-level `wandb.*` functions |
| `run.py` | The `Run` class and step semantics |
| `media.py` | `Image`, `Table`, ... |
| `plot.py` | `wandb.plot` |
| `paths.py` | Directory layout |
| `reader.py` | Reading a run back from disk |
| `registry.py` | The machine-wide run index |
| `plotting/` | Style, figures, renderer, scheduler |
| `dashboard/` | Server and static app |

## Limitations

- Nothing is uploaded, ever. No sweeps, reports, team sharing, or `wandb.Api()`.
- Image overlays (`boxes=`, `masks=`) are not drawn. Render them into the image before logging.
- System metrics (GPU utilization over time) are not tracked. The GPU model and count are
  recorded in `wandb-metadata.json`.
- `output.log` captures Python-level `print`/logging. Output written directly by C extensions
  to file descriptor 1/2 is not captured.
- If the real `wandb` was imported *before* RunRaccoon, code already holding that module is
  not redirected (RunRaccoon prints a warning). Import RunRaccoon first.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests cover the on-disk layout, step semantics, media naming, resume, metric pairing,
rendering, and the dashboard API.

## Releasing to PyPI

The release flow matches VoucherVisionGO-client: `setup.py` holds the package metadata, and
builds go to `dist/`, which is gitignored.

1. Bump `__version__` in `runraccoon/_version.py`. This is the only place the version lives;
   `setup.py` and `runraccoon.__version__` both read it.
2. Add an entry to `CHANGELOG.md`.
3. Build and upload:

```bash
pip install build twine            # once
python -m build
python -m twine upload dist/* --skip-existing --verbose
```

`--skip-existing` lets `dist/` keep older builds without twine failing on versions already
on PyPI. Twine asks for credentials: use `__token__` as the username and a PyPI API token as
the password, or put them in `~/.pypirc`.

Before the first upload you can rehearse on TestPyPI with
`python -m twine upload --repository testpypi dist/*`.

License: MIT.
