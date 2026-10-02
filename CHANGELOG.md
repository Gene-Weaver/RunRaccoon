# Changelog

## 0.1.2

- **Fix: a run could hang forever in `finish()` (0.1.0 and 0.1.1).** Live plots were started
  with `subprocess.Popen` from a background thread. If PyTorch's DataLoader forked workers at
  that moment, they inherited Popen's internal pipe, which stalled the plot thread and then
  `finish()` behind it. Renders now start with `os.posix_spawn` (no such pipe) and no lock is
  held while spawning. System metadata is collected in `init()` itself, not in a thread.
  Upgrading is recommended.
- **QC contact sheets.** Ultralytics training now gets a per-epoch contact sheet automatically
  (`runraccoon/qc_contact_sheet`): fixed val images, ground truth in white, predictions in
  per-class colors (boxes, masks, keypoints + skeleton, OBB), plus a final sheet from best.pt.
  The overlay data is saved too, so sheets can be redrawn in other styles. For other code,
  `runraccoon.ContactSheet(images, captions=...)` logs any grid of images.
- **Training-progress GIFs.** In `finish()`, every image key logged at three or more steps
  becomes a 1080p GIF (landscape or portrait, 0.25 s per step by default), plus one GIF per
  panel. GIFs stay compact: one shared palette, ordered dithering, temporal denoising, and only
  changed pixels per frame. New command: `runraccoon gif <run>`.
- **Dashboard: GIFs tab** with autoplaying GIFs and a controller for speed, hold, orientation,
  resolution, line width, point size, opacity, mask fill, and per-class / keypoint / skeleton /
  ground-truth colors. It has a live preview and regenerates GIFs with a progress bar.
- **Dashboard: popup image viewer.** Images open large in place, with the step slider, ←/→ for
  steps, ↑/↓ for images within a step, and Esc to close. Plots and GIFs open there too.
- **Parallel GIF building:** one process per GIF, plus threads inside each for composing,
  palette building, and quantizing. The worker budget defaults to max(physical cores,
  threads) - 2 (`gif_workers` / `RUNRACCOON_GIF_WORKERS`, `runraccoon gif --workers`). Output
  is byte-identical to a single-threaded build. 13 1080p GIFs of 51 frames take 8 s on a
  32-core machine, versus 106 s on one core. The number of processes is capped by available
  memory, so 4K GIFs don't exhaust RAM.
- GIF settings saved from the dashboard are used everywhere: the end-of-run build,
  `runraccoon gif` (`--defaults` ignores them), and the dashboard. The controller has
  **Reset to defaults** and a **Show** popup that turns each class, keypoint, skeleton, and
  the ground truth on or off. Skeleton lines now follow the line width.
- **Plain PyTorch / DINOv2 / UNet support** (`runraccoon.integrations.segmentation`, `.dino`):
  - `SegMetrics` logs pixel accuracy, mean class accuracy, mIoU and Dice, plus per-class IoU and
    confusion-matrix charts. It works on full masks or on sparse point labels (weak supervision).
  - `semantic_qc_sheet` makes restyleable ground-truth | prediction contact sheets. Label maps are
    stored as PNGs; sparse points are drawn as class-colored dots.
  - `EmbeddingPCA` (PCA→RGB maps of patch tokens, colors stable across epochs) and `pca_scatter`
    (labeled embeddings in 2-D).
  - Charts now handle many classes: scatter colored by class with a legend, and bar charts and
    heatmaps that grow with the class count.
- **Dashboard: Star and Hide.** Click a run's status dot to star it (a gold ★ in the list and the
  header) or hide it. Hidden runs move to a "View hidden runs" drawer at the bottom of the sidebar,
  which scrolls on its own. Both are stored in the run index.
- **Dashboard: best checkpoint.** A magenta line on every chart marks the step that would be
  kept as best right now, and the legend sits next to the metric filter. Rules, in order:
  your `define_metric(..., summary="min"|"max")`; Ultralytics' best.pt fitness (matched to
  the installed version, 8.3.198 changed it); the lowest val/loss.
- **Dashboard: run list** shows `Start: … End: …`, or a projected `estEnd: …` once the first
  train+val epoch has finished. The sidebar is wider so it always fits.
- Runs started by Ultralytics itself use `WANDB_PROJECT` instead of the output path as
  their project name.
- A resumed run keeps its original start time in the run index.

## 0.1.1

- Default dashboard port changed from 8765 to 8473 (8765 is commonly taken by other local apps).
- `runraccoon dashboard` now honors `RUNRACCOON_PORT`, the same variable training runs use.
- Dashboard sidebar: removed the subtitle under the RunRaccoon name.
- Source distributions explicitly exclude `.pypirc` and `.env`.

## 0.1.0

- First release.
- wandb-compatible API: `init`, `log` (with wandb's step/commit semantics), `finish`, `config`,
  `summary`, `define_metric`, `Image`, `Table`, `Histogram`, `plot.*`, `plot_table`,
  `Artifact`, `log_artifact`, `save`, `alert`; `login` / `watch` accepted as no-ops.
- `import runraccoon` registers itself as `wandb`, so library integrations (Ultralytics' wandb
  callback) log locally and can never reach the cloud.
- Runs are written to `<output dir>/runraccoon/run-<time>-<id>/` using wandb's file names
  (`config.yaml`, `wandb-summary.json`, `wandb-metadata.json`, `output.log`,
  `media/images/<key>_<step>_<sha20>.<ext>`, ...), plus `wandb-history.jsonl`.
- Figures: `plots/progress.png` refreshed once per epoch, `plots/summary/run_summary.png`
  report card, one figure per metric section, custom charts, and `history.csv`; optional PDF/SVG.
- Localhost dashboard listing every active and past run on the machine, with live charts,
  an image step slider, rendered plots, summary, config and logs.
- `runraccoon` CLI: `dashboard`, `ls`, `replot`, `register`, `forget`, `gc`.
