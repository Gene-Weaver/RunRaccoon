# Changelog

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
