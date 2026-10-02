"""The smallest useful RunRaccoon script - a fake training loop with train/val/test metrics and a
QC image every few epochs. No GPU or framework needed.

    python examples/minimal.py

Then look in examples/output/runraccoon/latest-run/files/plots/ or the dashboard URL printed at startup.
"""
import time
from pathlib import Path

import numpy as np

import runraccoon as wandb

# Your project's output dir; runs go to <OUTPUT_DIR>/runraccoon/run-<time>-<id>/
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

EPOCHS = 40

wandb.init(project="raccoon-examples", name="minimal", dir=OUTPUT_DIR,
           config={"epochs": EPOCHS, "lr": 3e-4, "batch": 16})
wandb.define_metric("val/loss", summary="min")         # optional: summary keeps the best, not the last

rng = np.random.default_rng(0)
for epoch in range(EPOCHS):
    train_loss = 1.8 * np.exp(-epoch / 10) + 0.12 + rng.normal(0, 0.03)
    val_loss = 2.0 * np.exp(-epoch / 12) + 0.22 + rng.normal(0, 0.05)
    val_iou = 0.93 - 0.75 * np.exp(-epoch / 9) + rng.normal(0, 0.01)
    wandb.log({"train/loss": train_loss, "val/loss": val_loss, "val/iou": val_iou,
               "lr": 3e-4 * 0.95 ** epoch, "epoch": epoch})

    if epoch % 5 == 0:                                   # QC image -> files/media/images/qc/overlay_<step>_<sha>.png
        img = rng.random((96, 160, 3)) * 0.3
        img[30:66, 40:120, 1] = min(1.0, val_iou)
        wandb.log({"qc/overlay": wandb.Image(img, caption=f"epoch {epoch}")}, commit=False)
    time.sleep(0.05)

# Held-out results usually go straight into the summary (they show up as headline tiles).
wandb.summary["test/iou"] = 0.91
wandb.summary["test/loss"] = 0.27

# Custom charts render to files/plots/charts/<key>.png
recall = np.linspace(0, 1, 50)
wandb.log({"curves/pr": wandb.plot.line_series(xs=recall, ys=[1 - recall ** 3, 1 - recall ** 2],
                                               keys=["leaf", "stem"], title="Precision-Recall", xname="recall")})
wandb.finish()
