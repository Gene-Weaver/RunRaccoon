"""Semantic segmentation (UNet / ResUNet / DINOv2 probe style) with RunRaccoon's helpers.

A plain training loop gets the same outputs an Ultralytics run gets: validation metrics, per-class
IoU and confusion-matrix charts, a QC contact sheet every epoch (ground truth | prediction) that
becomes a training-progress GIF, and - for DINOv2 / ViT backbones - patch-embedding PCA visuals.

The "model" here is simulated with numpy so the example runs anywhere:

    python examples/semantic_segmentation.py
"""
from pathlib import Path

import numpy as np

import runraccoon as wandb
from runraccoon.integrations.dino import EmbeddingPCA, pca_scatter
from runraccoon.integrations.segmentation import SegMetrics, semantic_qc_sheet

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CLASSES = ["background", "leaf", "petiole", "hole"]
EPOCHS = 12
rng = np.random.default_rng(0)


def toy_sample():
    """An image with a leaf (1), its petiole (2) and a hole (3), plus the true label map."""
    img = (np.full((96, 128, 3), 225) + rng.normal(0, 8, (96, 128, 3))).clip(0, 255).astype(np.uint8)
    yy, xx = np.mgrid[:96, :128]
    mask = np.zeros((96, 128), np.uint8)
    mask[((yy - 48) / 30) ** 2 + ((xx - 60) / 45) ** 2 < 1] = 1
    mask[(np.abs(yy - 48) < 3) & (xx > 100) & (xx < 125)] = 2
    mask[((yy - 40) / 6) ** 2 + ((xx - 50) / 6) ** 2 < 1] = 3
    img[mask == 1] = (120, 150, 60)
    img[mask == 2] = (90, 110, 40)
    return img, mask


def toy_prediction(mask, quality):
    """A prediction that gets better with `quality` (0..1)."""
    pred = mask.copy()
    noise = rng.random(mask.shape) > quality
    pred[noise] = rng.integers(0, len(CLASSES), noise.sum())
    return pred


val = [toy_sample() for _ in range(8)]
wandb.init(project="raccoon-examples", name="semantic-segmentation", dir=OUTPUT_DIR,
           config={"epochs": EPOCHS, "classes": CLASSES})
wandb.define_metric("val/mIoU", summary="max")              # drives the dashboard's best-checkpoint line

# DINOv2-style embedding visuals (here: fake 8x8 patch tokens that separate leaf from background).
tokens = rng.normal(size=(len(val), 64, 32)).astype(np.float32)
patch_labels = np.stack([m[6::12, 8::16].reshape(-1) for _, m in val])
tokens += patch_labels[..., None] * 2.0
pca = EmbeddingPCA().fit(tokens)
wandb.log({"dino/pca": pca.sheet([img for img, _ in val], tokens, grid=(8, 8)),
           "dino/embedding_scatter": pca_scatter(tokens, patch_labels, CLASSES)}, step=0)

metrics = SegMetrics(len(CLASSES), CLASSES, exclude_from_mean=[0])
for epoch in range(1, EPOCHS + 1):
    quality = 0.35 + 0.6 * epoch / EPOCHS
    preds = [toy_prediction(m, quality) for _, m in val]
    metrics.reset()
    for (img, m), p in zip(val, preds):
        metrics.update(p, m)
    wandb.log({"train/loss": 1.5 * (1 - quality) + 0.05, "val/loss": 1.6 * (1 - quality) + 0.08, "epoch": epoch,
               "qc/val": semantic_qc_sheet([img for img, _ in val], [m for _, m in val], preds, CLASSES,
                                           background=0, title=f"epoch {epoch}")},
              step=epoch, commit=False)
    metrics.log(prefix="val", step=epoch)                    # val/mIoU, val/pixel_acc, ... + charts

wandb.finish()                                               # figures + GIFs in output/runraccoon/latest-run/files/plots
