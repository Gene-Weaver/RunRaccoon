"""Semantic-segmentation helpers for plain PyTorch training loops (UNet, ResUNet, DINOv2 probes, ...).

These give a hand-written training loop the outputs Ultralytics runs get automatically:
per-epoch validation metrics, a confusion matrix, per-class charts and a QC contact sheet that
becomes a training-progress GIF (with restylable colors / per-class visibility).

    from runraccoon.integrations.segmentation import SegMetrics, semantic_qc_sheet

    metrics = SegMetrics(num_classes, class_names=names, ignore_index=255)
    for epoch in range(epochs):
        ...train...
        metrics.reset()
        for images, masks in val_loader:
            metrics.update(model(images), masks)                 # logits (N,C,H,W) or labels (N,H,W)
        metrics.log(run, prefix="val", step=epoch + 1)          # val/mIoU, val/pixel_acc, ... + charts
        run.log({"qc/val": semantic_qc_sheet(val_images, val_masks, preds, names)}, step=epoch + 1)

Weak supervision: when only sparse points are labeled, pass the predicted labels *at those points*
and the point labels: `metrics.update(pred_at_points, point_labels)` - any shapes, flattened.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def to_labels(pred: Any) -> np.ndarray:
    """Logits / probabilities (N,C,H,W) or (C,H,W) -> label maps; label maps pass through."""
    a = _np(pred)
    if a.dtype.kind == "f" and a.ndim >= 3:
        return a.argmax(axis=-3)
    return a.astype(np.int64)


class SegMetrics:
    """Confusion-matrix accumulator: pixel accuracy, mean class accuracy, mIoU, mean Dice."""

    def __init__(self, num_classes: int, class_names: Sequence[str] | dict | None = None,
                 ignore_index: int | None = None, exclude_from_mean: Sequence[int] = ()):
        self.num_classes = int(num_classes)
        if isinstance(class_names, dict):
            class_names = [str(class_names.get(i, i)) for i in range(self.num_classes)]
        self.names = list(class_names) if class_names is not None else [str(i) for i in range(self.num_classes)]
        self.ignore_index = ignore_index
        self.exclude = set(exclude_from_mean)          # e.g. a background class that would inflate the mean
        self.reset()

    def reset(self) -> None:
        self.cm = np.zeros((self.num_classes, self.num_classes), np.int64)

    def update(self, pred: Any, target: Any) -> None:
        p = to_labels(pred).reshape(-1)
        t = _np(target).astype(np.int64).reshape(-1)
        keep = (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
        if self.ignore_index is not None:
            keep &= t != self.ignore_index
        self.cm += np.bincount(self.num_classes * t[keep] + p[keep], minlength=self.num_classes ** 2
                               ).reshape(self.num_classes, self.num_classes)

    def per_class(self) -> dict:
        tp = np.diag(self.cm).astype(float)
        gt, pr = self.cm.sum(1).astype(float), self.cm.sum(0).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (gt + pr - tp)
            dice = 2 * tp / (gt + pr)
            acc = tp / gt
        return {"iou": iou, "dice": dice, "acc": acc, "support": gt}

    def compute(self) -> dict:
        pc = self.per_class()
        present = [i for i in range(self.num_classes) if pc["support"][i] > 0 and i not in self.exclude]
        total = self.cm.sum()

        def mean(v):
            vals = [v[i] for i in present if np.isfinite(v[i])]
            return float(np.mean(vals)) if vals else float("nan")
        return {"pixel_acc": float(np.trace(self.cm) / total) if total else float("nan"),
                "mean_acc": mean(pc["acc"]), "mIoU": mean(pc["iou"]), "mDice": mean(pc["dice"]),
                "classes_present": len(present), "n": int(total)}

    def log(self, run=None, prefix: str = "val", step: int | None = None, charts: bool = True) -> dict:
        """Log the scalars (`<prefix>/mIoU` ...) and, if `charts`, a per-class IoU bar chart and a
        row-normalized confusion matrix (rendered to files/plots/charts/)."""
        if run is None:
            import runraccoon
            run = runraccoon.run
        m = self.compute()
        data: dict = {f"{prefix}/{k}": v for k, v in m.items() if k not in ("classes_present", "n")}
        if charts:
            from runraccoon.media import Table
            from runraccoon.plot import CustomChart
            pc = self.per_class()
            rows = [[self.names[i], round(float(pc["iou"][i]), 4)] for i in range(self.num_classes)
                    if pc["support"][i] > 0 and np.isfinite(pc["iou"][i])]
            rows.sort(key=lambda r: -r[1])
            data[f"{prefix}_charts/per_class_iou"] = CustomChart(Table(columns=["class", "IoU"], data=rows), "bar",
                                                                 {"label": "class", "value": "IoU"},
                                                                 title=f"{prefix} IoU per class", x_title="IoU")
            used = [i for i in range(self.num_classes) if self.cm[i].sum() or self.cm[:, i].sum()]
            norm = self.cm[np.ix_(used, used)].astype(float)
            norm = norm / np.maximum(norm.sum(1, keepdims=True), 1)
            cells = [[self.names[used[c]], self.names[used[r]], round(float(norm[r, c]), 3)]
                     for r in range(len(used)) for c in range(len(used))]
            data[f"{prefix}_charts/confusion_matrix"] = CustomChart(
                Table(columns=["Predicted", "Actual", "fraction"], data=cells), "heatmap",
                {"x": "Predicted", "y": "Actual", "value": "fraction"}, title=f"{prefix} confusion (row-normalized)",
                x_title="Predicted", y_title="Actual")
        run.log(data, step=step)
        return m


def _image_rgb(img: Any, mean: Sequence[float] | None = None, std: Sequence[float] | None = None) -> np.ndarray:
    """A CHW / HWC tensor or array (optionally ImageNet-normalized) -> HxWx3 uint8."""
    a = _np(img).astype(np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = a.transpose(1, 2, 0)
    if mean is not None and std is not None:
        a = a * np.asarray(std, np.float32) + np.asarray(mean, np.float32)
    if a.max() <= 1.5:
        a = a * 255.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    return np.repeat(a, 3, axis=2) if a.shape[-1] == 1 else a


def semantic_qc_sheet(images: Sequence[Any], gt_masks: Sequence[Any] | None, preds: Sequence[Any],
                      class_names: Sequence[str] | dict, points: Sequence[Sequence] | None = None,
                      background: int | None = None, ignore_index: int | None = None,
                      captions: Sequence[str] | None = None, mean=None, std=None, title: str | None = None,
                      subtitle: str | None = None, cols: int | None = 4, tile_px: int = 720,
                      labels: tuple[str, str] = ("ground truth", "prediction")):
    """A restylable QC contact sheet: per tile, ground truth | prediction side by side.

    images:    tensors / arrays (CHW or HWC; pass `mean`/`std` to undo ImageNet normalization)
    gt_masks:  label maps (or None when only sparse points exist)
    preds:     logits (C,H,W) or label maps (H,W)
    points:    per image, sparse labels [(x, y, class), ...] in pixels of that image
    labels:    captions of the two halves, e.g. ("tile", "pseudo-mask")
    """
    from runraccoon.qc import QCSheet
    names = class_names if isinstance(class_names, dict) else {i: n for i, n in enumerate(class_names)}
    imgs, tiles = [], []
    for i, img in enumerate(images):
        rgb = _image_rgb(img, mean, std)
        h, w = rgb.shape[:2]
        tile: dict = {"split": True, "gt": {}, "pred": {"mask": to_labels(preds[i]).astype(np.uint8)},
                      "caption": captions[i] if captions else None}
        if tuple(labels) != ("ground truth", "prediction"):
            tile["labels"] = list(labels)
        if gt_masks is not None and gt_masks[i] is not None:
            tile["gt"]["mask"] = _np(gt_masks[i]).astype(np.uint8)
        if points is not None and points[i]:
            tile["gt"]["points"] = [[round(float(x) / w, 5), round(float(y) / h, 5), int(c)] for x, y, c in points[i]]
        imgs.append(rgb)
        tiles.append(tile)
    meta = {"names": {int(k): str(v) for k, v in names.items()}, "task": "semantic",
            "background": background, "ignore_index": ignore_index}
    return QCSheet(imgs, tiles, meta, cols=cols, tile_px=tile_px, title=title, subtitle=subtitle)
