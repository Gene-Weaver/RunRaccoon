"""DINOv2 (and other ViT) embedding visuals: PCA of patch tokens.

    from runraccoon.integrations.dino import EmbeddingPCA, pca_scatter

    pca = EmbeddingPCA().fit(tokens)                        # (B, N, D) patch tokens of reference images
    wandb.log({"dino/pca": pca.sheet(images, tokens, grid=(16, 16))})   # image | PCA->RGB per tile
    wandb.log({"dino/embedding_scatter": pca_scatter(point_tokens, point_labels, class_names)})

`EmbeddingPCA` is fit once and then reused, so a color means the same direction in embedding
space at every epoch - logging `pca.sheet(...)` each epoch of a fine-tuned backbone therefore makes
a meaningful training-progress GIF. With a frozen backbone, log it once.

DINOv2 patch tokens: `model.forward_features(x)["x_norm_patchtokens"]`, grid = (H // 14, W // 14).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


class EmbeddingPCA:
    """PCA of patch embeddings -> 3 components -> RGB (robustly scaled, stable across calls)."""

    def __init__(self, n_components: int = 3, max_fit_tokens: int = 200_000, seed: int = 0):
        self.k, self.max_fit, self.seed = n_components, max_fit_tokens, seed
        self.mean = self.components = self.lo = self.hi = None

    def fit(self, tokens: Any) -> "EmbeddingPCA":
        x = _np(tokens).reshape(-1, _np(tokens).shape[-1])
        if len(x) > self.max_fit:
            x = x[np.random.default_rng(self.seed).choice(len(x), self.max_fit, replace=False)]
        self.mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[: self.k]
        proj = (x - self.mean) @ self.components.T
        self.lo, self.hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
        return self

    def transform(self, tokens: Any) -> np.ndarray:
        if self.components is None:
            self.fit(tokens)
        x = _np(tokens)
        return (x.reshape(-1, x.shape[-1]) - self.mean) @ self.components.T

    def rgb(self, tokens: Any, grid: tuple[int, int]) -> list[np.ndarray]:
        """(B, N, D) tokens -> B uint8 images of shape (grid_h, grid_w, 3)."""
        x = _np(tokens)
        b = x.shape[0] if x.ndim == 3 else 1
        proj = self.transform(x)[:, :3]
        scaled = np.clip((proj - self.lo[:3]) / np.maximum(self.hi[:3] - self.lo[:3], 1e-8), 0, 1)
        return list((scaled.reshape(b, grid[0], grid[1], 3) * 255).astype(np.uint8))

    def sheet(self, images: Sequence[Any], tokens: Any, grid: tuple[int, int], captions: Sequence[str] | None = None,
              mean=None, std=None, title: str | None = "patch-embedding PCA (first 3 components as RGB)",
              cols: int = 4, tile_px: int = 720):
        """Contact sheet: each tile is the image next to its PCA->RGB map (upsampled, nearest)."""
        from PIL import Image as PILImage

        from runraccoon.integrations.segmentation import _image_rgb
        from runraccoon.qc import ContactSheet
        maps = self.rgb(tokens, grid)
        tiles = []
        for img, m in zip(images, maps):
            rgb = PILImage.fromarray(_image_rgb(img, mean, std))
            pca = PILImage.fromarray(m).resize(rgb.size, PILImage.Resampling.NEAREST)
            pair = PILImage.new("RGB", (rgb.width * 2 + 6, rgb.height), (18, 18, 18))
            pair.paste(rgb, (0, 0))
            pair.paste(pca, (rgb.width + 6, 0))
            tiles.append(pair)
        return ContactSheet(tiles, captions=captions, cols=cols, tile_px=tile_px, title=title)


def pca_scatter(tokens: Any, labels: Sequence[int], class_names: Sequence[str] | dict, max_points: int = 6000,
                pca: EmbeddingPCA | None = None, title: str = "patch embeddings, PCA (labeled points)", seed: int = 0):
    """2-D PCA scatter of labeled embeddings, one color per class (a custom chart)."""
    from runraccoon.media import Table
    from runraccoon.plot import CustomChart
    x = _np(tokens).reshape(-1, _np(tokens).shape[-1])
    y = np.asarray(labels).reshape(-1)
    if len(x) > max_points:
        idx = np.random.default_rng(seed).choice(len(x), max_points, replace=False)
        x, y = x[idx], y[idx]
    p = pca if pca is not None and pca.components is not None else EmbeddingPCA(2).fit(x)
    xy = p.transform(x)[:, :2]
    names = class_names if isinstance(class_names, dict) else dict(enumerate(class_names))
    rows = [[str(names.get(int(c), c)), round(float(a), 4), round(float(b), 4)] for (a, b), c in zip(xy, y)]
    order = [str(names[k]) for k in sorted(names)]       # class-id order -> same colors as the QC overlays
    return CustomChart(Table(columns=["class", "pc1", "pc2"], data=rows), "scatter",
                       {"x": "pc1", "y": "pc2", "series": "class", "series_order": order},
                       title=title, x_title="PC 1", y_title="PC 2")
