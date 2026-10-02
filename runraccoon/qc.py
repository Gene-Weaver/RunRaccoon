"""Contact sheets: one image summarizing many QC images, logged every epoch.

`ContactSheet` is a loggable media type that lays out any images as a captioned grid:

    wandb.log({"qc/samples": runraccoon.ContactSheet(images, captions=names, title=f"epoch {ep}")})

It also records where every tile sits, so the per-panel GIFs made at the end of the run cut
tiles exactly instead of guessing.

For Ultralytics training, RunRaccoon makes one automatically (see integrations/ultralytics.py):
fixed val images, ground truth in white, predictions in Ultralytics' colors, every epoch.
"""
from __future__ import annotations

import io
import math
from typing import Any, Sequence

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from runraccoon.media import Image, _to_pil

BG = (18, 18, 18)
INK = (240, 240, 236)
MUTED = (160, 159, 152)


def _font(size: int):
    try:
        from matplotlib import font_manager
        for family in ("Roboto", "Inter", "Liberation Sans", "DejaVu Sans"):
            path = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            if path:
                return ImageFont.truetype(path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def _as_pil(x: Any) -> PILImage.Image:
    if isinstance(x, PILImage.Image):
        return x.convert("RGB")
    if isinstance(x, str) or hasattr(x, "__fspath__"):
        return PILImage.open(x).convert("RGB")
    if isinstance(x, Image):
        if x._path is not None:
            return PILImage.open(x._path).convert("RGB")
        return PILImage.open(io.BytesIO(x._bytes)).convert("RGB")
    return _to_pil(x).convert("RGB")


def build_sheet(images: Sequence[Any], captions: Sequence[str] | None = None, cols: int | None = None,
                tile_px: int = 720, title: str | None = None, subtitle: str | None = None,
                gap: int = 8) -> tuple[PILImage.Image, list[list[int]]]:
    """Compose the grid. Returns (sheet, tile boxes [l, t, r, b] in reading order)."""
    tiles = []
    for im in images:
        im = _as_pil(im)
        k = tile_px / im.width
        tiles.append(im.resize((tile_px, max(1, round(im.height * k))), PILImage.Resampling.LANCZOS))
    n = len(tiles)
    cols = cols or min(n, max(1, round(math.sqrt(n * 16 / 9 * (tiles[0].height / tile_px)))))
    rows = math.ceil(n / cols)
    cap_h = 26 if captions else 0
    tile_h = max(t.height for t in tiles)
    head = 0
    if title or subtitle:
        head = 16 + (34 if title else 0) + (24 if subtitle else 0)
    W = cols * tile_px + (cols + 1) * gap
    H = head + rows * (tile_h + cap_h) + (rows + 1) * gap
    sheet = PILImage.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    y = 10
    if title:
        d.text((gap + 2, y), title, fill=INK, font=_font(26))
        y += 34
    if subtitle:
        d.text((gap + 2, y), subtitle, fill=MUTED, font=_font(18))
    small = _font(16)
    boxes = []
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = gap + c * (tile_px + gap)
        yy = head + gap + r * (tile_h + cap_h + gap)
        sheet.paste(t, (x, yy))
        boxes.append([x, yy, x + t.width, yy + t.height])
        if captions and i < len(captions) and captions[i]:
            text = str(captions[i])
            while d.textlength(text, font=small) > tile_px - 8 and len(text) > 4:
                text = text[:-2]
            d.text((x + 2, yy + t.height + 4), text if text == str(captions[i]) else text[:-1] + "…", fill=MUTED, font=small)
    return sheet, boxes


class ContactSheet(Image):
    """A grid of images logged as one image (format jpg by default)."""

    def __init__(self, images: Sequence[Any], captions: Sequence[str] | None = None, cols: int | None = None,
                 tile_px: int = 720, title: str | None = None, subtitle: str | None = None,
                 file_type: str = "jpg", caption: str | None = None):
        if not images:
            raise ValueError("ContactSheet needs at least one image")
        sheet, self.boxes = build_sheet(images, captions, cols, tile_px, title, subtitle)
        super().__init__(sheet, caption=caption, file_type=file_type)

    def _bind(self, files_dir, key: str, step: int) -> dict:
        ref = super()._bind(files_dir, key, step)
        ref["_runraccoon_sheet"] = {"boxes": self.boxes}                # exact tile positions for panel GIFs
        return ref


# ================================================================== restylable QC overlays
# A QC sheet logged by RunRaccoon (QCSheet) keeps, besides the rendered jpg:
#   media/qc/<key>/base_<sha>.jpg   each QC image once (<= 1600 px), without overlays
#   media/qc/<key>/step_<n>.json    per step: ground truth + predictions for every tile
# Coordinates are normalized (0-1), so overlays can be redrawn at any size and in any style
# (GIF regeneration from the dashboard uses this).

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

DEFAULT_STYLE = {
    "line_width": 2.0,      # px on a 720 px tile; scales with the drawn size
    "point_size": 4.0,      # keypoint radius, same scale
    "opacity": 0.95,        # strokes and points
    "fill_opacity": 0.30,   # mask fill
    "show_gt": True,
    "show_pred": True,
    "show_labels": True,
    "halo": True,           # dark outline under ground truth so it reads on any background
    "colors": {},           # "gt", "skeleton", "class:<id>", "kpt:<j>" -> "#rrggbb"
    "hidden": [],           # the same feature ids, not drawn at all (a class hides its gt and predictions)
}


def resolve_style(style: dict | None) -> dict:
    out = {**DEFAULT_STYLE, **(style or {})}
    out["colors"] = {**(style or {}).get("colors", {})}
    out["hidden"] = list((style or {}).get("hidden") or [])
    return out


def feature_color(style: dict, feature: str, index: int = 0) -> str:
    c = style["colors"].get(feature)
    if c:
        return c
    if feature == "gt":
        return "#ffffff"
    if feature == "skeleton":
        return "#f2f2ee"
    return PALETTE[index] if index < len(PALETTE) else _extra_hue(index - len(PALETTE))


def _extra_hue(i: int) -> str:
    """Distinct colors past the 8-slot palette (semantic ontologies can have dozens of classes):
    golden-angle hue steps at a fixed, mid lightness. Every color can be changed in the dashboard."""
    import colorsys
    h = (0.11 + i * 0.381966) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.55 if i % 2 else 0.45, 0.70)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _rgba(hex_color: str, alpha: float) -> tuple:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(max(0.0, min(1.0, alpha)) * 255)


def qc_features(meta: dict, tiles: list[dict]) -> list[dict]:
    """Every overlay feature actually present (for the color picker): gt, classes, keypoints, skeleton."""
    names = {int(k): v for k, v in (meta.get("names") or {}).items()}
    skip = {meta.get("background"), meta.get("ignore_index")}
    present, has_gt, has_kpts = set(), False, False
    for t in tiles:
        for side in ("gt", "pred"):
            d = t.get(side) or {}
            for b in d.get("boxes", []):
                present.add(int(b[4]))
            for p in d.get("polygons", []):
                present.add(int(p[0]))
            for pt in d.get("points", []):
                present.add(int(pt[2]))
            present.update(int(c) for c in d.get("classes", []) if c not in skip)
            has_kpts |= bool(d.get("keypoints"))
            has_gt |= side == "gt" and not t.get("split") and any(d.get(k) for k in ("boxes", "polygons", "keypoints"))
    style = resolve_style(None)
    feats = [{"id": "gt", "label": "Ground truth", "color": feature_color(style, "gt")}] if has_gt else []
    for c in sorted(present):
        feats.append({"id": f"class:{c}", "label": f"class: {names.get(c, c)}", "color": feature_color(style, f"class:{c}", c)})
    if has_kpts:
        k = int((meta.get("kpt_shape") or [0])[0] or 0)
        kn = meta.get("kpt_names") or []
        for j in range(k):
            feats.append({"id": f"kpt:{j}", "label": f"keypoint: {kn[j] if j < len(kn) else j}",
                          "color": feature_color(style, f"kpt:{j}", j)})
        if meta.get("skeleton"):
            feats.append({"id": "skeleton", "label": "skeleton lines", "color": feature_color(style, "skeleton")})
    return feats


def draw_overlays(img: PILImage.Image, tile: dict, meta: dict, style: dict | None = None,
                  files_dir=None) -> PILImage.Image:
    """Draw ground truth + predictions onto `img` at its current size.

    Instance data (boxes / polygons / keypoints): ground truth as outlines, predictions on top.
    Semantic data (`mask` label maps, sparse `points`): class-colored fills with boundaries.
    `tile["split"]`: ground truth | prediction side by side (sparse points are shown on both).
    """
    st = resolve_style(style)
    if tile.get("split"):
        gt, pred = tile.get("gt") or {}, tile.get("pred") or {}
        pts = gt.get("points", [])
        left = _draw_semantic(img, gt.get("mask"), pts, meta, st, files_dir)
        right = _draw_semantic(img, pred.get("mask"), pts, meta, st, files_dir)
        right = _draw_instances(right, {"pred": {k: v for k, v in pred.items() if k != "mask"}}, meta, st)
        return _side_by_side(left, right, tuple(tile.get("labels") or ("ground truth", "prediction")))
    out = _draw_instances(img, tile, meta, st)
    pred = tile.get("pred") or {}
    if pred.get("mask") is not None or (tile.get("gt") or {}).get("points"):
        out = _draw_semantic(out, pred.get("mask"), (tile.get("gt") or {}).get("points", []), meta, st, files_dir)
    return out


def _load_mask(src, files_dir) -> np.ndarray | None:
    import numpy as np
    if src is None:
        return None
    if isinstance(src, str):
        if files_dir is None:
            return None
        from pathlib import Path
        return np.asarray(PILImage.open(Path(files_dir) / src))
    return np.asarray(src)


def _draw_semantic(img: PILImage.Image, mask_src, points, meta: dict, st: dict, files_dir) -> PILImage.Image:
    """A label map as translucent class fills + class-colored boundaries, then sparse points."""
    import numpy as np
    base = img.convert("RGBA")
    w, h = base.size
    scale = min(w, h) / 720
    lw = max(1, round(st["line_width"] * scale))
    hidden = set(st["hidden"])
    skip = {meta.get("background"), meta.get("ignore_index"), 255}
    m = _load_mask(mask_src, files_dir)
    if m is not None:
        m = np.asarray(PILImage.fromarray(m.astype(np.uint8)).resize((w, h), PILImage.Resampling.NEAREST))
        fill = np.zeros((256, 4), np.uint8)
        stroke = np.zeros((256, 4), np.uint8)
        for c in np.unique(m):
            c = int(c)
            if c in skip or f"class:{c}" in hidden:
                continue
            fill[c] = _rgba(feature_color(st, f"class:{c}", c), st["fill_opacity"])
            stroke[c] = _rgba(feature_color(st, f"class:{c}", c), st["opacity"])
        edge = np.zeros(m.shape, bool)
        edge[:-1, :] |= m[:-1, :] != m[1:, :]
        edge[1:, :] |= m[1:, :] != m[:-1, :]
        edge[:, :-1] |= m[:, :-1] != m[:, 1:]
        edge[:, 1:] |= m[:, 1:] != m[:, :-1]
        for _ in range(lw - 1):                                    # thicken boundaries to the line width
            grown = edge.copy()
            grown[:-1, :] |= edge[1:, :]
            grown[1:, :] |= edge[:-1, :]
            grown[:, :-1] |= edge[:, 1:]
            grown[:, 1:] |= edge[:, :-1]
            edge = grown
        layer = fill[m]
        layer[edge] = stroke[m][edge]
        base = PILImage.alpha_composite(base, PILImage.fromarray(layer, "RGBA"))
    if points:
        rad = max(3, round(st["point_size"] * scale * 1.4))           # sparse labels are the ground truth: keep them legible
        layer = PILImage.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for x, y, c in points:
            if f"class:{int(c)}" in hidden:
                continue
            cx, cy = x * w, y * h
            d.ellipse([cx - rad - 1.5, cy - rad - 1.5, cx + rad + 1.5, cy + rad + 1.5], fill=(18, 18, 18, int(st["opacity"] * 230)))
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=_rgba(feature_color(st, f"class:{int(c)}", int(c)), st["opacity"]))
        base = PILImage.alpha_composite(base, layer)
    return base.convert("RGB")


def _side_by_side(left: PILImage.Image, right: PILImage.Image, labels: tuple[str, str]) -> PILImage.Image:
    gap = max(4, left.width // 120)
    out = PILImage.new("RGB", (left.width + gap + right.width, max(left.height, right.height)), BG)
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    d = ImageDraw.Draw(out)
    font = _font(max(11, left.height // 22))
    for x0, text in ((0, labels[0]), (left.width + gap, labels[1])):
        tw = d.textlength(text, font=font)
        d.rectangle([x0, 0, x0 + tw + 12, font.size + 10 if hasattr(font, "size") else 22], fill=(18, 18, 18))
        d.text((x0 + 6, 4), text, fill=INK, font=font)
    return out


def _draw_instances(img: PILImage.Image, tile: dict, meta: dict, st: dict) -> PILImage.Image:
    """Boxes / polygons / keypoints (normalized coords)."""
    base = img.convert("RGBA")
    w, h = base.size
    scale = min(w, h) / 720
    lw = max(1, round(st["line_width"] * scale))
    rad = max(1, round(st["point_size"] * scale))
    a, fa = st["opacity"], st["fill_opacity"]
    layer = PILImage.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    names = {int(k): v for k, v in (meta.get("names") or {}).items()}
    hidden = set(st["hidden"])
    skeleton = [] if "skeleton" in hidden else meta.get("skeleton") or []
    font = _font(max(10, round(13 * scale)))

    def shown_class(c) -> bool:
        return f"class:{int(c)}" not in hidden

    def px(pts):
        return [(x * w, y * h) for x, y in pts]

    def keypoints(sets, hollow: bool, color_of):
        for inst in sets:
            pts = [(p[0] * w, p[1] * h, p[2] if len(p) > 2 else 1.0) for p in inst]
            if not hollow:
                for i, j in skeleton:
                    if f"kpt:{i}" in hidden or f"kpt:{j}" in hidden:
                        continue
                    if i < len(pts) and j < len(pts) and pts[i][2] > 0 and pts[j][2] > 0:
                        d.line([pts[i][:2], pts[j][:2]], fill=_rgba(feature_color(st, "skeleton"), a * 0.85), width=lw)
            for j, (x, y, v) in enumerate(pts):
                if v <= 0 or f"kpt:{j}" in hidden:
                    continue
                box = [x - rad, y - rad, x + rad, y + rad]
                if hollow:
                    if st["halo"]:
                        d.ellipse([x - rad - 1, y - rad - 1, x + rad + 1, y + rad + 1], outline=(18, 18, 18, int(a * 255)), width=lw + 2)
                    d.ellipse(box, outline=_rgba(color_of(j), a), width=max(1, lw))
                else:
                    d.ellipse(box, fill=_rgba(color_of(j), a), outline=(18, 18, 18, int(a * 200)), width=1)

    gt = tile.get("gt") or {}
    if st["show_gt"] and "gt" not in hidden:
        gcol = feature_color(st, "gt")
        shapes = [px(p[1]) for p in gt.get("polygons", []) if shown_class(p[0])] or [
            px([(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]) for b in gt.get("boxes", []) if shown_class(b[4])]
        for poly in shapes:
            if len(poly) >= 2:
                if st["halo"]:
                    d.line(poly + [poly[0]], fill=(18, 18, 18, int(a * 255)), width=lw + 2, joint="curve")
                d.line(poly + [poly[0]], fill=_rgba(gcol, a), width=lw, joint="curve")
        keypoints(gt.get("keypoints", []), True, lambda j: gcol)

    pred = tile.get("pred") or {}
    if st["show_pred"]:
        for p in pred.get("polygons", []):
            if not shown_class(p[0]):
                continue
            c = feature_color(st, f"class:{int(p[0])}", int(p[0]))
            poly = px(p[2])
            if len(poly) >= 3:
                d.polygon(poly, fill=_rgba(c, fa))
                d.line(poly + [poly[0]], fill=_rgba(c, a), width=lw, joint="curve")
        for b in pred.get("boxes", []):
            if not shown_class(b[4]):
                continue
            c = feature_color(st, f"class:{int(b[4])}", int(b[4]))
            x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
            d.rectangle([x1, y1, x2, y2], outline=_rgba(c, a), width=lw)
            if st["show_labels"]:
                text = f"{names.get(int(b[4]), int(b[4]))} {b[5]:.2f}" if len(b) > 5 else str(names.get(int(b[4]), int(b[4])))
                tw = d.textlength(text, font=font)
                th = font.size + 4 if hasattr(font, "size") else 14
                ty = y1 - th if y1 - th >= 0 else y1
                d.rectangle([x1, ty, x1 + tw + 6, ty + th], fill=_rgba(c, min(1.0, a)))
                d.text((x1 + 3, ty + 1), text, fill=(255, 255, 255, 255), font=font)
        keypoints(pred.get("keypoints", []), False, lambda j: feature_color(st, f"kpt:{j}", j))
    return PILImage.alpha_composite(base, layer).convert("RGB")


def render_qc_sheet(step_doc: dict, files_dir, style: dict | None = None, tile_px: int | None = None,
                    with_header: bool = True) -> tuple[PILImage.Image, list[list[int]]]:
    """Re-render a whole QC sheet from its saved step JSON."""
    from pathlib import Path
    meta = step_doc.get("meta", {})
    tiles = [draw_overlays(PILImage.open(Path(files_dir) / t["base"]), t, meta, style, files_dir) for t in step_doc["tiles"]]
    return build_sheet(tiles, [t.get("caption") for t in step_doc["tiles"]], step_doc.get("cols"),
                       tile_px or step_doc.get("tile_px", 720),
                       step_doc.get("title") if with_header else None, step_doc.get("subtitle") if with_header else None)


class QCSheet(ContactSheet):
    """A contact sheet that also saves its overlay data, so it can be re-drawn in other styles.

    images: base images (no overlays); tiles: [{"caption", "gt": {...}, "pred": {...}}, ...] with
    normalized coordinates - boxes [x1, y1, x2, y2, cls(, conf)], polygons [cls(, conf), [[x, y], ...]],
    keypoints [[[x, y, v], ...], ...]; meta: {"names", "kpt_shape", "kpt_names", "skeleton"}.
    """

    BASE_MAX = 1600

    def __init__(self, images: Sequence[Any], tiles: Sequence[dict], meta: dict, cols: int | None = None,
                 tile_px: int = 720, title: str | None = None, subtitle: str | None = None,
                 style: dict | None = None):
        bases = []
        for im in images:
            im = _as_pil(im)
            k = min(1.0, self.BASE_MAX / max(im.size))
            bases.append(im.resize((round(im.width * k), round(im.height * k)), PILImage.Resampling.LANCZOS) if k < 1 else im)
        drawn = [draw_overlays(b, t, meta, style) for b, t in zip(bases, tiles)]          # arrays drawn in memory
        super().__init__(drawn, captions=[t.get("caption") for t in tiles], cols=cols, tile_px=tile_px,
                         title=title, subtitle=subtitle)
        self._bases, self._tiles, self._meta = bases, list(tiles), dict(meta)
        self._layout = {"cols": cols, "tile_px": tile_px, "title": title, "subtitle": subtitle}

    def _bind(self, files_dir, key: str, step: int) -> dict:
        import json
        from pathlib import Path

        from runraccoon.utils import atomic_write_text, safe_relpath, sha256_bytes
        ref = super()._bind(files_dir, key, step)
        qc_dir = Path(files_dir) / "media" / "qc" / safe_relpath(key)
        qc_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np
        skip = {self._meta.get("background"), self._meta.get("ignore_index")}

        def store(data: bytes, prefix: str, ext: str) -> str:
            path = qc_dir / f"{prefix}_{sha256_bytes(data)[:16]}.{ext}"
            if not path.exists():                       # identical content (same QC image, same mask) -> stored once
                path.write_bytes(data)
            return path.relative_to(files_dir).as_posix()

        tiles = []
        for base, t in zip(self._bases, self._tiles):
            buf = io.BytesIO()
            base.convert("RGB").save(buf, format="JPEG", quality=92)
            t = {**t, "base": store(buf.getvalue(), "base", "jpg")}
            for side in ("gt", "pred"):
                d = dict(t.get(side) or {})
                classes = {int(p[2]) for p in d.get("points", [])}
                if d.get("mask") is not None and not isinstance(d["mask"], str):
                    m = np.asarray(d["mask"]).astype(np.uint8)
                    k = min(1.0, 1024 / max(m.shape))                # label maps: nearest-neighbor, <= 1024 px
                    im = PILImage.fromarray(m)
                    if k < 1:
                        im = im.resize((round(m.shape[1] * k), round(m.shape[0] * k)), PILImage.Resampling.NEAREST)
                    mb = io.BytesIO()
                    im.save(mb, format="PNG", optimize=True)
                    d["mask"] = store(mb.getvalue(), f"{side}mask", "png")
                    classes |= {int(c) for c in np.unique(m)}
                if classes:
                    d["classes"] = sorted(c for c in classes if c not in skip)
                if d:
                    t[side] = d
            tiles.append(t)
        doc = {"version": 1, "step": step, "meta": self._meta, "tiles": tiles, **self._layout}
        doc_path = qc_dir / f"step_{step}.json"
        atomic_write_text(doc_path, json.dumps(doc))
        ref["_runraccoon_sheet"]["qc"] = doc_path.relative_to(files_dir).as_posix()
        return ref
