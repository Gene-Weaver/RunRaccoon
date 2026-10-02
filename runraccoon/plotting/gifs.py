"""Training-progress GIFs from logged QC images, made when a run finishes.

For every image key logged at several steps (e.g. a per-epoch `qc/contact_sheet`):

    files/plots/gifs/<key>.gif                 the full image, one frame per step
    files/plots/gifs/<key>/panel_01.gif ...    one GIF per panel of that image

Panels:
  * a list of images logged under one key (`[wandb.Image(...), ...]`) -> each list position is a
    panel, and the full GIF is a grid of them composed here;
  * a single composite image (a contact sheet) -> panels are found by detecting the background-
    colored gutters between tiles (header strips are skipped, each tile is trimmed).

Every frame is an exact 1920x1080 (landscape) or 1080x1920 (portrait) canvas: the image is fit
with Lanczos resampling, labeled "<key> - epoch N", with a thin progress bar. Quality: one shared
256-color palette per GIF (no frame-to-frame color flicker) and Floyd-Steinberg dithering; Pillow
stores only the changed region of each frame.
"""
from __future__ import annotations

import fnmatch
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from runraccoon.utils import safe_relpath


@dataclass
class GifOptions:
    seconds_per_frame: float = 0.25     # how long each step is shown
    orientation: str = "landscape"      # "landscape" (1920x1080) or "portrait" (1080x1920)
    resolution: int = 1080              # short side of the canvas, px
    hold_last_s: float = 1.5            # extra time on the final frame before looping
    panels: bool = True                 # also one GIF per panel
    label: bool = True                  # "<key> - epoch N" banner + progress bar
    keys: tuple | None = None           # glob patterns of media keys; None = every multi-step image key
    min_frames: int = 3                 # skip keys logged at fewer steps than this
    max_frames: int = 400               # evenly subsample longer runs (the last step is always kept)
    workers: int | None = None          # parallel workers; None = max(physical cores, threads) - 2
    style: dict | None = None           # overlay style for RunRaccoon QC sheets (runraccoon.qc.DEFAULT_STYLE keys)

    @classmethod
    def from_dict(cls, d: dict | None) -> "GifOptions":
        o = cls()
        for k, v in (d or {}).items():
            if k in cls.__dataclass_fields__ and v is not None:
                setattr(o, k, tuple(v) if k == "keys" and not isinstance(v, str) else v)
        if isinstance(o.keys, str):
            o.keys = tuple(p.strip() for p in o.keys.split(",") if p.strip())
        return o

    @property
    def canvas(self) -> tuple[int, int]:
        short, long_ = int(self.resolution), int(round(self.resolution * 16 / 9))
        return (long_, short) if self.orientation != "portrait" else (short, long_)


def saved_options(data) -> GifOptions:
    """A run's GIF options: what the run was configured with, overridden by the settings last
    used to regenerate from the dashboard (plots/gifs/options.json). The end-of-run build,
    `runraccoon gif` and the dashboard all start from this, so they never disagree."""
    import json
    merged = dict(data.wandb_internal.get("gif") or {})
    try:
        merged.update(json.loads((data.paths.plots / "gifs" / "options.json").read_text()))
    except (OSError, ValueError):
        pass
    return GifOptions.from_dict(merged)


# ----------------------------------------------------------------------------- parallelism
def _physical_cores() -> int | None:
    """Physical core count from /proc/cpuinfo (Linux); None if unknown."""
    try:
        cores, phys, core = set(), None, None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
            elif not line.strip() and core is not None:
                cores.add((phys, core))
                phys = core = None
        if core is not None:
            cores.add((phys, core))
        return len(cores) or None
    except OSError:
        return None


def _available_memory() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def default_workers() -> int:
    """max(physical cores, hardware threads) - 2, at least 1 (respects CPU affinity / taskset)."""
    try:
        threads = len(os.sched_getaffinity(0))
    except AttributeError:                                   # macOS / Windows
        threads = os.cpu_count() or 1
    return max(1, max(_physical_cores() or 0, threads) - 2)


# ----------------------------------------------------------------------------- panel finding
def _bands(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Runs of True in a 1-D mask, at least min_len long."""
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def find_panels(img: Image.Image, tol: int = 24) -> list[tuple[int, int, int, int]]:
    """Boxes (left, top, right, bottom) of the tiles in a contact sheet, in reading order.

    Background = the color of the image's corners. A row/column is a gutter when (almost) none of
    its pixels differ from it. Thin row bands (a header line) are dropped before the columns are
    measured, so header text that crosses gutters doesn't hide them.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    h, w, _ = a.shape
    corners = np.array([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]])
    bg = np.median(corners, axis=0)
    content = np.abs(a - bg).max(axis=2) > tol

    row_bands = _bands(content.mean(axis=1) > 0.15, max(4, h // 60))
    if not row_bands:
        return []
    tallest = max(b - t for t, b in row_bands)
    row_bands = [(t, b) for t, b in row_bands if b - t >= 0.35 * tallest]      # drop header strips
    rows_mask = np.zeros(h, bool)
    for t, b in row_bands:
        rows_mask[t:b] = True
    col_bands = _bands(content[rows_mask].mean(axis=0) > 0.15, max(4, w // 60))

    boxes = []
    for t, b in row_bands:
        for l, r in col_bands:
            cell = content[t:b, l:r]
            if cell.mean() < 0.05:
                continue                                                         # empty grid slot
            ys, xs = np.nonzero(cell)
            boxes.append((l + int(xs.min()), t + int(ys.min()), l + int(xs.max()) + 1, t + int(ys.max()) + 1))
    return boxes


# ------------------------------------------------------------------------------- rendering
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        from matplotlib import font_manager
        for family in ("Roboto", "Inter", "Liberation Sans", "DejaVu Sans"):
            path = font_manager.findfont(font_manager.FontProperties(family=family, weight="semibold"),
                                         fallback_to_default=False)
            if path:
                return ImageFont.truetype(path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def compose_frame(img: Image.Image, title: str, step_text: str, progress: float, opts: GifOptions,
                  bg: tuple[int, int, int]) -> Image.Image:
    W, H = opts.canvas
    canvas = Image.new("RGB", (W, H), bg)
    pad = int(H * 0.025) if W > H else int(W * 0.025)
    top = pad
    if opts.label:
        fs = max(14, int(min(W, H) * 0.026))
        font, small = _font(fs), _font(int(fs * 0.85))
        light = sum(bg) > 382
        ink, muted = ((29, 29, 27), (110, 109, 104)) if light else ((240, 240, 236), (160, 159, 152))
        d = ImageDraw.Draw(canvas)
        d.text((pad, pad), title, fill=ink, font=font)
        tw = d.textlength(step_text, font=small)
        d.text((W - pad - tw, pad + (fs - int(fs * 0.85)) // 2), step_text, fill=muted, font=small)
        top = pad + int(fs * 1.6)
        bar_h = max(3, H // 270)                                                 # thin progress bar
        d.rectangle([0, H - bar_h, W, H], fill=(70, 70, 66) if not light else (225, 224, 217))
        d.rectangle([0, H - bar_h, int(W * progress), H], fill=(57, 135, 229) if not light else (42, 120, 214))
    box_w, box_h = W - 2 * pad, H - top - pad - (H // 135 if opts.label else 0)
    k = min(box_w / img.width, box_h / img.height)
    size = (max(1, int(img.width * k)), max(1, int(img.height * k)))
    fitted = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((W - size[0]) // 2, top + (box_h - size[1]) // 2))
    return canvas


def compose_grid(images: Sequence[Image.Image], target_aspect: float, bg: tuple[int, int, int], gap: int = 8) -> Image.Image:
    """Lay out a list of panels (one logged step of an image list) as a grid near target_aspect."""
    n = len(images)
    tw = max(im.width for im in images)
    th = max(im.height for im in images)
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        aspect = (cols * tw + (cols + 1) * gap) / (rows * th + (rows + 1) * gap)
        score = abs(math.log(aspect / target_aspect))
        if best is None or score < best[0]:
            best = (score, cols, rows)
    _, cols, rows = best
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im.convert("RGB"), (gap + c * (tw + gap) + (tw - im.width) // 2, gap + r * (th + gap) + (th - im.height) // 2))
    return sheet


_BAYER8 = (np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
                     [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
                     [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                     [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], np.float32) + 0.5) / 64 - 0.5
TRANSPARENT = 255          # palette index reserved for "same as the previous frame"


def _palette_lut(pal: np.ndarray, bits: int = 6, threads: int = 1) -> np.ndarray:
    """(2^bits)^3 lookup table: quantized RGB -> nearest palette index (chunks run on threads;
    numpy releases the GIL for the distance math)."""
    n = 1 << bits
    centers = (np.arange(n, dtype=np.float32) + 0.5) * (256 / n)
    grid = np.stack(np.meshgrid(centers, centers, centers, indexing="ij"), -1).reshape(-1, 3)
    pal = pal.astype(np.float32)
    out = np.empty(len(grid), np.uint8)

    def chunk(i: int) -> None:
        d = ((grid[i:i + 16384, None, :] - pal[None, :, :]) ** 2).sum(-1)
        out[i:i + 16384] = d.argmin(1)

    starts = range(0, len(grid), 16384)
    if threads > 1:
        with ThreadPoolExecutor(threads) as tp:
            list(tp.map(chunk, starts))
    else:
        for i in starts:
            chunk(i)
    return out.reshape(n, n, n)


def save_gif(frames: list[Image.Image], path: Path, opts: GifOptions, denoise: int = 10, threads: int = 1) -> Path:
    """High-quality, compact GIF.

    * one 255-color palette for the whole GIF, built from samples of all frames (no color flicker);
    * ordered (Bayer) dithering - position-stable, so an unchanged pixel maps to the same color
      in every frame (Floyd-Steinberg error diffusion makes every frame differ);
    * temporal denoising: pixels that only wobble by JPEG noise (<= `denoise` levels) keep their
      previous value;
    * pixels identical to the previous frame are written as transparent, so each frame stores
      only what actually changed.

    `threads`: the palette lookup table and the per-frame dither + palette mapping run on a
    thread pool. Only the temporal-denoise chain (each frame depends on the previous) is
    sequential, so the output is identical to a single-threaded run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_idx = sorted(set(np.linspace(0, len(frames) - 1, min(len(frames), 12)).round().astype(int)))
    thumbs = [frames[i].resize((frames[i].width // 4, frames[i].height // 4), Image.Resampling.BILINEAR) for i in sample_idx]
    mosaic = Image.new("RGB", (thumbs[0].width, thumbs[0].height * len(thumbs)))
    for i, t in enumerate(thumbs):
        mosaic.paste(t, (0, i * t.height))
    pal_img = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    pal = np.array(pal_img.getpalette()[:255 * 3], np.uint8).reshape(-1, 3)
    if len(pal) < 255:
        pal = np.vstack([pal, np.zeros((255 - len(pal), 3), np.uint8)])
    lut = _palette_lut(pal, threads=threads)
    flat_palette = pal.reshape(-1).tolist() + [0, 0, 0]          # index 255 = transparent slot

    w, h = frames[0].size
    dither = np.tile(_BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w, None] * 12.0

    def index(src: np.ndarray) -> np.ndarray:                 # dither + palette lookup (thread-safe)
        v = (np.clip(src + dither, 0, 255).astype(np.uint8) >> 2)
        return lut[v[..., 0], v[..., 1], v[..., 2]]

    idxs: list = [None] * len(frames)
    prev_src = None
    window = max(2, 2 * threads)                              # frames in flight (bounds memory)
    with ThreadPoolExecutor(max(1, threads)) as tp:
        pending: dict = {}
        for i, f in enumerate(frames):                        # sequential: the denoise chain
            src = np.asarray(f.convert("RGB"), np.int16)
            if prev_src is not None:
                still = np.abs(src - prev_src).max(axis=2) <= denoise
                src = np.where(still[..., None], prev_src, src)
            pending[i] = tp.submit(index, src)                # parallel: mapping to the palette
            prev_src = src
            if len(pending) >= window:
                j = min(pending)
                idxs[j] = pending.pop(j).result()
        for j in sorted(pending):
            idxs[j] = pending[j].result()

    out_frames = []
    for i, idx in enumerate(idxs):
        draw = idx if i == 0 else np.where(idx == idxs[i - 1], np.uint8(TRANSPARENT), idx)
        frame = Image.fromarray(draw.astype(np.uint8), mode="P")
        frame.putpalette(flat_palette)
        frame.info["transparency"] = TRANSPARENT
        out_frames.append(frame)

    ms = max(20, int(round(opts.seconds_per_frame * 1000)))
    durations = [ms] * len(out_frames)
    durations[-1] = ms + int(opts.hold_last_s * 1000)
    tmp = path.with_name(f".{path.name}.tmp")
    out_frames[0].save(tmp, format="GIF", save_all=True, append_images=out_frames[1:], duration=durations,
                       loop=0, optimize=False, disposal=1, transparency=TRANSPARENT)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------ GIF tasks
@dataclass
class _Task:
    key: str
    title: str
    steps: list                 # [(step, [abs image paths...], tile boxes or None, qc json or None), ...]
    panel: int | None           # None = full image; j = list index or contact-sheet tile
    boxes: list | None          # contact-sheet tile boxes of the last frame (recorded or detected)
    x_word: str
    out: str
    opts: dict
    files: str = ""             # run files/ dir (for QC overlay data)
    threads: int = 1            # threads this GIF's process may use


def _subsample(steps: list, max_frames: int) -> list:
    if len(steps) <= max_frames:
        return steps
    idx = sorted(set(np.linspace(0, len(steps) - 1, max_frames).round().astype(int)))
    return [steps[i] for i in idx]


def _render_task(t: _Task) -> str | None:
    opts = GifOptions.from_dict(t.opts)
    W, H = opts.canvas
    ref = Image.open(t.steps[-1][1][0])
    bg = tuple(int(c) for c in ref.convert("RGB").getpixel((0, 0)))
    is_list = max(len(p) for _, p, _, _ in t.steps) > 1
    restyle = all(q for _, _, _, q in t.steps)          # RunRaccoon QC sheets: redraw overlays in opts.style

    def frame(i: int):
        s, paths, step_boxes, qc_path = t.steps[i]
        img = _source_image(t, opts, i, paths, step_boxes, qc_path, restyle, is_list, ref, bg, W, H)
        return None if img is None else compose_frame(img, t.title, f"{t.x_word} {s}", (i + 1) / len(t.steps), opts, bg)

    if t.threads > 1:                                    # decode / resize / draw release the GIL
        with ThreadPoolExecutor(t.threads) as tp:
            frames = [f for f in tp.map(frame, range(len(t.steps))) if f is not None]
    else:
        frames = [f for f in map(frame, range(len(t.steps))) if f is not None]
    if len(frames) < 2:
        return None
    return str(save_gif(frames, Path(t.out), opts, threads=t.threads))


def _source_image(t: _Task, opts: GifOptions, i: int, paths, step_boxes, qc_path, restyle: bool, is_list: bool,
                  ref: Image.Image, bg, W: int, H: int):
    """The image shown in frame i (before the title / fit), or None to skip the frame."""
    if restyle:
        return _restyled(Path(t.files), qc_path, t.panel, opts, W, H)
    if t.panel is None:
        ims = [Image.open(p) for p in paths]
        return compose_grid(ims, W / H, bg) if is_list else ims[0]
    if is_list:
        return Image.open(paths[t.panel]) if t.panel < len(paths) else None
    img = Image.open(paths[0])
    if step_boxes:                                        # ContactSheet recorded exact tile positions
        if t.panel >= len(step_boxes):
            return None
        box = tuple(step_boxes[t.panel])
    elif img.size == ref.size:
        box = tuple(t.boxes[t.panel])
    else:                                                 # layout changed at this step: re-detect
        b2 = find_panels(img)
        if len(b2) != len(t.boxes):
            return None
        box = b2[t.panel]
    return img.crop(box)


def _restyled(files: Path, qc_path: str, panel: int | None, opts: GifOptions, W: int, H: int):
    """A frame's image re-drawn from saved QC overlay data in opts.style."""
    import json

    from runraccoon.qc import draw_overlays, render_qc_sheet
    doc = json.loads((files / qc_path).read_text())
    if panel is None:
        sheet, _ = render_qc_sheet(doc, files, opts.style)
        return sheet
    if panel >= len(doc["tiles"]):
        return None
    tile = doc["tiles"][panel]
    base = Image.open(files / tile["base"]).convert("RGB")
    k = min(W * 0.95 / base.width, H * 0.85 / base.height)         # draw at (about) final size: crisp lines
    base = base.resize((max(1, round(base.width * k)), max(1, round(base.height * k))), Image.Resampling.LANCZOS)
    return draw_overlays(base, tile, doc.get("meta", {}), opts.style, files)


def render_gifs(data, opts: GifOptions, x_word: str = "step", workers: int | None = None,
                progress=None) -> list[Path]:
    """Make every GIF for a run (`data` is a runraccoon.reader.RunData)."""
    files = data.paths.files
    by_key: dict[str, dict] = {}
    for key, refs in data.media_refs().items():
        if opts.keys and not any(fnmatch.fnmatchcase(key, p) for p in opts.keys):
            continue
        for r in refs:
            by_key.setdefault(key, {}).setdefault(r["step"], []).append(
                (r.get("index", 0), str(files / r["path"]), r.get("boxes"), r.get("qc")))

    out_dir = data.paths.plots / "gifs"
    tasks: list[_Task] = []
    for key, steps in by_key.items():
        if len(steps) < max(2, opts.min_frames):
            continue
        ordered = _subsample([(s, [e[1] for e in sorted(v, key=lambda e: e[0])],
                               v[0][2] if len(v) == 1 else None, v[0][3] if len(v) == 1 else None)
                              for s, v in sorted(steps.items(), key=lambda kv: kv[0])], opts.max_frames)
        rel = safe_relpath(key)
        common = dict(steps=ordered, x_word=x_word, opts=vars(opts), files=str(files))
        tasks.append(_Task(key=key, title=key, panel=None, boxes=None, out=str(out_dir / f"{rel}.gif"), **common))
        if not opts.panels:
            continue
        n_list = max(len(e[1]) for e in ordered)
        if n_list > 1:
            panels, boxes = range(n_list), None
        else:
            boxes = ordered[-1][2] or find_panels(Image.open(ordered[-1][1][0]))     # recorded, else detected
            panels = range(len(boxes)) if len(boxes) >= 2 else range(0)
        for j in panels:
            tasks.append(_Task(key=key, title=f"{key} - panel {j + 1}", panel=j, boxes=boxes,
                               out=str(out_dir / rel / f"panel_{j + 1:02d}.gif"), **common))
    if not tasks:
        return []
    # Two levels: one process per GIF, and the worker budget split into threads inside each.
    budget = max(1, int(workers or opts.workers or default_workers()))
    n = min(len(tasks), budget)
    W, H = opts.canvas                                         # each GIF process holds its frames: cap by RAM
    per_gif = max(len(t.steps) for t in tasks) * W * H * 3 * 3
    avail = _available_memory()
    if avail:
        n = max(1, min(n, int(avail * 0.4 // per_gif)))
    for t in tasks:
        t.threads = max(1, budget // n)
    results = []
    if progress:
        progress(0, len(tasks))
    if n <= 1:
        for t in tasks:
            results.append(_render_task(t))
            if progress:
                progress(len(results), len(tasks))
    else:
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(n) as pool:
            for fut in as_completed([pool.submit(_render_task, t) for t in tasks]):
                results.append(fut.result())
                if progress:
                    progress(len(results), len(tasks))
    return [Path(r) for r in results if r]
