"""Automatic per-epoch QC contact sheet for Ultralytics training (detect / segment / pose / obb).

No training-code changes: when Ultralytics loads its wandb callbacks (which, with RunRaccoon
imported, are RunRaccoon's), the epoch-end callback is wrapped so that after every validation:

  1. a fixed sample of val images is chosen once (evenly spaced, deterministic);
  2. the newest weights (last.pt; best.pt for the final re-validation) predict on them;
  3. each tile shows ground truth as white outlines with a dark halo and predictions on top
     (boxes, masks, keypoints + skeleton, OBB); the overlay data is saved too, so the GIFs can be
     re-drawn in another style from the dashboard;
  4. the sheet is logged as `runraccoon/qc_contact_sheet` at the epoch's step, so it shows up
     in the dashboard's Media tab and becomes a GIF (plus one GIF per tile) when the run ends.

Settings: qc_sheet (on/off), qc_images, qc_every, qc_cols, qc_tile_px, qc_conf, qc_key.
Everything is wrapped in try/except: a QC problem is logged, never raised into training.
"""
from __future__ import annotations

import logging
import sys
from importlib.abc import Loader, MetaPathFinder

log = logging.getLogger("runraccoon")
WB_MODULE = "ultralytics.utils.callbacks.wb"


# ------------------------------------------------------------------------- QC sheet state
class _QCState:
    def __init__(self):
        self.indices: list[int] | None = None
        self.failed = False


_state = _QCState()


def _settings():
    mod = sys.modules.get("runraccoon")
    run = getattr(mod, "run", None) if mod else None
    return run, (run.settings if run is not None else None)


def _gt_data(label: dict, task: str) -> dict:
    """Ground truth of one val image, normalized: boxes / polygons / keypoints."""
    import numpy as np
    cls = np.asarray(label.get("cls", [])).reshape(-1).astype(int)
    out: dict = {"boxes": [], "polygons": [], "keypoints": []}
    segs = label.get("segments") or []
    if task == "segment" and len(segs):
        out["polygons"] = [[int(c), np.asarray(s, float).round(5).tolist()] for c, s in zip(cls, segs)]
    else:
        for c, (x, y, bw, bh) in zip(cls, np.asarray(label.get("bboxes", np.zeros((0, 4))), float)):
            out["boxes"].append([round(x - bw / 2, 5), round(y - bh / 2, 5), round(x + bw / 2, 5), round(y + bh / 2, 5), int(c)])
    kp = label.get("keypoints")
    if kp is not None and len(kp):
        out["keypoints"] = np.asarray(kp, float).round(5).tolist()
    return out


def _pred_data(res, task: str) -> dict:
    """Predictions of one image, normalized."""
    out: dict = {"boxes": [], "polygons": [], "keypoints": []}
    if task == "obb" and getattr(res, "obb", None) is not None:
        for poly, c, conf in zip(res.obb.xyxyxyxyn.tolist(), res.obb.cls.tolist(), res.obb.conf.tolist()):
            out["polygons"].append([int(c), round(conf, 4), [[round(x, 5), round(y, 5)] for x, y in poly]])
        return out
    if res.boxes is not None:
        cls, conf = res.boxes.cls.tolist(), res.boxes.conf.tolist()
        if task == "segment" and res.masks is not None:
            for c, cf, poly in zip(cls, conf, res.masks.xyn):
                out["polygons"].append([int(c), round(cf, 4), poly.round(5).tolist()])
        else:
            for (x1, y1, x2, y2), c, cf in zip(res.boxes.xyxyn.tolist(), cls, conf):
                out["boxes"].append([round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5), int(c), round(cf, 4)])
    if getattr(res, "keypoints", None) is not None and res.keypoints.xyn is not None and len(res.keypoints.xyn):
        xy = res.keypoints.xyn.tolist()
        kc = res.keypoints.conf.tolist() if res.keypoints.conf is not None else [[1.0] * len(i) for i in xy]
        out["keypoints"] = [[[round(x, 5), round(y, 5), round(c, 3)] for (x, y), c in zip(inst, cs)] for inst, cs in zip(xy, kc)]
    return out


def qc_epoch_end(trainer) -> None:
    """Predict on the fixed QC sample and log a (restylable) contact sheet. Never raises."""
    run, s = _settings()
    if run is None or s is None or not s.qc_sheet or _state.failed:
        return
    try:
        epoch = trainer.epoch + 1
        final = epoch > trainer.epochs                       # final_eval re-validates best.pt at epochs+1
        if not final and epoch % max(1, int(s.qc_every)) and epoch != trainer.epochs:
            return
        weights = trainer.best if final and trainer.best.exists() else trainer.last
        if not weights.exists():
            return
        ds = trainer.validator.dataloader.dataset if trainer.validator and trainer.validator.dataloader else None
        if ds is None:
            return
        if _state.indices is None:
            n = len(ds.im_files)
            k = min(int(s.qc_images), n)
            _state.indices = sorted({int(i * n / k) for i in range(k)})
        import cv2
        import torch
        from ultralytics import YOLO

        from runraccoon.qc import QCSheet
        model = YOLO(str(weights))
        task = trainer.args.task
        files = [ds.im_files[i] for i in _state.indices]
        results = model.predict(files, imgsz=trainer.args.imgsz, conf=float(s.qc_conf), device=trainer.device,
                                verbose=False, retina_masks=task == "segment")
        images, tiles = [], []
        for i, res in zip(_state.indices, results):
            images.append(cv2.cvtColor(res.orig_img, cv2.COLOR_BGR2RGB))
            gt, pred = _gt_data(ds.labels[i], task), _pred_data(res, task)
            n_gt = len(ds.labels[i].get("cls", []))
            n_pred = len(pred["boxes"]) + len(pred["polygons"])
            name = ds.im_files[i].replace("\\", "/").rsplit("/", 1)[-1]
            tiles.append({"caption": f"{name}  ·  {n_pred} pred / {n_gt} gt", "gt": gt, "pred": pred})
        data = getattr(trainer, "data", {}) or {}
        meta = {"names": {int(k): v for k, v in (getattr(model, "names", None) or data.get("names") or {}).items()},
                "task": task, "kpt_shape": list(data.get("kpt_shape") or []) or None,
                "kpt_names": data.get("kpt_names"), "skeleton": data.get("skeleton")}
        m = trainer.metrics or {}
        fit = [f"{k.split('/')[-1]} {v:.3f}" for k, v in m.items() if k.startswith("metrics/mAP50-95")]
        stage = f"final (best.pt, epoch {epoch - 1} model)" if final else f"epoch {epoch}/{trainer.epochs}"
        sheet = QCSheet(images, tiles, meta, cols=int(s.qc_cols) or None, tile_px=int(s.qc_tile_px),
                        title=f"{run.name}  ·  {stage}",
                        subtitle="white = ground truth   ·   colors = prediction" + ("   ·   " + "   ".join(fit) if fit else ""))
        run.log({s.qc_key: sheet}, step=epoch)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:                                     # QC must never break training
        _state.failed = True
        log.warning("QC contact sheet disabled for this run: %s: %s", type(e).__name__, e)


# ------------------------------------------------------------------- hook into Ultralytics
def _patch(module) -> None:
    cbs = getattr(module, "callbacks", None)
    if not isinstance(cbs, dict) or "on_fit_epoch_end" not in cbs or getattr(cbs["on_fit_epoch_end"], "_runraccoon", False):
        return
    original = cbs["on_fit_epoch_end"]

    def on_fit_epoch_end(trainer):
        qc_epoch_end(trainer)          # first, so the sheet lands in the same history row as the metrics
        original(trainer)

    on_fit_epoch_end._runraccoon = True
    cbs["on_fit_epoch_end"] = on_fit_epoch_end


class _PatchingLoader(Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        try:
            _patch(module)
        except Exception as e:  # pragma: no cover
            log.info("ultralytics QC hook not installed: %s", e)


class _Finder(MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != WB_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = getattr(finder, "find_spec", lambda *a: None)(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchingLoader(spec.loader)
                return spec
        return None


def install() -> None:
    """Patch Ultralytics' wandb callbacks now if loaded, otherwise as soon as they are imported."""
    if WB_MODULE in sys.modules:
        _patch(sys.modules[WB_MODULE])
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
