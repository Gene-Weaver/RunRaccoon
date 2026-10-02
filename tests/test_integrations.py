import json

import numpy as np
from PIL import Image

import runraccoon as wandb
from runraccoon.integrations.dino import EmbeddingPCA, pca_scatter
from runraccoon.integrations.segmentation import SegMetrics, semantic_qc_sheet
from runraccoon.plotting.gifs import GifOptions, render_gifs
from runraccoon.qc import draw_overlays, qc_features
from runraccoon.reader import RunData


def test_seg_metrics_against_hand_computed_values():
    m = SegMetrics(3, ["bg", "a", "b"], ignore_index=255)
    m.update(np.array([0, 1, 1, 1, 2, 0]), np.array([0, 0, 1, 1, 255, 2]))     # one ignored pixel
    out = m.compute()
    # confusion (rows=true): bg->bg 1, bg->a 1 | a->a 2 | b->bg 1
    assert out["pixel_acc"] == 3 / 5
    iou_bg, iou_a, iou_b = 1 / (2 + 2 - 1), 2 / (2 + 3 - 2), 0.0
    assert abs(out["mIoU"] - np.mean([iou_bg, iou_a, iou_b])) < 1e-9
    logits = np.zeros((1, 3, 1, 2)); logits[0, 2, 0, 0] = 1; logits[0, 1, 0, 1] = 1  # argmax over C
    m.reset(); m.update(logits, np.array([[[2, 1]]]))
    assert m.compute()["pixel_acc"] == 1.0


def test_semantic_sheet_saves_masks_and_restyles():
    rng = np.random.default_rng(0)
    run = wandb.init(config={"epochs": 3})
    names = ["background", "honey", "bee"]
    gt = np.zeros((64, 64), np.uint8); gt[10:40, 10:40] = 1; gt[40:60, 30:60] = 2
    for step in range(3):
        pred = gt.copy(); pred[:, : 10 + 5 * step] = 0
        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        wandb.log({"qc/seg": semantic_qc_sheet([img, img], [gt, gt], [pred, pred], names, background=0,
                                               points=[[(20, 20, 1), (45, 50, 2)]] * 2, tile_px=160)}, step=step)
    files = run.paths.files
    wandb.finish()
    data = RunData(run.paths)
    ref = data.media_refs()["qc/seg"][-1]
    doc = json.loads((files / ref["qc"]).read_text())
    tile = doc["tiles"][0]
    assert tile["split"] and tile["gt"]["mask"].endswith(".png") and tile["pred"]["mask"].endswith(".png")
    assert tile["gt"]["classes"] == [1, 2]                                       # background excluded
    assert [f["id"] for f in qc_features(doc["meta"], doc["tiles"])] == ["class:1", "class:2"]
    base = Image.open(files / tile["base"])
    shown = np.asarray(draw_overlays(base, tile, doc["meta"], None, files), np.int16)
    no_bee = np.asarray(draw_overlays(base, tile, doc["meta"], {"hidden": ["class:2"]}, files), np.int16)
    assert np.abs(shown - no_bee).sum() > 0
    gifs = render_gifs(data, GifOptions(resolution=180, style={"colors": {"class:1": "#00ff00"}}), workers=1)
    assert {p.name for p in gifs} == {"seg.gif", "panel_01.gif", "panel_02.gif"}


def test_embedding_pca_is_stable_and_scatter_has_classes():
    rng = np.random.default_rng(1)
    tokens = rng.normal(size=(2, 16, 32)).astype(np.float32)
    tokens[:, :8] += 4
    pca = EmbeddingPCA().fit(tokens)
    a, b = pca.rgb(tokens, (4, 4)), pca.rgb(tokens, (4, 4))
    assert a[0].shape == (4, 4, 3) and a[0].dtype == np.uint8 and np.array_equal(a[0], b[0])
    chart = pca_scatter(tokens.reshape(-1, 32), np.r_[np.zeros(16), np.ones(16)], ["comb", "bee"])
    assert chart.kind == "scatter" and {r[0] for r in chart.table.data} == {"comb", "bee"}
