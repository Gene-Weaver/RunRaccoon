import json
import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

import runraccoon as wandb
from runraccoon.panels import estimate_end
from runraccoon.plotting.gifs import GifOptions, find_panels, render_gifs
from runraccoon.qc import ContactSheet, QCSheet, build_sheet, draw_overlays, qc_features
from runraccoon.reader import RunData


def _tile(color, size=(120, 80)):
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[...] = color
    a[10:30, 10:60] = 255 - np.array(color)            # some structure so it is not flat
    return Image.fromarray(a)


def test_find_panels_on_a_headered_grid():
    sheet, boxes = build_sheet([_tile((200, 120, 40)) for _ in range(6)], cols=3, tile_px=120, title="header text")
    found = find_panels(sheet)
    assert len(found) == 6
    for (l, t, r, b), (L, T, R, B) in zip(found, boxes):
        assert abs(l - L) <= 1 and abs(t - T) <= 1 and abs(r - R) <= 1 and abs(b - B) <= 1


def test_contact_sheet_records_tile_boxes():
    run = wandb.init()
    wandb.log({"qc/sheet": ContactSheet([_tile((30, 90, 200)) for _ in range(4)], captions=list("abcd"), cols=2)})
    refs = RunData(run.paths).media_refs()["qc/sheet"]
    assert refs[0]["boxes"] and len(refs[0]["boxes"]) == 4
    wandb.finish()


META = {"names": {0: "leaf", 1: "petiole"}, "kpt_shape": [2, 3], "kpt_names": ["tip", "base"], "skeleton": [[0, 1]]}
TILE = {"caption": "img", "gt": {"boxes": [[0.1, 0.1, 0.5, 0.5, 0]], "polygons": [], "keypoints": [[[0.2, 0.2, 2], [0.4, 0.4, 2]]]},
        "pred": {"boxes": [[0.12, 0.1, 0.52, 0.5, 1, 0.9]], "polygons": [[0, 0.8, [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9]]]],
                 "keypoints": [[[0.21, 0.2, 0.9], [0.41, 0.4, 0.8]]]}}


def test_overlay_style_changes_the_drawing():
    base = _tile((90, 90, 90), (300, 200))
    a = np.asarray(draw_overlays(base, TILE, META, {"line_width": 1}), np.int16)
    b = np.asarray(draw_overlays(base, TILE, META, {"line_width": 6}), np.int16)
    c = np.asarray(draw_overlays(base, TILE, META, {"colors": {"class:1": "#00ff00"}}), np.int16)
    off = np.asarray(draw_overlays(base, TILE, META, {"show_gt": False, "show_pred": False}), np.int16)
    assert np.abs(a - b).sum() > 0 and np.abs(a - c).sum() > 0
    assert np.abs(off - np.asarray(base, np.int16)).sum() == 0
    ids = [f["id"] for f in qc_features(META, [TILE])]
    assert ids == ["gt", "class:0", "class:1", "kpt:0", "kpt:1", "skeleton"]


def test_qc_sheet_saves_restylable_data_and_gifs_use_it():
    run = wandb.init(config={"epochs": 3})
    for step in range(3):
        wandb.log({"rr/qc": QCSheet([_tile((60 + 40 * step, 80, 120), (160, 100)) for _ in range(2)], [TILE, TILE], META,
                                    cols=2, tile_px=160, title=f"epoch {step}")}, step=step)
    files = run.paths.files
    wandb.finish()
    data = RunData(run.paths)
    refs = data.media_refs()["rr/qc"]
    doc = json.loads((files / refs[0]["qc"]).read_text())
    assert len(doc["tiles"]) == 2 and (files / doc["tiles"][0]["base"]).exists()
    assert len(list((files / "media" / "qc" / "rr" / "qc").glob("base_*.jpg"))) == 3      # deduplicated per image
    opts = GifOptions(resolution=180, style={"line_width": 5})
    written = render_gifs(data, opts, workers=1)
    assert sorted(p.name for p in written) == ["panel_01.gif", "panel_02.gif", "qc.gif"]
    gif = Image.open(written[0] if written[0].name == "qc.gif" else [p for p in written if p.name == "qc.gif"][0])
    assert gif.size == (320, 180) and gif.n_frames == 3
    portrait = render_gifs(data, GifOptions(resolution=180, orientation="portrait", panels=False), workers=1)
    assert Image.open(portrait[0]).size == (180, 320)


def test_estimate_end():
    rows = [{"_step": 1, "_timestamp": 100.0, "_runtime": 0, "train/loss": 1},
            {"_step": 2, "_timestamp": 160.0, "_runtime": 60, "val/loss": 1},
            {"_step": 3, "_timestamp": 210.0, "_runtime": 110, "val/loss": 1},
            {"_step": 4, "_timestamp": 260.0, "_runtime": 160, "val/loss": 1}]
    assert estimate_end(rows[:1], 10) is None                       # no validation yet
    assert estimate_end(rows[:2], 10) == 160 + 9 * 60               # first epoch incl. startup
    assert estimate_end(rows, 10) == 260 + 7 * 50                   # median of recent epochs
    assert estimate_end(rows, None) is None


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_dashboard_gif_endpoints():
    from runraccoon.dashboard.server import Dashboard
    run = wandb.init(config={"epochs": 3})
    for step in range(3):
        wandb.log({"rr/qc": QCSheet([_tile((60 + 40 * step, 80, 120), (160, 100))], [TILE], META, tile_px=160)}, step=step)
    rid = run.id
    wandb.finish()
    port = _port()
    app = Dashboard(port=port)
    threading.Thread(target=app.serve, daemon=True).start()
    for _ in range(50):
        if app.httpd is not None:
            break
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}/api/runs/{rid}/gifs"
    try:
        cfg = json.loads(urllib.request.urlopen(f"{base}/config").read())
        assert cfg["restylable"] == ["rr/qc"] and {f["id"] for f in cfg["features"]} >= {"gt", "class:0", "kpt:1"}
        prev = urllib.request.urlopen(f"{base}/preview?options=" + urllib.parse.quote(json.dumps({"resolution": 360})))
        assert prev.headers["Content-Type"] == "image/png"
        body = json.dumps({"options": {"resolution": 360, "seconds_per_frame": 0.5}, "style": {"line_width": 4}}).encode()
        try:                                                          # no custom header -> refused
            urllib.request.urlopen(urllib.request.Request(f"{base}/render", data=body, method="POST"))
            raise AssertionError("render without X-RunRaccoon header must be refused")
        except urllib.error.HTTPError as e:
            assert e.code == 403
        req = urllib.request.Request(f"{base}/render", data=body, method="POST",
                                     headers={"X-RunRaccoon": "1", "Content-Type": "application/json"})
        assert urllib.request.urlopen(req).status == 202
        for _ in range(200):
            job = json.loads(urllib.request.urlopen(f"{base}/config").read())["job"]
            if job["state"] != "running":
                break
            time.sleep(0.1)
        assert job["state"] == "done" and job["written"] >= 1
        assert json.loads(urllib.request.urlopen(f"{base}/config").read())["options"]["seconds_per_frame"] == 0.5
    finally:
        app.shutdown()


def test_hidden_features_are_not_drawn():
    base = _tile((90, 90, 90), (300, 200))
    every = np.asarray(draw_overlays(base, TILE, META, None), np.int16)
    no_cls1 = np.asarray(draw_overlays(base, TILE, META, {"hidden": ["class:1"]}), np.int16)
    all_off = np.asarray(draw_overlays(base, TILE, META, {"hidden": ["gt", "class:0", "class:1", "kpt:0", "kpt:1", "skeleton"]}), np.int16)
    assert np.abs(every - no_cls1).sum() > 0                                    # class 1's box is gone
    assert np.abs(all_off - np.asarray(base, np.int16)).sum() == 0              # everything hidden = bare image
