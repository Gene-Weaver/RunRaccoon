import json
import socket
import threading
import urllib.request

import numpy as np
import pytest

import runraccoon as wandb


def _training_run():
    run = wandb.init(project="demo", name="render-me", config={"epochs": 12})
    for ep in range(12):
        wandb.log({"train/loss": 1 / (ep + 1), "val/loss": 1.1 / (ep + 1), "val/acc": ep / 12, "epoch": ep})
    wandb.log({"qc/img": wandb.Image(np.zeros((6, 6, 3)))})
    wandb.log({"cm": wandb.plot.confusion_matrix(y_true=[0, 1, 1, 0], preds=[0, 1, 0, 0], class_names=["a", "b"])})
    wandb.summary["test/acc"] = 0.9
    return run


def test_final_render_writes_all_figures():
    pytest.importorskip("matplotlib")
    from runraccoon.plotting.render import render_run
    run = _training_run()
    p = run.paths
    wandb.finish()
    written = {x.relative_to(p.plots).as_posix() for x in render_run(p.run_dir, final=True, formats=("png", "pdf"))}
    for name in ("progress.png", "summary/run_summary.png", "summary/run_summary.pdf", "summary/loss.png",
                 "summary/performance.png", "charts/cm.png", "history.csv"):
        assert name in written, name


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.read()


def test_dashboard_api():
    from runraccoon.dashboard.server import Dashboard
    run = _training_run()
    rid = run.id
    port = _free_port()
    app = Dashboard(port=port)
    threading.Thread(target=app.serve, daemon=True).start()
    import time
    for _ in range(50):
        if app.httpd is not None:
            break
        time.sleep(0.05)
    try:
        assert json.loads(_get(port, "/api/ping")[1])["app"] == "runraccoon"
        runs = json.loads(_get(port, "/api/runs")[1])["runs"]
        assert runs[0]["id"] == rid and runs[0]["status"] == "running"
        panels = json.loads(_get(port, f"/api/runs/{rid}/panels")[1])
        titles = {p["title"] for s in panels["sections"] for p in s["panels"]}
        assert {"loss", "acc"} <= titles
        again = json.loads(_get(port, f"/api/runs/{rid}/panels?v={panels['version']}")[1])
        assert again.get("unchanged") is True
        media = json.loads(_get(port, f"/api/runs/{rid}/media")[1])["media"]
        path = media["qc/img"][0]["path"]
        assert _get(port, f"/files/{rid}/{path}")[0] == 200
        with pytest.raises(urllib.error.HTTPError):
            _get(port, f"/files/{rid}/..%2F..%2F..%2Fetc%2Fpasswd")
        assert b"RunRaccoon" in _get(port, "/")[1]
    finally:
        app.shutdown()
        wandb.finish()
