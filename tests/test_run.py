import json
import sys

import numpy as np
import pytest
import yaml

import runraccoon as wandb
from runraccoon.history import read_rows
from runraccoon.paths import RunPaths


def _paths(run) -> RunPaths:
    return run.paths


def test_import_replaces_wandb_module():
    assert sys.modules["wandb"] is wandb
    import wandb as w  # noqa: F401 - resolves to RunRaccoon
    assert w.init is wandb.init and hasattr(w, "__version__")


def test_layout_matches_wandb(isolated):
    run = wandb.init(project="p", name="n", config={"lr": 0.1, "layers": [1, 2]})
    wandb.log({"train/loss": 1.0})
    p = _paths(run)
    wandb.finish()

    assert p.run_dir.parent == isolated / "runraccoon"
    assert p.run_dir.name.startswith("run-") and p.run_dir.name.endswith(f"-{run.id}")
    assert (isolated / "runraccoon" / "latest-run").resolve() == p.run_dir
    for f in ("config.yaml", "wandb-summary.json", "wandb-history.jsonl", "wandb-metadata.json", "requirements.txt"):
        assert (p.files / f).exists(), f
    cfg = yaml.safe_load(p.config.read_text())
    assert cfg["lr"] == {"value": 0.1} and cfg["layers"] == {"value": [1, 2]}
    assert cfg["_wandb"]["value"]["project"] == "p"
    summary = json.loads(p.summary.read_text())
    assert summary["train/loss"] == 1.0 and "_runtime" in summary
    assert wandb.run is None


def test_step_semantics():
    run = wandb.init()
    wandb.log({"a": 1})                       # step 0
    wandb.log({"b": 2}, commit=False)         # step 1, open
    wandb.log({"c": 3})                       # step 1, committed
    wandb.log({"d": 4}, step=5)               # opens 5 (nothing pending)
    wandb.log({"e": 5}, step=5)               # merges into 5
    wandb.log({"f": 6}, step=7)               # commits 5, opens 7
    wandb.log({"late": 1}, step=5)            # already committed: kept as an extra row
    p = _paths(run)
    wandb.finish()
    rows = {r["_step"]: r for r in read_rows(p.history)}
    assert rows[0]["a"] == 1
    assert rows[1]["b"] == 2 and rows[1]["c"] == 3
    assert rows[5]["d"] == 4 and rows[5]["e"] == 5 and rows[5]["late"] == 1
    assert rows[7]["f"] == 6


def test_values_are_normalized():
    run = wandb.init()
    wandb.log({"np": np.float32(0.5), "int": np.int64(3), "nested": {"x": 1}, "arr": np.arange(10), "s": "hi"})
    p = _paths(run)
    wandb.finish()
    row = read_rows(p.history)[0]
    assert row["np"] == 0.5 and row["int"] == 3 and row["nested.x"] == 1 and row["s"] == "hi"
    assert row["arr"]["_type"] == "histogram"


def test_image_files_use_wandb_names():
    run = wandb.init()
    wandb.log({"qc/sheet": wandb.Image(np.zeros((8, 12, 3), dtype=np.uint8), caption="c")}, step=3)
    wandb.log({"grid": [wandb.Image(np.ones((4, 4))), wandb.Image(np.zeros((4, 4)))]}, step=4)
    p = _paths(run)
    wandb.finish()
    files = sorted(x.relative_to(p.files).as_posix() for x in (p.media / "images").rglob("*.png"))
    assert any(f.startswith("media/images/qc/sheet_3_") and len(f.split("_")[-1]) == 20 + 4 for f in files)
    rows = {r["_step"]: r for r in read_rows(p.history)}
    assert rows[3]["qc/sheet"]["_type"] == "image-file" and rows[3]["qc/sheet"]["caption"] == "c"
    assert rows[4]["grid"]["_type"] == "images/separated" and rows[4]["grid"]["count"] == 2


def test_image_inputs(tmp_path):
    from PIL import Image as PILImage
    PILImage.new("RGB", (5, 7)).save(tmp_path / "x.jpg")
    assert wandb.Image(str(tmp_path / "x.jpg")).format == "jpg"
    assert wandb.Image(np.random.rand(3, 6, 9)).width == 9           # CHW float
    assert wandb.Image(np.random.rand(6, 9) * 50).height == 6         # arbitrary float range
    assert wandb.Image(PILImage.new("L", (2, 3))).width == 2


def test_custom_chart_table_name():
    run = wandb.init()
    wandb.log({"curves/PR": wandb.plot.line_series(xs=[0, 0.5, 1], ys=[[1, 0.8, 0.1]], keys=["a"])})
    p = _paths(run)
    wandb.finish()
    tables = [x.name for x in (p.media / "table").rglob("*.table.json")]
    assert tables and tables[0].startswith("PR_table_0_")
    assert "curves/PR_table" in json.loads(p.summary.read_text())


def test_define_metric_summary():
    run = wandb.init()
    wandb.define_metric("val/loss", summary="min")
    for v in (3.0, 1.0, 2.0):
        wandb.log({"val/loss": v})
    assert run.summary["val/loss"] == 1.0
    wandb.finish()


def test_resume_must():
    run = wandb.init(project="p", name="r", config={"a": 1})
    wandb.log({"x": wandb.Image(np.zeros((4, 4)))})
    wandb.log({"loss": 1.0})
    first = _paths(run).run_dir
    rid = run.id
    wandb.finish()

    import time
    time.sleep(1.1)                            # new run dir timestamp
    run2 = wandb.init(id=rid, resume="must")
    assert run2.resumed and run2.name == "r" and run2.config["a"] == 1 and run2.step == 2
    wandb.log({"loss": 0.5})
    second = _paths(run2).run_dir
    wandb.finish()
    assert second != first and second.name.endswith(rid)
    rows = read_rows(_paths(run2).history)
    assert [r["_step"] for r in rows] == [0, 1, 2]
    assert list((second / "files" / "media" / "images").glob("*.png"))

    with pytest.raises(ValueError):
        wandb.init(id="doesnotexist", resume="must")


def test_reinit_finishes_previous():
    a = wandb.init()
    b = wandb.init()
    assert a._finished and wandb.run is b


def test_disabled_mode_writes_nothing(isolated):
    run = wandb.init(mode="disabled")
    wandb.log({"a": 1})
    wandb.config.update({"b": 2})
    wandb.finish()
    assert not (isolated / "runraccoon").exists() and run.disabled


def test_artifact_manifest(tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights")
    run = wandb.init()
    art = wandb.Artifact("model", type="model")
    art.add_file(str(w))
    run.log_artifact(art, aliases=["best"])
    manifest = json.loads((run.paths.artifacts / "model" / "manifest.json").read_text())
    assert manifest["files"][0]["path"] == str(w) and manifest["aliases"] == ["latest", "best"]


def test_registry_status():
    from runraccoon import registry
    run = wandb.init(name="reg")
    assert registry.status(registry.read(run.id)) == "running"
    wandb.finish()
    assert registry.status(registry.read(run.id)) == "finished"
