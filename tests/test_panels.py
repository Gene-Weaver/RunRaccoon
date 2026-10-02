from runraccoon.panels import build_sections, classify, flat_panels, guess_goal


def rows_of(n, **series):
    return [{"_step": i, **{k: f(i) for k, f in series.items()}} for i in range(n)]


def test_classify():
    assert classify("train/box_loss") == ("train", "box_loss", "Loss")
    assert classify("validation/acc") == ("val", "acc", "Performance")
    assert classify("val_loss") == ("val", "loss", "Loss")
    assert classify("loss", {"loss", "val_loss"}) == ("train", "loss", "Loss")      # Keras
    assert classify("val_px/median_px") == (None, "median_px", "val_px")           # not a role prefix
    assert classify("metrics/mAP50(B)") == (None, "mAP50(B)", "metrics")


def test_goal_guess():
    assert guess_goal("box_loss") == "min" and guess_goal("mAP50-95(B)") == "max"
    assert guess_goal("mean_corner_err_px") == "min" and guess_goal("detected") is None


def test_pairing_and_live_filter():
    rows = rows_of(5, **{"train/loss": lambda i: 1 / (i + 1), "val/loss": lambda i: 1.2 / (i + 1),
                         "test/loss": lambda i: 1.3, "val/acc": lambda i: i / 5, "lr/pg0": lambda i: 0.01,
                         "epoch": lambda i: i, "model/params": lambda i: 5})
    panels = {p.id: p for p in flat_panels(build_sections(rows, live=True))}
    loss = panels["Loss:loss"]
    assert [s.role for s in loss.series] == ["train", "val", "test"]
    assert loss.goal == "min" and loss.best()[0].role == "val"
    assert "key:lr/pg0" not in panels and "key:epoch" not in panels and "key:model/params" not in panels
    full = {p.id for p in flat_panels(build_sections(rows, live=False))}
    assert "key:lr/pg0" in full


def test_epoch_axis_when_logged():
    rows = rows_of(4, **{"val/loss": lambda i: i, "epoch": lambda i: i * 2})
    p = flat_panels(build_sections(rows))[0]
    assert p.x_label == "epoch" and p.series[0].x == [0, 2, 4, 6]


def test_define_metric_hides_and_rebases():
    rows = rows_of(4, **{"a": lambda i: i, "b": lambda i: i, "it": lambda i: 10 * i})
    defs = [{"name": "a", "hidden": True}, {"name": "b", "step_metric": "it"}]
    panels = flat_panels(build_sections(rows, defs))
    assert [p.title for p in panels] == ["b"] and panels[0].series[0].x == [0, 10, 20, 30]


def test_include_globs():
    rows = rows_of(3, **{"train/loss": lambda i: i, "val/acc": lambda i: i, "other": lambda i: i})
    titles = {p.title for p in flat_panels(build_sections(rows, include=["val/*"]))}
    assert titles == {"acc"}
