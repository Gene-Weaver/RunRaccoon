"""`wandb.plot` equivalents. Each returns a `CustomChart`, which when logged under `key`

  * saves its table to  files/media/table/<key>_table_<step>_<sha20>.table.json  (wandb's name), and
  * is rendered by RunRaccoon to  files/plots/charts/<key>.png

    run.log({"pr": runraccoon.plot.line_series(xs=[r, r2], ys=[p, p2], keys=["cat", "dog"])})
"""
from __future__ import annotations

from typing import Any, Sequence

from runraccoon.media import Table

__all__ = ["CustomChart", "line", "line_series", "scatter", "bar", "histogram", "pr_curve", "roc_curve",
           "confusion_matrix", "plot_table"]


class CustomChart:
    """A table plus a description of how to draw it.

    kind: "line" (x, y, optional series column), "scatter", "bar", "histogram", or "heatmap".
    """

    def __init__(self, table: Table, kind: str, fields: dict, title: str | None = None,
                 x_title: str | None = None, y_title: str | None = None, spec: str | None = None):
        self.table, self.kind, self.fields = table, kind, dict(fields)
        self.title, self.x_title, self.y_title, self.spec = title, x_title, y_title, spec

    def chart_meta(self) -> dict:
        return {"kind": self.kind, "fields": self.fields, "title": self.title, "x_title": self.x_title,
                "y_title": self.y_title, "spec": self.spec}

    def _bind(self, files_dir, key: str, step: int) -> dict:
        return self.table._bind(files_dir, f"{key}_table", step, extra={"_runraccoon_chart": self.chart_meta()})


# ------------------------------------------------------------------------------------- builders
def line(table: Table, x: str, y: str, stroke: str | None = None, title: str | None = None,
         split_table: bool = False) -> CustomChart:
    fields = {"x": x, "y": y}
    if stroke:
        fields["series"] = stroke
    return CustomChart(table, "line", fields, title=title, x_title=x, y_title=y, spec="wandb/line/v0")


def line_series(xs: Sequence[Sequence[float]] | Sequence[float], ys: Sequence[Sequence[float]],
                keys: Sequence[str] | None = None, title: str | None = None, xname: str | None = None,
                split_table: bool = False) -> CustomChart:
    ys = [list(y) for y in ys]
    shared_x = len(xs) > 0 and not hasattr(xs[0], "__len__")
    keys = list(keys) if keys is not None else [f"key_{i}" for i in range(len(ys))]
    rows = []
    for i, y in enumerate(ys):
        x = xs if shared_x else xs[i]
        rows.extend([keys[i], float(xv), float(yv)] for xv, yv in zip(x, y))
    table = Table(columns=["lineKey", "step", "lineVal"], data=rows)
    return CustomChart(table, "line", {"x": "step", "y": "lineVal", "series": "lineKey"}, title=title,
                       x_title=xname or "x", y_title=None, spec="wandb/lineseries/v0")


def scatter(table: Table, x: str, y: str, title: str | None = None, split_table: bool = False) -> CustomChart:
    return CustomChart(table, "scatter", {"x": x, "y": y}, title=title, x_title=x, y_title=y,
                       spec="wandb/scatter/v0")


def bar(table: Table, label: str, value: str, title: str | None = None, split_table: bool = False) -> CustomChart:
    return CustomChart(table, "bar", {"label": label, "value": value}, title=title, x_title=value,
                       spec="wandb/bar/v0")


def histogram(table: Table, value: str, title: str | None = None, split_table: bool = False) -> CustomChart:
    return CustomChart(table, "histogram", {"value": value}, title=title, x_title=value, y_title="count",
                       spec="wandb/histogram/v0")


def plot_table(vega_spec_name: str, data_table: Table, fields: dict, string_fields: dict | None = None,
               split_table: bool = False) -> CustomChart:
    """wandb's generic custom-chart call (used by Ultralytics for its PR / F1 curves)."""
    sf = string_fields or {}
    spec = vega_spec_name.lower()
    kind = ("scatter" if "scatter" in spec else "bar" if "bar" in spec else
            "histogram" if "histogram" in spec else "heatmap" if "confusion" in spec else "line")
    mapped = {"x": fields.get("x", "x"), "y": fields.get("y", "y")}
    series = fields.get("class") or fields.get("stroke") or fields.get("series") or fields.get("lineKey")
    if series:
        mapped["series"] = series
    if kind == "bar":
        mapped = {"label": fields.get("label", "label"), "value": fields.get("value", "value")}
    if kind == "histogram":
        mapped = {"value": fields.get("value", "value")}
    return CustomChart(data_table, kind, mapped, title=sf.get("title"), x_title=sf.get("x-axis-title"),
                       y_title=sf.get("y-axis-title"), spec=vega_spec_name)


# ------------------------------------------------------------------------ classification charts
def _class_names(n: int, labels: Sequence[str] | None) -> list[str]:
    return [str(c) for c in labels] if labels is not None else [str(i) for i in range(n)]


def _onehot(y_true, n_classes: int):
    import numpy as np
    y = np.asarray(y_true).ravel().astype(int)
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1
    return out


def pr_curve(y_true: Any = None, y_probas: Any = None, labels: Sequence[str] | None = None,
             classes_to_plot: Sequence[int] | None = None, title: str | None = None, **_: Any) -> CustomChart:
    import numpy as np
    probas = np.asarray(y_probas, dtype=float)
    probas = probas[:, None] if probas.ndim == 1 else probas
    if probas.shape[1] == 1:                        # binary scores -> two columns
        probas = np.hstack([1 - probas, probas])
    names = _class_names(probas.shape[1], labels)
    truth = _onehot(y_true, probas.shape[1])
    rows = []
    for c in classes_to_plot if classes_to_plot is not None else range(probas.shape[1]):
        order = np.argsort(-probas[:, c])
        tp = np.cumsum(truth[order, c])
        precision = tp / np.arange(1, tp.size + 1)
        recall = tp / max(tp[-1], 1)
        for r, p in zip(recall[:: max(1, tp.size // 200)], precision[:: max(1, tp.size // 200)]):
            rows.append([names[c], round(float(r), 4), round(float(p), 4)])
    table = Table(columns=["class", "recall", "precision"], data=rows)
    return CustomChart(table, "line", {"x": "recall", "y": "precision", "series": "class"},
                       title=title or "Precision-Recall", x_title="Recall", y_title="Precision", spec="wandb/pr/v0")


def roc_curve(y_true: Any = None, y_probas: Any = None, labels: Sequence[str] | None = None,
              classes_to_plot: Sequence[int] | None = None, title: str | None = None, **_: Any) -> CustomChart:
    import numpy as np
    probas = np.asarray(y_probas, dtype=float)
    probas = probas[:, None] if probas.ndim == 1 else probas
    if probas.shape[1] == 1:
        probas = np.hstack([1 - probas, probas])
    names = _class_names(probas.shape[1], labels)
    truth = _onehot(y_true, probas.shape[1])
    rows = []
    for c in classes_to_plot if classes_to_plot is not None else range(probas.shape[1]):
        order = np.argsort(-probas[:, c])
        pos = truth[order, c]
        tpr = np.concatenate([[0], np.cumsum(pos) / max(pos.sum(), 1)])
        fpr = np.concatenate([[0], np.cumsum(1 - pos) / max((1 - pos).sum(), 1)])
        stride = max(1, tpr.size // 200)
        rows.extend([names[c], round(float(f), 4), round(float(t), 4)] for f, t in zip(fpr[::stride], tpr[::stride]))
    table = Table(columns=["class", "fpr", "tpr"], data=rows)
    return CustomChart(table, "line", {"x": "fpr", "y": "tpr", "series": "class"}, title=title or "ROC",
                       x_title="False positive rate", y_title="True positive rate", spec="wandb/roc/v0")


def confusion_matrix(probs: Any = None, y_true: Any = None, preds: Any = None,
                     class_names: Sequence[str] | None = None, title: str | None = None, **_: Any) -> CustomChart:
    import numpy as np
    if preds is None:
        preds = np.asarray(probs).argmax(axis=1)
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(preds).ravel().astype(int)
    n = int(max(y.max(initial=0), p.max(initial=0)) + 1)
    names = _class_names(n, class_names)
    counts = np.zeros((len(names), len(names)), dtype=int)
    np.add.at(counts, (y, p), 1)
    rows = [[names[i], names[j], int(counts[i, j])] for i in range(len(names)) for j in range(len(names))]
    table = Table(columns=["Actual", "Predicted", "nPredictions"], data=rows)
    return CustomChart(table, "heatmap", {"x": "Predicted", "y": "Actual", "value": "nPredictions"},
                       title=title or "Confusion matrix", x_title="Predicted", y_title="Actual",
                       spec="wandb/confusion_matrix/v1")
