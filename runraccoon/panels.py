"""Turn history rows into chart panels - the one place that decides *what* gets plotted.

Rules (shared by the PNG renderer and the dashboard so they always agree):

* **Pairing.** `train/box_loss`, `val/box_loss` and `test/box_loss` share one panel, as do
  Keras-style `loss` / `val_loss` and `train_acc` / `val_acc`. Paired panels live in the
  "Loss" and "Performance" sections. Role prefixes: train/training, val/valid/validation/eval/dev,
  test/testing.
* **Sections.** Every other key is grouped by its prefix before the first `/` (wandb's panel
  sections): `metrics/mAP50(B)` -> section "metrics". Keys without a prefix go to "Other".
* **What is "important".** Live plots skip learning rates, counters (epoch/step/iter...), and
  bookkeeping sections (model/, system/, time/). Pass explicit globs to override.
* **x-axis.** `define_metric(..., step_metric=...)` wins; otherwise `epoch` if it is logged
  alongside the metric and increases monotonically; otherwise the wandb `_step`.
* **Better direction.** loss/error/MAE/... are minimized, accuracy/mAP/IoU/F1/... maximized,
  `define_metric(goal=..., summary="min"|"max")` overrides. The best point is marked.
"""
from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

ROLE_ALIASES = {"train": "train", "training": "train", "trn": "train",
                "val": "val", "valid": "val", "validation": "val", "eval": "val", "dev": "val",
                "test": "test", "testing": "test"}
ROLES = ("train", "val", "test")
_ROLE_UNDERSCORE = re.compile(r"^(train|training|val|valid|validation|eval|test|testing)[_\-](.+)$", re.I)

COUNTER_NAMES = {"epoch", "epochs", "step", "steps", "global_step", "iter", "iters", "iteration",
                 "iterations", "batch", "batch_i", "batch_idx", "sample", "samples", "examples"}
LIVE_QUIET_SECTIONS = {"lr", "model", "system", "time", "timing", "perf", "speed", "throughput", "params", "trainer"}
_LR = re.compile(r"(^|[/_\-.])(lr|learning[_\-]?rate)([/_\-.]|$)|^lr", re.I)
_MINIMIZE = re.compile(r"loss|err|error|mae|mse|rmse|nll|perplexity|ppl|(^|[^a-z])(cer|wer)([^a-z]|$)|dist|px|latency", re.I)
_MAXIMIZE = re.compile(r"acc|map|iou|dice|f1|precision|recall|auc|(^|[^a-z])ap([^a-z]|$)|score|psnr|ssim|r2|fitness", re.I)

SECTION_LOSS, SECTION_PERF, SECTION_OTHER = "Loss", "Performance", "Other"


# ------------------------------------------------------------------------------------- model
@dataclass
class Series:
    key: str
    label: str
    role: str | None
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)


@dataclass
class Panel:
    id: str
    title: str
    section: str
    series: list[Series]
    goal: str | None = None          # "min" | "max" | None
    x_label: str = "step"
    x_key: str = "_step"             # the row field used as x (epoch, a step_metric, or _step)

    @property
    def is_loss(self) -> bool:
        return "loss" in self.title.lower()

    def primary(self) -> Series:
        """The series the 'best' marker refers to: val, then test, then train, then the first."""
        for role in ("val", "test", "train"):
            for s in self.series:
                if s.role == role and s.y:
                    return s
        return self.series[0]

    def best(self) -> tuple[Series, float, float] | None:
        if self.goal not in ("min", "max"):
            return None
        s = self.primary()
        pts = [(x, y) for x, y in zip(s.x, s.y) if y is not None and math.isfinite(y)]
        if len(pts) < 2 or len({y for _, y in pts}) == 1:       # nothing to mark on a constant series
            return None
        x, y = (min if self.goal == "min" else max)(pts, key=lambda p: p[1])
        return s, x, y

    def to_json(self, max_points: int | None = None, best_row: dict | None = None) -> dict:
        best = self.best()
        out = {"id": self.id, "title": self.title, "section": self.section, "goal": self.goal,
               "x_label": self.x_label, "series": []}
        if best_row is not None and _num(best_row.get(self.x_key)) is not None:
            out["checkpoint_x"] = _num(best_row.get(self.x_key))
        for s in self.series:
            x, y = (s.x, s.y) if not max_points else downsample(s.x, s.y, max_points)
            out["series"].append({"key": s.key, "label": s.label, "role": s.role, "x": x,
                                  "y": [v if v is not None and math.isfinite(v) else None for v in y],
                                  "last": _last_finite(s.y)})
        if best:
            out["best"] = {"key": best[0].key, "x": best[1], "y": best[2]}
        return out


@dataclass
class Section:
    name: str
    panels: list[Panel]


# ----------------------------------------------------------------------------- classification
def classify(key: str, all_keys: set[str] | None = None) -> tuple[str | None, str, str]:
    """-> (role, base_name, section) for a metric key."""
    if "/" in key:
        head, rest = key.split("/", 1)
        role = ROLE_ALIASES.get(head.lower())
        if role:
            return role, rest, SECTION_LOSS if "loss" in rest.lower() else SECTION_PERF
        return None, rest, head
    m = _ROLE_UNDERSCORE.match(key)
    if m:
        base = m.group(2)
        return ROLE_ALIASES[m.group(1).lower()], base, SECTION_LOSS if "loss" in base.lower() else SECTION_PERF
    if all_keys and any(f"{p}{key}" in all_keys for p in ("val_", "val/", "validation_", "test_", "test/")):
        return "train", key, SECTION_LOSS if "loss" in key.lower() else SECTION_PERF   # Keras: loss / val_loss
    return None, key, SECTION_OTHER


def is_counter(key: str) -> bool:
    leaf = key.rsplit("/", 1)[-1].lower()
    return key.lower() in COUNTER_NAMES or leaf in COUNTER_NAMES


def is_lr(key: str) -> bool:
    return bool(_LR.search(key))


def guess_goal(name: str) -> str | None:
    if _MINIMIZE.search(name):
        return "min"
    if _MAXIMIZE.search(name):
        return "max"
    return None


def _match_def(key: str, defs: list[dict]) -> dict:
    """The define_metric spec that applies to `key` (exact name wins over glob)."""
    best = {}
    for d in defs:
        name = d.get("name", "")
        if name == key:
            return d
        if any(c in name for c in "*?[") and fnmatch.fnmatchcase(key, name):
            best = d
    return best


# ------------------------------------------------------------------------------------ builder
def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def build_sections(rows: list[dict], metric_defs: Iterable[dict] | None = None,
                   include: Iterable[str] | None = None, live: bool = False,
                   x_label_hint: str | None = None) -> list[Section]:
    """Group every plottable scalar in `rows` into sections of panels.

    include: glob patterns; when given, only matching keys are plotted (live_metrics setting).
    live:    drop learning rates and bookkeeping sections (the per-epoch progress figure).
    """
    defs = list(metric_defs or [])
    include = list(include or [])

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k, v in row.items():
            if k not in seen and not k.startswith("_") and _num(v) is not None:
                seen.add(k)
                keys.append(k)

    step_metrics = {d["step_metric"] for d in defs if d.get("step_metric")}

    def wanted(k: str) -> bool:
        if k in step_metrics:                  # an x-axis, not a metric
            return False
        if include:
            return any(fnmatch.fnmatchcase(k, pat) for pat in include)
        if is_counter(k) or _match_def(k, defs).get("hidden"):
            return False
        if live:
            role, _, section = classify(k, seen)
            if is_lr(k) or (role is None and section.lower() in LIVE_QUIET_SECTIONS):
                return False
        return True

    keys = [k for k in keys if wanted(k)]
    epoch_x = _epoch_is_usable(rows)

    panels: dict[str, Panel] = {}
    for key in keys:
        spec = _match_def(key, defs)
        role, base, section = classify(key, seen)
        x_metric = spec.get("step_metric") or ("epoch" if key in epoch_x else None)
        xs, ys = [], []
        for row in rows:
            y = row.get(key)
            if y is None or isinstance(y, (dict, str)) or isinstance(y, bool):
                continue
            x = _num(row.get(x_metric)) if x_metric else _num(row.get("_step"))
            if x is None:
                continue
            xs.append(x)
            ys.append(float(y))
        if not xs:
            continue
        pid = f"{section}:{base}" if role else f"key:{key}"
        title = base if role else key          # paired panels: "box_loss"; others: the full logged key
        panel = panels.get(pid)
        if panel is None:
            goal = spec.get("goal") or ({"min": "min", "max": "max"}.get(spec.get("summary", "")) or guess_goal(base))
            goal = {"minimize": "min", "maximize": "max"}.get(goal, goal)
            x_label = x_metric or x_label_hint or "step"
            panel = panels[pid] = Panel(pid, title, section, [], goal, x_label, x_metric or "_step")
        panel.series.append(Series(key, role or title, role, xs, ys))

    for p in panels.values():
        p.series.sort(key=lambda s: ROLES.index(s.role) if s.role in ROLES else 9)
    # One-off values (model/parameters, a final score) are summary material, not charts - but only
    # once the run has real curves; early in training every metric has a single point.
    longest = max((len(s.x) for p in panels.values() for s in p.series), default=0)
    if longest >= 3:
        panels = {k: p for k, p in panels.items() if max(len(s.x) for s in p.series) > 1}

    order: list[str] = []
    for p in panels.values():
        if p.section not in order:
            order.append(p.section)
    rank = {SECTION_LOSS: 0, SECTION_PERF: 1}
    order.sort(key=lambda s: (rank.get(s, 2), s == SECTION_OTHER, s.lower() == "lr"))
    return [Section(s, [p for p in panels.values() if p.section == s]) for s in order]


def _epoch_is_usable(rows: list[dict]) -> set[str]:
    """Keys whose rows (nearly) always carry a monotonically increasing `epoch`."""
    count: dict[str, int] = {}
    with_epoch: dict[str, list[float]] = {}
    for row in rows:
        e = _num(row.get("epoch"))
        for k, v in row.items():
            if k.startswith("_") or _num(v) is None:
                continue
            count[k] = count.get(k, 0) + 1
            if e is not None:
                with_epoch.setdefault(k, []).append(e)
    out = set()
    for k, es in with_epoch.items():
        if len(es) >= 0.9 * count[k] and all(b > a for a, b in zip(es, es[1:])):
            out.add(k)
    return out


# ------------------------------------------------------------------------ best checkpoint
_ULTRA_MAP5095 = re.compile(r"^metrics/mAP50-95\((\w+)\)$")
_VAL_LOSS_KEYS = ("val/loss", "val_loss", "validation/loss", "valid/loss", "eval/loss")


def _version_tuple(v: str) -> tuple:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3])


def ultralytics_version(requirements_txt: str) -> str | None:
    m = re.search(r"(?im)^ultralytics==([\w.]+)", requirements_txt or "")
    return m.group(1) if m else None


def _argbest(rows: list[dict], score, goal: str):
    """(row, value) of the best row. Only a strictly better value replaces the incumbent, so ties
    keep the earliest step - the same rule Ultralytics uses for best.pt."""
    best = None
    for row in rows:
        v = score(row)
        if v is None or not math.isfinite(v):
            continue
        if best is None or (v > best[1] if goal == "max" else v < best[1]):
            best = (row, v)
    return best


def best_checkpoint(rows: list[dict], metric_defs: Iterable[dict] | None = None,
                    ultralytics_ver: str | None = None) -> dict | None:
    """Which step would be kept as the "best" model if training stopped now.

    1. `define_metric(key, summary="min"|"max")` or `goal=` (wandb's way to mark the metric
       that decides "best") - the first such exact key wins.
    2. Ultralytics runs (metrics/mAP50-95(B) etc. logged): best.pt fitness, version-matched:
       >= 8.3.198 sums mAP50-95 over box + pose/mask; older uses 0.1*mAP50 + 0.9*mAP50-95.
       Only per-epoch rows (with train/* losses) count, so the post-training re-validation of
       best.pt is never itself picked.
    3. Otherwise the lowest validation loss.
    """
    for d in metric_defs or []:
        name = d.get("name", "")
        goal = d.get("goal") or (d.get("summary") if d.get("summary") in ("min", "max") else None)
        goal = {"minimize": "min", "maximize": "max"}.get(goal, goal)
        if goal in ("min", "max") and name and not any(c in name for c in "*?["):
            hit = _argbest(rows, lambda r, n=name: _num(r.get(n)), goal)
            if hit:
                return {"row": hit[0], "value": hit[1], "rule": f"{goal} {name}", "source": "define_metric"}

    suffixes = sorted({m.group(1) for r in rows for k in r if (m := _ULTRA_MAP5095.match(k))})
    if suffixes:
        old = ultralytics_ver is not None and _version_tuple(ultralytics_ver) < (8, 3, 198)
        w50, w95 = (0.1, 0.9) if old else (0.0, 1.0)

        def fitness(r: dict) -> float | None:
            if not any(k.startswith("train/") for k in r):
                return None
            total = 0.0
            for s in suffixes:
                m95, m50 = _num(r.get(f"metrics/mAP50-95({s})")), _num(r.get(f"metrics/mAP50({s})"))
                if m95 is None:
                    return None
                total += w95 * m95 + w50 * (m50 or 0.0)
            return total

        hit = _argbest(rows, fitness, "max")
        if hit:
            parts = [f"mAP50-95({s})" if not old else f"0.1·mAP50({s})+0.9·mAP50-95({s})" for s in suffixes]
            return {"row": hit[0], "value": hit[1], "rule": "Ultralytics best.pt fitness = " + " + ".join(parts),
                    "source": "ultralytics"}

    for key in _VAL_LOSS_KEYS:
        if any(key in r for r in rows):
            hit = _argbest(rows, lambda r, k=key: _num(r.get(k)), "min")
            if hit:
                return {"row": hit[0], "value": hit[1], "rule": f"min {key}", "source": "val_loss"}
    return None


def estimate_end(rows: list[dict], total_epochs: Any) -> float | None:
    """Projected wall-clock time training finishes, or None until it can be estimated.

    Needs at least one completed epoch *including validation* (a row with val/ or metrics/ keys),
    so the projection covers a full train+val cycle. Uses absolute `_timestamp`s (robust to
    resumes): last completed epoch + remaining epochs x median duration of recent epochs
    (with one epoch done, its duration - incl. startup - is the estimate).
    """
    if not isinstance(total_epochs, int) or isinstance(total_epochs, bool) or total_epochs <= 0:
        return None
    done = [r["_timestamp"] for r in rows
            if isinstance(r.get("_timestamp"), (int, float))
            and any(k.startswith(("val/", "val_", "metrics/", "validation/")) for k in r)
            and any(not k.startswith("_") for k in r)]
    done = done[:total_epochs]                 # e.g. Ultralytics' extra re-validation of best.pt
    if not done:
        return None
    if len(done) == 1:
        first = next((r["_timestamp"] - r.get("_runtime", 0) for r in rows if isinstance(r.get("_timestamp"), (int, float))), None)
        if first is None:
            return None
        per_epoch = done[0] - first
    else:
        recent = done[-11:]
        diffs = sorted(b - a for a, b in zip(recent, recent[1:]))
        per_epoch = diffs[len(diffs) // 2]
    return done[-1] + max(0, total_epochs - len(done)) * per_epoch


def flat_panels(sections: list[Section]) -> list[Panel]:
    return [p for s in sections for p in s.panels]


def training_rows(rows: list[dict], config: dict) -> list[dict]:
    """Rows of the training epochs only. Ultralytics logs a post-training re-validation of best.pt
    at step epochs+1; plotted as "epoch 51 of 50" it would misreport where the best value was."""
    epochs = config.get("epochs")
    if isinstance(epochs, int) and not isinstance(epochs, bool) and x_label_hint(rows, config) == "epoch":
        return [r for r in rows if not (isinstance(r.get("_step"), (int, float)) and r["_step"] > epochs)]
    return rows


def x_label_hint(rows: list[dict], config: dict) -> str | None:
    """Ultralytics logs with step == epoch. If every step fits inside config['epochs'], say 'epoch'."""
    epochs = config.get("epochs")
    if isinstance(epochs, int) and rows:
        steps = [r.get("_step") for r in rows if isinstance(r.get("_step"), (int, float))]
        if steps and max(steps) <= epochs + 2 and min(steps) >= 0:
            return "epoch"
    return None


# --------------------------------------------------------------------------------- utilities
def downsample(x: list[float], y: list[float], max_points: int) -> tuple[list[float], list[float]]:
    """Keep the min and max of each bucket so spikes survive; always keep the last point."""
    n = len(x)
    if n <= max_points:
        return list(x), list(y)
    buckets = max(1, max_points // 2)
    size = n / buckets
    ox, oy = [], []
    for b in range(buckets):
        lo, hi = int(b * size), min(n, int((b + 1) * size))
        idx = [i for i in range(lo, hi) if y[i] is not None and math.isfinite(y[i])]
        if not idx:
            continue
        i_min = min(idx, key=lambda i: y[i])
        i_max = max(idx, key=lambda i: y[i])
        for i in sorted({i_min, i_max}):
            ox.append(x[i])
            oy.append(y[i])
    if ox and ox[-1] != x[-1]:
        ox.append(x[-1])
        oy.append(y[-1])
    return ox, oy


def _last_finite(ys: list[float]) -> float | None:
    for v in reversed(ys):
        if v is not None and math.isfinite(v):
            return v
    return None


def ema(values: list[float], weight: float) -> list[float]:
    """wandb-style exponential moving average with debiasing."""
    out, last, num = [], 0.0, 0
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(v)
            continue
        num += 1
        last = last * weight + (1 - weight) * v
        out.append(last / (1 - weight ** num))
    return out
