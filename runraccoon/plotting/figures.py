"""Figure builders: the per-epoch progress grid, per-section summaries, the run report card, and
custom charts (PR curves, confusion matrices, ...). Uses matplotlib's object API only - never
pyplot - so nothing here touches global figure state.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from runraccoon.panels import Panel, ema
from runraccoon.plotting.style import AXIS, INK, INK_2, MUTED, NEUTRAL, PALETTE, ROLE_STYLE, SURFACE
from runraccoon.utils import format_value

PANEL_W, PANEL_H = 2.35, 1.6          # inches
GAP_W, GAP_H = 0.5, 0.62
MARGIN_L, MARGIN_R, MARGIN_B = 0.45, 0.2, 0.66
HEADER_H = 1.05
DENSE_ABOVE = 80                      # above this many points lines are drawn solid and thinner
SMOOTH_ABOVE = 400                    # per-epoch curves (even 300 epochs) are drawn raw, like wandb


# ---------------------------------------------------------------------------------- panels
def _tick_fmt(v: float, _pos=None) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e4 or a < 1e-3:
        return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")
    if a >= 100:
        return f"{v:,.0f}"
    return f"{v:.3g}"


def _adaptive_formatter(axis):
    """Tick labels with just enough decimals to tell neighboring ticks apart."""
    def fmt(v: float, _pos=None) -> str:
        locs = [t for t in axis.get_majorticklocs() if np.isfinite(t)]
        step = min((abs(b - a) for a, b in zip(locs, locs[1:]) if b != a), default=0)
        if not step:
            return _tick_fmt(v)
        mag = max(abs(locs[0]), abs(locs[-1]))
        if mag >= 1e5 or (mag < 1e-3 and mag > 0):
            digits = max(0, int(math.ceil(-math.log10(step / mag))) )
            return f"{v:.{min(digits, 6)}e}".replace("e-0", "e-").replace("e+0", "e")
        digits = max(0, int(math.ceil(-math.log10(step) - 1e-9)))
        return f"{v:,.{min(digits, 8)}f}" if digits else f"{v:,.0f}"
    return FuncFormatter(fmt)


def _style_for(series_role: str | None, index: int) -> dict:
    if series_role in ROLE_STYLE:
        st = dict(ROLE_STYLE[series_role])
        st.pop("label")
        return st
    # Unpaired metrics get a neutral ink, so they never read as "train" in the role legend.
    slots = [NEUTRAL, PALETTE[6], PALETTE[3], PALETTE[4], PALETTE[5], PALETTE[7]]
    return {"color": slots[index] if index < len(slots) else MUTED, "linestyle": "-"}


def draw_panel(ax, panel: Panel, show_xlabel: bool = True, title: str | None = None) -> None:
    finite_all = []
    for i, s in enumerate(panel.series):
        st = _style_for(s.role, i)
        x = np.asarray(s.x, dtype=float)
        y = np.asarray(s.y, dtype=float)
        n = len(x)
        finite_all.append(y[np.isfinite(y)])
        if n > SMOOTH_ABOVE:              # long per-iteration series: faint raw + gently smoothed line
            weight = 0.6 + 0.3 * min(1.0, (n - SMOOTH_ABOVE) / 4000)
            ax.plot(x, y, color=st["color"], lw=0.6, alpha=0.2, solid_capstyle="butt")
            ax.plot(x, ema(list(y), weight), lw=1.5, **st)
        elif n > DENSE_ABOVE:              # dense curves: dashes turn into noise, so solid + thinner
            ax.plot(x, y, lw=1.05, color=st["color"], linestyle="-")
        else:
            ax.plot(x, y, lw=1.5, **st)
        if n <= 12:
            ax.plot(x, y, ls="none", marker="o", ms=3.4, mfc=st["color"], mec=SURFACE, mew=0.9, zorder=3)

    best = panel.best()
    if best:
        s, bx, by = best
        color = _style_for(s.role, panel.series.index(s))["color"]
        ax.plot([bx], [by], ls="none", marker="o", ms=6.0, mfc=SURFACE, mec=color, mew=1.5, zorder=6)
        x_word = panel.x_label if panel.x_label != "_step" else "step"
        ax.text(1.0, 1.025, f"{'min' if panel.goal == 'min' else 'max'} {format_value(by)} @ {x_word} {format_value(bx) if bx % 1 else int(bx)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=MUTED)

    ys = np.concatenate(finite_all) if finite_all else np.array([])
    if panel.is_loss and ys.size and ys.min() > 0 and ys.max() / ys.min() > 30:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_tick_fmt))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    else:
        if ys.size:                      # a (nearly) constant series is drawn flat, not magnified noise
            lo, hi = float(ys.min()), float(ys.max())
            floor = max(abs(lo), abs(hi)) * 5e-3 or 1e-9
            if hi - lo < floor:
                mid = (hi + lo) / 2
                ax.set_ylim(mid - floor, mid + floor)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(_adaptive_formatter(ax.yaxis))
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.xaxis.set_major_formatter(_adaptive_formatter(ax.xaxis))
    t = title or panel.title
    ax.set_title(t if len(t) <= 34 else t[:32] + "…")
    if show_xlabel:
        ax.set_xlabel(panel.x_label if panel.x_label != "_step" else "step", labelpad=2)
    if len(panel.series) > 1 and any(s.role is None for s in panel.series):
        ax.legend(loc="best")


def _roles_in(panels: Sequence[Panel]) -> list[str]:
    roles = {s.role for p in panels for s in p.series if s.role}
    return [r for r in ("train", "val", "test") if r in roles]


def _header(fig: Figure, title: str, subtitle: str, roles: Sequence[str], width_in: float, height_in: float,
            dense: bool = False) -> None:
    top = 1 - 0.2 / height_in
    fig.text(MARGIN_L / width_in, top, title, ha="left", va="top", fontsize=11, fontweight="semibold", color=INK)
    fig.text(MARGIN_L / width_in, top - 0.26 / height_in, subtitle, ha="left", va="top", fontsize=7.2, color=MUTED)
    if len(roles) >= 2 or (roles and roles != ["train"]):
        handles = [Line2D([], [], lw=1.6, color=ROLE_STYLE[r]["color"], linestyle="-" if dense else ROLE_STYLE[r]["linestyle"])
                   for r in roles]
        fig.legend(handles, roles, loc="upper right", ncol=len(roles), fontsize=7.2, frameon=False,
                   bbox_to_anchor=(1 - MARGIN_R / width_in, top + 0.04 / height_in), handlelength=2.4,
                   columnspacing=1.4, borderaxespad=0)


def _footer(fig: Figure, text: str, width_in: float, height_in: float) -> None:
    max_chars = int((width_in - MARGIN_L - MARGIN_R) * 24)       # ~24 chars per inch at 5.8 pt
    if len(text) > max_chars:
        text = text[: max_chars // 3] + " … " + text[-(2 * max_chars // 3 - 3):]
    fig.text(MARGIN_L / width_in, 0.08 / height_in, text, ha="left", va="bottom", fontsize=5.8, color=MUTED)


def _grid_figure(panels: Sequence[Panel], title: str, subtitle: str, footer: str, ncols: int = 4,
                 top_extra_in: float = 0.0, header_roles: Sequence[str] | None = None):
    n = max(1, len(panels))
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    width = MARGIN_L + MARGIN_R + ncols * PANEL_W + (ncols - 1) * GAP_W
    width = max(width, 5.2)
    height = HEADER_H + top_extra_in + nrows * PANEL_H + (nrows - 1) * GAP_H + MARGIN_B
    fig = Figure(figsize=(width, height))
    gs = fig.add_gridspec(nrows, ncols, left=MARGIN_L / width,
                          right=(MARGIN_L + ncols * PANEL_W + (ncols - 1) * GAP_W) / width,
                          top=1 - (HEADER_H + top_extra_in) / height, bottom=MARGIN_B / height,
                          wspace=GAP_W / PANEL_W, hspace=GAP_H / PANEL_H)
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        below = i + ncols >= n                     # nothing underneath -> this panel shows the x label
        draw_panel(ax, panel, show_xlabel=below)
    dense = all(len(s.x) > DENSE_ABOVE for p in panels for s in p.series if s.role)
    _header(fig, title, subtitle, header_roles if header_roles is not None else _roles_in(panels), width, height, dense)
    _footer(fig, footer, width, height)
    return fig, width, height


def _footer_text(run_dir: str) -> str:
    return f"RunRaccoon · rendered {_dt.datetime.now():%Y-%m-%d %H:%M} · {run_dir}"


# ---------------------------------------------------------------------------- figure kinds
def progress_figure(panels: Sequence[Panel], run_title: str, subtitle: str, run_dir: str,
                    max_panels: int = 16) -> Figure | None:
    if not panels:
        return None
    shown = list(panels)[:max_panels]
    if len(panels) > max_panels:
        extra = len(panels) - max_panels
        subtitle += f"   ·   {extra} more panel{'s' if extra != 1 else ''} in the summary figures"
    fig, _, _ = _grid_figure(shown, run_title, subtitle, _footer_text(run_dir))
    return fig


def section_figure(section_name: str, panels: Sequence[Panel], run_title: str, subtitle: str,
                   run_dir: str) -> Figure | None:
    if not panels:
        return None
    fig, _, _ = _grid_figure(panels, f"{run_title}  ·  {section_name}", subtitle, _footer_text(run_dir))
    return fig


def report_figure(panels: Sequence[Panel], tiles: Sequence[tuple[str, str, str]], run_title: str,
                  subtitle: str, run_dir: str) -> Figure | None:
    """The run report card: headline numbers on top, the key curves below."""
    if not panels and not tiles:
        return None
    tiles = list(tiles)[:6]
    tile_h = 0.6 if tiles else 0.0
    fig, width, height = _grid_figure(list(panels)[:8] or [], run_title, subtitle, _footer_text(run_dir),
                                      top_extra_in=tile_h)
    if not panels:
        for ax in fig.axes:
            ax.set_visible(False)
    if tiles:
        y_top = 1 - 0.68 / height                  # just under the subtitle
        left, right = MARGIN_L / width, 1 - MARGIN_R / width
        step = (right - left) / len(tiles)
        for i, (label, value, sub) in enumerate(tiles):
            x = left + i * step
            if i:
                fig.add_artist(Line2D([x - 0.012, x - 0.012], [y_top - 0.62 / height, y_top - 0.02 / height],
                                      transform=fig.transFigure, color=AXIS, lw=0.6))
            fig.text(x, y_top - 0.02 / height, label, ha="left", va="top", fontsize=6.6, color=INK_2)
            fig.text(x, y_top - 0.19 / height, value, ha="left", va="top", fontsize=14, fontweight="semibold", color=INK)
            fig.text(x, y_top - 0.5 / height, sub, ha="left", va="top", fontsize=6.0, color=MUTED)
    return fig


def chart_figure(table: dict, meta: dict, key: str) -> Figure | None:
    """Render a logged custom chart (wandb.plot.* / plot_table) from its saved table."""
    cols, data = table.get("columns", []), table.get("data", [])
    if not cols or not data:
        return None
    idx = {c: i for i, c in enumerate(cols)}
    f = meta.get("fields", {})
    kind = meta.get("kind", "line")

    def col(name):
        i = idx.get(name)
        return [r[i] for r in data] if i is not None else None

    n_rows = len(data)
    if kind == "bar":                                   # one row per class: grow with the class count
        h = max(3.3, 0.24 * n_rows + 1.1)
        fig = Figure(figsize=(5.2, h))
        ax = fig.add_axes([0.3, 0.6 / h, 0.62, 1 - 1.3 / h])
    elif kind == "heatmap":
        n = len({r[idx[f.get("x")]] for r in data}) if f.get("x") in idx else 8
        side = max(4.4, 0.28 * n + 2.2)
        fig = Figure(figsize=(side, side * 0.92))
        ax = fig.add_axes([0.26, 0.24, 0.7, 0.68])
    elif kind == "scatter" and f.get("series"):
        n_series = len({r[idx[f["series"]]] for r in data}) if f["series"] in idx else 1
        if n_series > 14:                               # two legend columns: give them room
            fig = Figure(figsize=(9.2, 4.8))
            ax = fig.add_axes([0.07, 0.11, 0.5, 0.79])
        else:
            fig = Figure(figsize=(6.6, 4.4))
            ax = fig.add_axes([0.1, 0.12, 0.58, 0.78])
    else:
        fig = Figure(figsize=(4.4, 3.3))
        ax = fig.add_axes([0.14, 0.15, 0.82, 0.7])
    title = meta.get("title") or key.rsplit("/", 1)[-1]

    if kind == "line":
        xs, ys = col(f.get("x")), col(f.get("y"))
        if xs is None or ys is None:
            return None
        groups: dict[str, tuple[list, list]] = {}
        series = col(f.get("series")) if f.get("series") else None
        for j, (x, y) in enumerate(zip(xs, ys)):
            g = str(series[j]) if series else ""
            groups.setdefault(g, ([], []))
            groups[g][0].append(x)
            groups[g][1].append(y)
        names = list(groups)
        has_mean = "mean" in names
        classes = [g for g in names if g != "mean"]
        many = len(classes) > 3
        handles, labels = [], []
        for i, g in enumerate(classes):
            gx, gy = groups[g]
            if many:
                ax.plot(gx, gy, color=MUTED, lw=0.6, alpha=0.45)
            else:
                (h,) = ax.plot(gx, gy, color=PALETTE[(i + (1 if has_mean else 0)) % len(PALETTE)], lw=1.2)
                handles.append(h)
                labels.append(g or "series")
        if has_mean:
            gx, gy = groups["mean"]
            (h,) = ax.plot(gx, gy, color=PALETTE[0], lw=2.0)
            handles.insert(0, h)
            labels.insert(0, f"all classes ({len(classes)})" if classes else "mean")
            if many:
                handles.append(Line2D([], [], color=MUTED, lw=0.8))
                labels.append("individual classes")
        if len(handles) > 1:
            ax.legend(handles, labels, loc="best")
        lo = min(min(v for v in xs if v is not None), min(v for v in ys if v is not None))
        hi = max(max(v for v in xs if v is not None), max(v for v in ys if v is not None))
        if lo >= 0 and hi <= 1.0001:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
    elif kind == "scatter":
        xs, ys = col(f.get("x")), col(f.get("y"))
        if f.get("series"):                             # one color per class, legend outside the plot
            from runraccoon.qc import feature_color
            groups: dict = {}
            for g, x, y in zip(col(f["series"]), xs, ys):
                groups.setdefault(str(g), ([], []))
                groups[str(g)][0].append(x)
                groups[str(g)][1].append(y)
            order = sorted(groups, key=lambda g: -len(groups[g][0]))         # legend: most points first
            ids = {str(n): i for i, n in enumerate(f.get("series_order") or [])}  # colors follow class ids
            style = {"colors": {}}
            for i, g in enumerate(order):
                c = ids.get(g, i)
                ax.scatter(*groups[g], s=7, alpha=0.6, linewidths=0, label=f"{g} ({len(groups[g][0])})",
                           color=feature_color(style, f"class:{c}", c))
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, markerscale=2, frameon=False,
                      ncol=1 if len(order) <= 14 else 2, handletextpad=0.3, borderaxespad=0, columnspacing=1.0)
        else:
            ax.scatter(xs, ys, s=14, color=PALETTE[0], alpha=0.75, edgecolors=SURFACE, linewidths=0.6)
    elif kind == "bar":
        labels_, values = col(f.get("label")), col(f.get("value"))
        if labels_ is None or values is None:
            return None
        pos = np.arange(len(values))
        ax.barh(pos, values, height=0.6, color=PALETTE[0])
        ax.set_yticks(pos, [str(v) for v in labels_])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        for p, v in zip(pos, values):
            ax.text(v, p, f" {format_value(v)}", va="center", ha="left", fontsize=6.5, color=INK_2)
    elif kind == "histogram":
        vals = [v for v in col(f.get("value")) or [] if isinstance(v, (int, float))]
        ax.hist(vals, bins=min(40, max(5, int(len(vals) ** 0.5))), color=PALETTE[0], edgecolor=SURFACE, linewidth=0.8)
    elif kind == "heatmap":
        xs, ys, vs = col(f.get("x")), col(f.get("y")), col(f.get("value"))
        xl, yl = list(dict.fromkeys(xs)), list(dict.fromkeys(ys))
        m = np.zeros((len(yl), len(xl)))
        for x, y, v in zip(xs, ys, vs):
            m[yl.index(y), xl.index(x)] += v
        ax.imshow(m, cmap="Blues", aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(xl)), xl, rotation=45 if len(xl) > 6 else 0, ha="right" if len(xl) > 6 else "center")
        ax.set_yticks(range(len(yl)), yl)
        if m.size <= 225:
            thresh = m.max() * 0.55
            for (r, c), v in np.ndenumerate(m):
                ax.text(c, r, format_value(int(v)) if float(v).is_integer() else f"{v:.2f}", ha="center",
                        va="center", fontsize=6.5, color=SURFACE if v > thresh else INK)
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        return None
    ax.set_title(title)
    if meta.get("x_title"):
        ax.set_xlabel(meta["x_title"])
    if meta.get("y_title"):
        ax.set_ylabel(meta["y_title"])
    return fig
